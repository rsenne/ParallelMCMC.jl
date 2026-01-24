using Test
using Random
using LinearAlgebra

using ParallelMCMC
const MALA = ParallelMCMC.MALA
const DEER = ParallelMCMC.DEER

# Standard normal target in R^d
logp_stdnormal(x) = -0.5 * dot(x, x)
gradlogp_stdnormal(x) = -x

# Count MALA rejections explicitly from the tape using the MH indicator.
function count_mala_rejections(logp, gradlogp, x0, ϵ, ξs, us)
    x = copy(x0)
    rejects = 0
    T = length(ξs)
    @assert length(us) == T
    for t in 1:T
        a = MALA.mala_accept_indicator(logp, gradlogp, x, ϵ, ξs[t], us[t])  # 1.0 accept, 0.0 reject
        if a == 0.0
            rejects += 1
            # x unchanged on rejection
        else
            x = MALA.mala_step_taped(logp, gradlogp, x, ϵ, ξs[t], us[t])
        end
    end
    return rejects
end
@testset "MALA Rejections Occur" begin
    rng = MersenneTwister(12345)

    D = 3
    T = 32
    ϵ = 0.35

    x0 = randn(rng, D)
    ξs = [randn(rng, D) for _ in 1:T]
    us = rand(rng, T)

    # Ground truth: sequential taped MALA
    xs_seq = MALA.run_mala_sequential_taped(
        logp_stdnormal, gradlogp_stdnormal, x0, ϵ, ξs, us
    )
    @test length(xs_seq) == T + 1
    # Ensure we actually exercised the rejection (gate a=0) case at least once.
    rejects = count_mala_rejections(logp_stdnormal, gradlogp_stdnormal, x0, ϵ, ξs, us)
    @test rejects > 0
end

@testset "DEER recovers sequential taped MALA trajectory (matrix API)" begin
    rng = MersenneTwister(12345)

    D = 3
    T = 32
    ϵ = 0.05

    x0 = randn(rng, D)
    ξs = [randn(rng, D) for _ in 1:T]
    us = rand(rng, T)

    # Ground truth: sequential taped MALA
    xs_seq = MALA.run_mala_sequential_taped(
        logp_stdnormal, gradlogp_stdnormal, x0, ϵ, ξs, us
    )
    @test length(xs_seq) == T + 1

    tape = [(ξ=ξs[t], u=us[t]) for t in 1:T]
    step_fwd =
        (x, tt) ->
            MALA.mala_step_taped(logp_stdnormal, gradlogp_stdnormal, x, ϵ, tt.ξ, tt.u)

    # Surrogate used ONLY for Jacobians; `a` is passed in as a DI.Constant from consts(...)
    step_lin =
        (x, tt, a) ->
            MALA.mala_step_surrogate(logp_stdnormal, gradlogp_stdnormal, x, ϵ, tt.ξ, a)

    # Compute accept indicator at the linearization point x, and freeze it for AD
    consts =
        (x, tt) -> (
            MALA.mala_accept_indicator(
                logp_stdnormal, gradlogp_stdnormal, x, ϵ, tt.ξ, tt.u
            ),
        )

    rec = DEER.TapedRecursion(step_fwd, step_lin, tape; consts=consts, const_example=(0.0,))

    S_deer, info = DEER.solve(
        rec,
        x0;
        jacobian=:diag,
        damping=0.5,
        tol_abs=1e-10,
        tol_rel=1e-8,
        maxiter=200,
        return_info=true,
    )

    @test info.converged
    @test size(S_deer) == (D, T)

    for t in 1:T
        @test isapprox(view(S_deer, :, t), xs_seq[t + 1]; rtol=1e-6, atol=1e-8)
    end
end

@testset "DEER full Jacobian matches sequential taped MALA (small case, matrix API)" begin
    rng = MersenneTwister(2024)

    D = 2
    T = 10
    ϵ = 0.03

    x0 = randn(rng, D)
    ξs = [randn(rng, D) for _ in 1:T]
    us = rand(rng, T)

    xs_seq = MALA.run_mala_sequential_taped(
        logp_stdnormal, gradlogp_stdnormal, x0, ϵ, ξs, us
    )

    tape = [(ξ=ξs[t], u=us[t]) for t in 1:T]

    step_fwd =
        (x, tt) ->
            MALA.mala_step_taped(logp_stdnormal, gradlogp_stdnormal, x, ϵ, tt.ξ, tt.u)
    step_lin =
        (x, tt, a) ->
            MALA.mala_step_surrogate(logp_stdnormal, gradlogp_stdnormal, x, ϵ, tt.ξ, a)
    consts =
        (x, tt) -> (
            MALA.mala_accept_indicator(
                logp_stdnormal, gradlogp_stdnormal, x, ϵ, tt.ξ, tt.u
            ),
        )
    rec = DEER.TapedRecursion(step_fwd, step_lin, tape; consts=consts, const_example=(0.0,))

    S_deer, info = DEER.solve(
        rec,
        x0;
        jacobian=:full,
        damping=0.7,
        tol_abs=1e-10,
        tol_rel=1e-8,
        maxiter=200,
        return_info=true,
    )

    @test info.converged
    @test size(S_deer) == (D, T)

    for t in 1:T
        @test isapprox(view(S_deer, :, t), xs_seq[t + 1]; rtol=1e-6, atol=1e-8)
    end
end

@testset "affine scan diag matches sequential recursion (matrix API)" begin
    rng = MersenneTwister(1)
    D = 5
    T = 64
    s0 = randn(rng, D)

    # Use matrices A,B with shape D×T
    A = Matrix{Float64}(undef, D, T)
    B = Matrix{Float64}(undef, D, T)
    for t in 1:T
        A[:, t] .= 0.9 .+ 0.02 .* randn(rng, D) # stable-ish
        B[:, t] .= 0.1 .* randn(rng, D)
    end

    # Sequential reference: s_t = A[:,t].*s_{t-1} + B[:,t]
    S_ref = Matrix{Float64}(undef, D, T)
    s_prev = copy(s0)
    for t in 1:T
        s_prev = view(A, :, t) .* s_prev .+ view(B, :, t)
        @inbounds S_ref[:, t] .= s_prev
    end

    # Scan
    S_scan = DEER.solve_affine_scan_diag(A, B, s0)

    @test size(S_scan) == (D, T)
    @test isapprox(S_scan, S_ref; rtol=1e-12, atol=1e-12)
end

@testset "DEER stochastic diagonal (JVP/Hutchinson) works (matrix API)" begin
    rng = MersenneTwister(777)

    D = 3
    T = 32
    ϵ = 0.15  # slightly larger to ensure we see some rejections and make gating relevant

    x0 = randn(rng, D)
    ξs = [randn(rng, D) for _ in 1:T]
    us = rand(rng, T)

    # Ground truth: sequential taped MALA
    xs_seq = MALA.run_mala_sequential_taped(
        logp_stdnormal, gradlogp_stdnormal, x0, ϵ, ξs, us
    )
    @test length(xs_seq) == T + 1

    # Ensure at least one rejection occurred (so gating is exercised)
    n_reject = count(t -> xs_seq[t + 1] == xs_seq[t], 1:T)
    @test n_reject > 0

    tape = [(ξ=ξs[t], u=us[t]) for t in 1:T]
    step_fwd =
        (x, tt) ->
            MALA.mala_step_taped(logp_stdnormal, gradlogp_stdnormal, x, ϵ, tt.ξ, tt.u)
    step_lin =
        (x, tt, a) ->
            MALA.mala_step_surrogate(logp_stdnormal, gradlogp_stdnormal, x, ϵ, tt.ξ, a)
    consts =
        (x, tt) -> (
            MALA.mala_accept_indicator(
                logp_stdnormal, gradlogp_stdnormal, x, ϵ, tt.ξ, tt.u
            ),
        )

    rec = DEER.TapedRecursion(step_fwd, step_lin, tape; consts=consts, const_example=(0.0,))

    S_deer, info = DEER.solve(
        rec,
        x0;
        jacobian=:stoch_diag,
        probes=2,
        rng=MersenneTwister(42),  # deterministic probes
        damping=0.5,
        tol_abs=1e-9,
        tol_rel=1e-7,
        maxiter=400,
        return_info=true,
    )

    @test info.converged
    @test size(S_deer) == (D, T)

    for t in 1:T
        @test isapprox(view(S_deer, :, t), xs_seq[t + 1]; rtol=1e-5, atol=1e-7)
    end

    # Prepare both preps explicitly
    prepJ = DEER.prepare(rec, x0)
    prepPF = DEER.prepare_pushforward(rec, x0)

    tcheck = 5
    xbar = xs_seq[tcheck]  # state before step tcheck (xs_seq[t] is after step t-1)

    diag_exact = DEER.jac_diag(rec, prepJ, xbar, tcheck)

    diag_est = DEER.jac_diag_stoch(
        rec, prepPF, xbar, tcheck; probes=200, rng=MersenneTwister(123), zbuf=zeros(D)
    )

    # The estimator is Monte Carlo; allow a modest tolerance.
    @test isapprox(diag_est, diag_exact; rtol=0.15, atol=0.15)
end
