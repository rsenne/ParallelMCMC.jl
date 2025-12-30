using Test
using Random
using LinearAlgebra

using ParallelMCMC
const MALA = ParallelMCMC.MALA
const DEER = ParallelMCMC.DEER

# Standard normal target in R^d
logp_stdnormal(x) = -0.5 * dot(x, x)
gradlogp_stdnormal(x) = -x

@testset "DEER recovers sequential taped MALA trajectory (matrix API)" begin
    rng = MersenneTwister(12345)

    D = 3
    T = 32
    ϵ = 0.05

    x0 = randn(rng, D)
    ξs = [randn(rng, D) for _ in 1:T]
    us = rand(rng, T)

    # Ground truth: sequential taped MALA
    xs_seq = MALA.run_mala_sequential_taped(logp_stdnormal, gradlogp_stdnormal, x0, ϵ, ξs, us)
    @test length(xs_seq) == T + 1

    tape = [(ξ = ξs[t], u = us[t]) for t in 1:T]
    step = (x, tt) -> MALA.mala_step_taped(logp_stdnormal, gradlogp_stdnormal, x, ϵ, tt.ξ, tt.u)
    rec = DEER.TapedRecursion(step, tape)

    S_deer, info = DEER.solve(
        rec,
        x0;
        jacobian = :diag,
        damping = 0.5,
        tol_abs = 1e-10,
        tol_rel = 1e-8,
        maxiter = 200,
        return_info = true,
    )

    @test info.converged
    @test size(S_deer) == (D, T)

    for t in 1:T
        @test isapprox(view(S_deer, :, t), xs_seq[t + 1]; rtol = 1e-6, atol = 1e-8)
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

    xs_seq = MALA.run_mala_sequential_taped(logp_stdnormal, gradlogp_stdnormal, x0, ϵ, ξs, us)

    tape = [(ξ = ξs[t], u = us[t]) for t in 1:T]
    step = (x, tt) -> MALA.mala_step_taped(logp_stdnormal, gradlogp_stdnormal, x, ϵ, tt.ξ, tt.u)
    rec = DEER.TapedRecursion(step, tape)

    S_deer, info = DEER.solve(
        rec,
        x0;
        jacobian = :full,
        damping = 0.7,
        tol_abs = 1e-10,
        tol_rel = 1e-8,
        maxiter = 200,
        return_info = true,
    )

    @test info.converged
    @test size(S_deer) == (D, T)

    for t in 1:T
        @test isapprox(view(S_deer, :, t), xs_seq[t + 1]; rtol = 1e-6, atol = 1e-8)
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
    @test isapprox(S_scan, S_ref; rtol = 1e-12, atol = 1e-12)
end
