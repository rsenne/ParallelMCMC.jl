using Test
using Random
using LinearAlgebra

using ParallelMCMC
isdefined(@__MODULE__, :TestUtils) || include("utils.jl")
using .TestUtils
const MALA = ParallelMCMC.MALA
const DEER = ParallelMCMC.DEER

logp_jd(x) = -0.5 * dot(x, x)
gradlogp_jd(x) = -x
hvp_jd(x, v) = -v
logp_batch_jd(X) = vec(-0.5 .* sum(abs2, X; dims=1))
gradlogp_batch_jd(X) = -X
hvp_batch_jd(X, V) = -V

function jd_mala_problem(seed, D, T, ε)
    rng = MersenneTwister(seed)
    x0 = randn(rng, D)
    ξs = [randn(rng, D) for _ in 1:T]
    us = rand(rng, T)
    xs = run_mala_sequential_taped(logp_jd, gradlogp_jd, x0, ε, ξs, us)
    tape = [(ξ=ξs[t], u=us[t]) for t in 1:T]
    step_fwd = (x, tt) -> MALA.mala_step_taped(logp_jd, gradlogp_jd, x, ε, tt.ξ, tt.u)
    jvp =
        (x, tt, v) -> MALA.mala_step_surrogate_sigmoid_jvp(
            logp_jd, gradlogp_jd, x, ε, tt.ξ, tt.u, v, hvp_jd
        )
    rec = DEER.TapedRecursion(step_fwd, jvp, tape)
    return x0, rec, reduce(hcat, xs[2:end])
end

@testset "linearization uses the damped, clipped Jacobian on every path" begin
    rng = MersenneTwister(20261103)
    D, T = 5, 9
    Atrue = 1.5 .* randn(rng, D, T)
    Btrue = randn(rng, D, T)
    s0 = randn(rng, D)
    S_in = randn(rng, D, T)
    Xbar = hcat(s0, S_in[:, 1:(T - 1)])
    k, c = 0.5, 0.6
    A_expected = clamp.(k .* Atrue, -c, c)
    B_expected = Atrue .* Xbar .+ Btrue .- A_expected .* Xbar

    step_fwd = (x, t) -> view(Atrue, :, t) .* x .+ view(Btrue, :, t)
    jvp = (x, t, v) -> view(Atrue, :, t) .* v
    batch = (X, Z) -> (Atrue .* X .+ Btrue, Atrue .* Z)
    recs = (
        (:diag, DEER.TapedRecursion(step_fwd, jvp, collect(1:T))),
        (:stoch_diag, DEER.TapedRecursion(step_fwd, jvp, collect(1:T))),
        (
            :stoch_diag,
            DEER.TapedRecursion(step_fwd, jvp, collect(1:T); fwd_and_jvp_batch=batch),
        ),
    )
    for (jacobian, rec) in recs
        ws = DEER.DEERWorkspace(S_in, s0)
        S_out = similar(S_in)
        DEER.deer_update!(
            ws, S_out, rec, s0, S_in; jacobian=jacobian, jacobian_damping=k, jacobian_clip=c
        )
        @test ws.A ≈ A_expected atol = 1e-12
        @test ws.B ≈ B_expected atol = 1e-12
        @test S_out ≈ DEER.solve_affine_seq(A_expected, B_expected, s0) atol = 1e-10
    end

    ws = DEER.DEERWorkspace(S_in, s0)
    rec = last(recs[1])
    S_out = similar(S_in)
    @test_throws ArgumentError DEER.deer_update!(
        ws, S_out, rec, s0, S_in; jacobian_damping=0.0
    )
    @test_throws ArgumentError DEER.deer_update!(
        ws, S_out, rec, s0, S_in; jacobian_damping=1.5
    )
    @test_throws ArgumentError DEER.deer_update!(ws, S_out, rec, s0, S_in; jacobian_clip=0)
    @test_throws ArgumentError ParallelMALASampler(0.1; jacobian_damping=-0.5)
    @test_throws ArgumentError ParallelMALASampler(0.1; jacobian_clip=-1)
end

@testset "default Jacobian options leave DEER unchanged" begin
    x0, rec, _ = jd_mala_problem(5, 3, 32, 0.35)
    for jacobian in (:diag, :stoch_diag)
        S_a = DEER.solve(rec, x0; jacobian=jacobian, rng=MersenneTwister(2))
        S_b = DEER.solve(
            rec,
            x0;
            jacobian=jacobian,
            rng=MersenneTwister(2),
            jacobian_damping=1,
            jacobian_clip=Inf,
        )
        @test S_a == S_b
    end

    s = ParallelMALASampler(0.1)
    @test s.jacobian_damping == 1
    @test s.jacobian_clip == 1
    s2 = ParallelMALASampler(0.1; jacobian_damping=0.5, jacobian_clip=Inf)
    @test s2.jacobian_damping == 0.5
    @test s2.jacobian_clip == Inf
end

@testset "fixed point is the sequential MALA trajectory" begin
    options = (
        (jacobian_damping=0.5,),
        (jacobian_clip=1.0,),
        (jacobian_damping=0.4, jacobian_clip=1.0),
    )
    x0, rec, S_ref = jd_mala_problem(12345, 3, 32, 0.35)
    for jacobian in (:diag, :stoch_diag), opts in options
        S, info = DEER.solve(
            rec,
            x0;
            jacobian=jacobian,
            tol_abs=1e-10,
            tol_rel=1e-8,
            maxiter=200,
            rng=MersenneTwister(1),
            return_info=true,
            opts...,
        )
        @test info.converged
        @test S ≈ S_ref rtol = 1e-6 atol = 1e-8
    end
end

@testset "damping and clipping converge where undamped DEER does not" begin
    #=
    Standard normal with ε = 1.5: about a third of the diagonal Jacobian entries
    exceed 1 in magnitude, so the linear recursion amplifies errors over the
    1024 steps. DEER's iterates wander to |x| in the hundreds and it needs ~390
    iterations; with k = 0.5 it takes ~100, with clipping ~170.
    =#
    x0, rec, S_ref = jd_mala_problem(1, 4, 1024, 1.5)
    kw = (jacobian=:diag, tol_abs=1e-10, tol_rel=1e-8, maxiter=200, return_info=true)

    _, info = @test_logs (:warn, r"did not converge in `maxiter = 200`") DEER.solve(
        rec, x0; kw...
    )
    @test !info.converged
    @test_logs DEER.solve(rec, x0; kw..., warn=false)

    for opts in (
        (jacobian_damping=0.5,),
        (jacobian_clip=1.0,),
        (jacobian_damping=0.5, jacobian_clip=1.0),
    )
        S, info = DEER.solve(rec, x0; kw..., opts...)
        @test info.converged
        @test S ≈ S_ref rtol = 1e-6 atol = 1e-6
    end
end

@testset "ParallelMALASampler warns once when DEER blocks do not converge" begin
    model = DensityModel(logp_jd, gradlogp_jd, 3; hvp=hvp_jd)
    x0 = [0.5, -1.0, 2.0]
    run(sampler) = sample(
        MersenneTwister(9),
        model,
        sampler,
        40;
        initial_params=x0,
        progress=false,
        chain_type=Any,
    )
    # 40 samples with T = 16 take three solves, and one iteration converges none of them.
    sampler = ParallelMALASampler(0.3; T=16, maxiter=1, tol_abs=1e-12, tol_rel=1e-10)
    @test_logs (:warn, r"for 3 of 3 trajectory blocks") run(sampler)
    @test_logs run(ParallelMALASampler(0.3; T=16, maxiter=200))
end

@testset "ParallelMALASampler with Jacobian damping samples the same chain" begin
    model = DensityModel(logp_jd, gradlogp_jd, 3; hvp=hvp_jd)
    kw = (T=16, jacobian=:diag, tol_abs=1e-12, tol_rel=1e-10)
    x0 = [0.5, -1.0, 2.0]

    run(sampler) = sample(
        MersenneTwister(9),
        model,
        sampler,
        40;
        initial_params=x0,
        progress=false,
        chain_type=Any,
    )
    chain_deer = run(ParallelMALASampler(0.3; jacobian_clip=Inf, kw...))
    chain_jd = run(ParallelMALASampler(0.3; jacobian_damping=0.5, jacobian_clip=1.0, kw...))
    for i in 1:40
        @test chain_jd[i].x ≈ chain_deer[i].x rtol = 1e-6 atol = 1e-8
    end
end

@testset "ParallelMALASampler with Jacobian damping on the batched path" begin
    model = DensityModel(
        logp_jd,
        gradlogp_jd,
        3;
        hvp=hvp_jd,
        logdensity_batch=logp_batch_jd,
        grad_logdensity_batch=gradlogp_batch_jd,
        hvp_batch=hvp_batch_jd,
    )
    kw = (T=32, tol_abs=1e-12, tol_rel=1e-10)
    x0 = [0.5, -1.0, 2.0]

    # The tape is drawn before the solve, so the first block shares it.
    _, state_deer = ParallelMCMC.AbstractMCMC.step(
        MersenneTwister(9),
        model,
        ParallelMALASampler(0.3; jacobian_clip=Inf, kw...);
        initial_params=x0,
    )
    _, state_jd = ParallelMCMC.AbstractMCMC.step(
        MersenneTwister(9),
        model,
        ParallelMALASampler(0.3; jacobian_damping=0.5, jacobian_clip=1.0, kw...);
        initial_params=x0,
    )
    @test state_jd.trajectory ≈ state_deer.trajectory rtol = 1e-6 atol = 1e-8
end
