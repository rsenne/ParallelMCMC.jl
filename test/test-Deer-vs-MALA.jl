using Test
using Random
using LinearAlgebra

const MALA = ParallelMCMC.MALA
const DEER = ParallelMCMC.DEER

# Standard normal target in R^d
# logp(x) = -1/2 ||x||^2  (up to constant), gradlogp(x) = -x
logp_stdnormal(x) = -0.5 * dot(x, x)
gradlogp_stdnormal(x) = -x

@testset "DEER recovers sequential taped MALA trajectory" begin
    rng = MersenneTwister(12345)

    D = 3
    T = 32
    ϵ = 0.05

    x0 = randn(rng, D)

    ξs = [randn(rng, D) for _ in 1:T]
    us = rand(rng, T)                 # in (0,1)

    # Ground truth: sequential taped MALA
    xs_seq = MALA.run_mala_sequential_taped(
        logp_stdnormal,
        gradlogp_stdnormal,
        x0,
        ϵ,
        ξs,
        us,
    )
    @test length(xs_seq) == T + 1

    # Build the taped recursion for DEER
    tape = [(ξ = ξs[t], u = us[t]) for t in 1:T]
    step = (x, tt) -> MALA.mala_step_taped(
        logp_stdnormal,
        gradlogp_stdnormal,
        x,
        ϵ,
        tt.ξ,
        tt.u,
    )
    rec = DEER.TapedRecursion(step, tape)

    # Solve with quasi-DEER (diagonal Jacobian)
    s_deer, info = DEER.solve(
        rec,
        x0;
        jacobian = :diag,
        damping = 0.5,        # damping helps robustness
        tol_abs = 1e-10,
        tol_rel = 1e-8,
        maxiter = 200,
        return_info = true,
    )

    @test info.converged
    @test length(s_deer) == T

    # Compare DEER states to sequential states (xs_seq[2:end])
    for t in 1:T
        @test isapprox(s_deer[t], xs_seq[t+1]; rtol = 1e-6, atol = 1e-8)
    end
end

@testset "DEER full Jacobian matches sequential taped MALA (small case)" begin
    rng = MersenneTwister(2024)

    D = 2
    T = 10
    ϵ = 0.03

    x0 = randn(rng, D)
    ξs = [randn(rng, D) for _ in 1:T]
    us = rand(rng, T)

    xs_seq = MALA.run_mala_sequential_taped(
        logp_stdnormal,
        gradlogp_stdnormal,
        x0,
        ϵ,
        ξs,
        us,
    )

    tape = [(ξ = ξs[t], u = us[t]) for t in 1:T]
    step = (x, tt) -> MALA.mala_step_taped(
        logp_stdnormal,
        gradlogp_stdnormal,
        x,
        ϵ,
        tt.ξ,
        tt.u,
    )
    rec = DEER.TapedRecursion(step, tape)

    s_deer, info = DEER.solve(
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
    @test length(s_deer) == T

    for t in 1:T
        @test isapprox(s_deer[t], xs_seq[t+1]; rtol = 1e-6, atol = 1e-8)
    end
end
