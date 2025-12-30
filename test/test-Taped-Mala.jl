using Test
using Random, LinearAlgebra
using ParallelMCMC

# A simple target: standard Normal in d dims
# logp(x) = -0.5*||x||^2  (ignoring constant)
logp(x) = -0.5 * dot(x, x)
gradlogp(x) = -x

@testset "taped MALA determinism" begin
    rng = MersenneTwister(123)
    d = 5
    T = 50
    ϵ = 0.1

    x0 = randn(rng, d)
    ξs = [randn(rng, d) for _ in 1:T]
    us = rand(rng, T)

    xs1 = ParallelMCMC.MALA.run_mala_sequential_taped(logp, gradlogp, x0, ϵ, ξs, us)
    xs2 = ParallelMCMC.MALA.run_mala_sequential_taped(logp, gradlogp, x0, ϵ, ξs, us)

    @test length(xs1) == T + 1
    @test all(xs1[i] == xs2[i] for i in 1:T)  # exact equality should hold
end

@testset "taped MALA changes with tape" begin
    rng = MersenneTwister(123)
    d = 3
    T = 20
    ϵ = 0.1

    x0 = randn(rng, d)
    ξs1 = [randn(rng, d) for _ in 1:T]
    us1 = rand(rng, T)

    rng2 = MersenneTwister(456)
    ξs2 = [randn(rng2, d) for _ in 1:T]
    us2 = rand(rng2, T)

    xs1 = ParallelMCMC.MALA.run_mala_sequential_taped(logp, gradlogp, x0, ϵ, ξs1, us1)
    xs2 = ParallelMCMC.MALA.run_mala_sequential_taped(logp, gradlogp, x0, ϵ, ξs2, us2)

    @test any(xs1[i] != xs2[i] for i in 2:T)
end
