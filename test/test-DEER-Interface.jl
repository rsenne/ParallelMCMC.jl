using Test
using Random
using LinearAlgebra
using Statistics
using MCMCChains

using ParallelMCMC

logp_deer(x) = -0.5 * dot(x, x)
gradlogp_deer(x) = -x

@testset "DEERSampler construction" begin
    s = DEERSampler(0.05)
    @test s isa ParallelMCMC.AbstractMCMC.AbstractSampler
    @test s.epsilon == 0.05
    @test s.T == 64
    @test s.maxiter == 200
    @test s.jacobian === :diag

    # keyword overrides
    s2 = DEERSampler(0.1; T=32, jacobian=:stoch_diag, damping=0.8)
    @test s2.T == 32
    @test s2.jacobian === :stoch_diag
    @test s2.damping == 0.8
end

@testset "DEERSampler initial step" begin
    rng = MersenneTwister(42)
    model = DensityModel(logp_deer, gradlogp_deer, 3)
    sampler = DEERSampler(0.05; T=16)

    trans, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler)

    @test trans isa DEERTransition
    @test state isa DEERState
    @test length(trans.x) == 3
    @test isfinite(trans.logp)
    @test trans.logp ≈ logp_deer(trans.x)
    @test state.t == 1
    @test size(state.trajectory) == (3, 16)
    @test length(state.tape) == 16
end

@testset "DEERSampler initial step respects initial_params" begin
    rng = MersenneTwister(42)
    model = DensityModel(logp_deer, gradlogp_deer, 3)
    sampler = DEERSampler(0.05; T=8)
    x0 = [1.0, 2.0, 3.0]
    x0_copy = copy(x0)

    trans, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler; initial_params=x0)

    # initial_params not mutated
    @test x0 == x0_copy
    # first sample should differ from x0 (DEER solves the trajectory)
    @test length(trans.x) == 3
    @test isfinite(trans.logp)
end

@testset "DEERSampler sequential steps advance trajectory index" begin
    rng = MersenneTwister(7)
    model = DensityModel(logp_deer, gradlogp_deer, 2)
    sampler = DEERSampler(0.05; T=8)

    trans, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler)
    @test state.t == 1

    trans2, state2 = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
    @test state2.t == 2
    @test trans2.x ≈ state.trajectory[:, 2]

    trans3, state3 = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state2)
    @test state3.t == 3
end

@testset "DEERSampler re-solves at trajectory boundary" begin
    rng = MersenneTwister(99)
    model = DensityModel(logp_deer, gradlogp_deer, 2)
    T = 4
    sampler = DEERSampler(0.05; T=T)

    _, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler)

    # advance to t == T
    for _ in 1:(T - 1)
        _, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
    end
    @test state.t == T

    # next step should trigger re-solve: t resets to 1 with a new trajectory
    _, state_new = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
    @test state_new.t == 1
    # new trajectory should differ from old (different tape)
    @test state_new.trajectory !== state.trajectory
end

@testset "DEERSampler sample() end-to-end" begin
    model = DensityModel(logp_deer, gradlogp_deer, 2)
    sampler = DEERSampler(0.05; T=16)

    samples = sample(MersenneTwister(1), model, sampler, 50; progress=false)
    @test length(samples) == 50
end

@testset "DEERSampler sample() with chain_type=Chains" begin
    model = DensityModel(logp_deer, gradlogp_deer, 2)
    sampler = DEERSampler(0.05; T=16)

    chain = sample(
        MersenneTwister(1),
        model,
        sampler,
        100;
        chain_type=MCMCChains.Chains,
        progress=false,
    )

    @test chain isa MCMCChains.Chains
    @test size(chain, 1) == 100
    @test :logp in names(chain, :internals)
    @test all(isfinite, chain[:logp])

    param_names = names(chain, :parameters)
    @test length(param_names) == 2
end

@testset "DEERSampler sample() with custom param_names" begin
    model = DensityModel(logp_deer, gradlogp_deer, 2)
    sampler = DEERSampler(0.05; T=16)

    chain = sample(
        MersenneTwister(2),
        model,
        sampler,
        40;
        chain_type=MCMCChains.Chains,
        progress=false,
        param_names=[:mu, :sigma],
    )

    @test :mu in names(chain, :parameters)
    @test :sigma in names(chain, :parameters)
end

@testset "DEERSampler stationary distribution" begin
    D = 3
    model = DensityModel(logp_deer, gradlogp_deer, D)
    sampler = DEERSampler(0.1; T=32, damping=0.5)

    chain = sample(
        MersenneTwister(2025),
        model,
        sampler,
        5_000;
        chain_type=MCMCChains.Chains,
        progress=false,
    )

    burn = 500
    post = Array(chain[burn:end, :, :])  # (N-burn) × D

    mu = vec(mean(post; dims=1))
    vars = vec(var(post; dims=1))

    @test maximum(abs.(mu)) < 0.15
    @test maximum(abs.(vars .- 1.0)) < 0.25
end

@testset "DEERSampler parallel chains via MCMCThreads" begin
    model = DensityModel(logp_deer, gradlogp_deer, 2)
    sampler = DEERSampler(0.05; T=8)

    chains = sample(
        MersenneTwister(42),
        model,
        sampler,
        ParallelMCMC.AbstractMCMC.MCMCThreads(),
        40,
        2;
        chain_type=MCMCChains.Chains,
        progress=false,
    )

    @test chains isa MCMCChains.Chains
    @test size(chains, 1) == 40   # samples per chain
    @test size(chains, 3) == 2    # number of chains
end
