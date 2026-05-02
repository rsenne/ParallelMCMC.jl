using Test
using Random
using LinearAlgebra
using Statistics
using MCMCChains

using ParallelMCMC
const MALA_iface = ParallelMCMC.MALA

logp_iface(x) = -0.5 * dot(x, x)
gradlogp_iface(x) = -x

@testset "AbstractMCMC interface" begin
    @testset "DensityModel construction" begin
        m = DensityModel(logp_iface, gradlogp_iface, 3)
        @test m isa ParallelMCMC.AbstractMCMC.AbstractModel
        @test m.dim == 3
        @test m.logdensity([0.0, 0.0, 0.0]) == 0.0
        @test m.grad_logdensity([1.0, 2.0, 3.0]) == [-1.0, -2.0, -3.0]
    end

    @testset "MALASampler construction" begin
        s = MALASampler(0.1)
        @test s isa ParallelMCMC.AbstractMCMC.AbstractSampler
        @test s.epsilon == 0.1
    end

    @testset "initial step draws from rng" begin
        rng = MersenneTwister(42)
        model = DensityModel(logp_iface, gradlogp_iface, 5)
        sampler = MALASampler(0.1)

        transition, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler)

        @test transition isa MALATransition
        @test state isa MALAState
        @test length(transition.x) == 5
        @test length(state.x) == 5
        @test transition.logp == logp_iface(transition.x)
        @test state.logp == transition.logp
        @test transition.accepted == true
        @test state.workspace isa MALA_iface.MALAWorkspace
    end

    @testset "initial step respects initial_params" begin
        rng = MersenneTwister(42)
        model = DensityModel(logp_iface, gradlogp_iface, 3)
        sampler = MALASampler(0.1)
        x0 = [1.0, 2.0, 3.0]

        transition, state = ParallelMCMC.AbstractMCMC.step(
            rng, model, sampler; initial_params=x0
        )

        @test transition.x == x0
        @test state.x == x0
        @test transition.logp == logp_iface(x0)
    end

    @testset "initial step does not mutate initial_params" begin
        rng = MersenneTwister(42)
        model = DensityModel(logp_iface, gradlogp_iface, 3)
        sampler = MALASampler(0.1)
        x0 = [1.0, 2.0, 3.0]
        x0_copy = copy(x0)

        ParallelMCMC.AbstractMCMC.step(rng, model, sampler; initial_params=x0)
        @test x0 == x0_copy
    end

    @testset "subsequent step produces valid transition" begin
        rng = MersenneTwister(42)
        model = DensityModel(logp_iface, gradlogp_iface, 4)
        sampler = MALASampler(0.1)

        t1, s1 = ParallelMCMC.AbstractMCMC.step(rng, model, sampler)
        t2, s2 = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, s1)

        @test t2 isa MALATransition
        @test s2 isa MALAState
        @test length(t2.x) == 4
        @test t2.accepted isa Bool
        @test isfinite(t2.logp)
        @test s2.x == t2.x
        @test s2.logp == t2.logp
        @test s2.workspace === s1.workspace
    end

    @testset "step determinism with fixed rng" begin
        model = DensityModel(logp_iface, gradlogp_iface, 3)
        sampler = MALASampler(0.2)

        rng1 = MersenneTwister(999)
        t1a, s1a = ParallelMCMC.AbstractMCMC.step(rng1, model, sampler)
        t1b, s1b = ParallelMCMC.AbstractMCMC.step(rng1, model, sampler, s1a)

        rng2 = MersenneTwister(999)
        t2a, s2a = ParallelMCMC.AbstractMCMC.step(rng2, model, sampler)
        t2b, s2b = ParallelMCMC.AbstractMCMC.step(rng2, model, sampler, s2a)

        @test t1a.x == t2a.x
        @test t1b.x == t2b.x
        @test t1b.accepted == t2b.accepted
    end

    @testset "rejection preserves logp from state" begin
        model = DensityModel(logp_iface, gradlogp_iface, 3)
        sampler = MALASampler(0.5)

        # Run enough steps to hit a rejection
        rng = MersenneTwister(12345)
        _, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler)

        found_rejection = false
        for _ in 1:500
            t, state_new = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
            if !t.accepted
                # On rejection, x doesn't move and logp should match prior state
                @test t.x == state.x
                @test t.logp == state.logp
                found_rejection = true
                break
            end
            state = state_new
        end
        @test found_rejection
    end

    @testset "sample() runs end-to-end" begin
        model = DensityModel(logp_iface, gradlogp_iface, 3)
        sampler = MALASampler(0.1)

        chain = sample(model, sampler, 100; progress=false)
        @test length(chain) == 100
    end

    @testset "sample() with chain_type=Chains" begin
        model = DensityModel(logp_iface, gradlogp_iface, 2)
        sampler = MALASampler(0.15)

        chain = sample(model, sampler, 200; chain_type=MCMCChains.Chains, progress=false)

        @test chain isa MCMCChains.Chains
        @test size(chain, 1) == 200

        # Parameter columns present
        param_names = names(chain, :parameters)
        @test length(param_names) == 2

        # Internal columns present
        internal_names = names(chain, :internals)
        @test :logp in internal_names
        @test :accepted in internal_names

        # logp values should be finite
        @test all(isfinite, chain[:logp])

        # accepted values should be 0 or 1
        acc = chain[:accepted]
        @test all(a -> a == 0.0 || a == 1.0, acc)
    end

    @testset "sample() with custom param_names" begin
        model = DensityModel(logp_iface, gradlogp_iface, 2)
        sampler = MALASampler(0.15)

        chain = sample(
            model,
            sampler,
            50;
            chain_type=MCMCChains.Chains,
            progress=false,
            param_names=[:mu, :sigma],
        )

        @test chain isa MCMCChains.Chains
        param_names = names(chain, :parameters)
        @test :mu in param_names
        @test :sigma in param_names
    end

    @testset "stationary distribution via sample()" begin
        D = 3
        model = DensityModel(logp_iface, gradlogp_iface, D)
        sampler = MALASampler(0.3)

        chain = sample(
            MersenneTwister(2025),
            model,
            sampler,
            20_000;
            chain_type=MCMCChains.Chains,
            progress=false,
        )

        burn = 3_000
        post = Array(chain[burn:end, :, :])  # (N-burn) × D

        mu = vec(mean(post; dims=1))
        @test maximum(abs.(mu)) < 0.1

        vars = vec(var(post; dims=1))
        @test maximum(abs.(vars .- 1.0)) < 0.15
    end
end
