using Test
using Random
using LinearAlgebra
using Statistics
using FlexiChains

using ParallelMCMC
const MALA = ParallelMCMC.MALA

logp_adapt(x) = -0.5 * dot(x, x)
gradlogp_adapt(x) = -x

@testset "mala_step_with_logα returns same x_next and accepted as mala_step_full" begin
    rng = MersenneTwister(1)
    D = 4
    for _ in 1:20
        x = randn(rng, D)
        ξ = randn(rng, D)
        u = rand(rng) * 0.999 + 1e-15

        x1, a1 = MALA.mala_step_full(logp_adapt, gradlogp_adapt, x, 0.1, ξ, u)
        x2, a2, logα = MALA.mala_step_with_logα(logp_adapt, gradlogp_adapt, x, 0.1, ξ, u)

        @test x1 == x2
        @test a1 == a2
        @test isfinite(logα)
        # When accepted logα ≥ log(u), when rejected logα < log(u)
        @test a2 == (log(u) < logα)
    end
end

@testset "mala_step_with_logα! matches allocating wrapper" begin
    rng = MersenneTwister(2)
    D = 4
    x = randn(rng, D)
    ξ = randn(rng, D)
    u = rand(rng)

    x_ref, accepted_ref, logα_ref = MALA.mala_step_with_logα(
        logp_adapt, gradlogp_adapt, x, 0.1, ξ, u
    )

    ws = MALA.MALAWorkspace(x)
    x_next = similar(x)
    x_out, accepted, logα = MALA.mala_step_with_logα!(
        x_next, ws, logp_adapt, gradlogp_adapt, x, 0.1, ξ, u
    )

    @test x_out === x_next
    @test x_next == x_ref
    @test accepted == accepted_ref
    @test logα ≈ logα_ref
end

@testset "AdaptiveMALASampler construction" begin
    s = AdaptiveMALASampler(0.1)
    @test s isa ParallelMCMC.AbstractMCMC.AbstractSampler
    @test s.epsilon_init == 0.1
    @test s.n_warmup == 1000
    @test s.target_accept ≈ 0.574
    @test s.gamma ≈ 0.05
    @test s.t0 ≈ 10.0
    @test s.kappa ≈ 0.75
    @test s.cholM === nothing

    s2 = AdaptiveMALASampler(0.05; n_warmup=500, target_accept=0.65)
    @test s2.n_warmup == 500
    @test s2.target_accept ≈ 0.65
end

@testset "AdaptiveMALASampler initial step" begin
    rng = MersenneTwister(42)
    model = DensityModel(logp_adapt, gradlogp_adapt, 3)
    sampler = AdaptiveMALASampler(0.1)

    trans, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler)

    @test trans isa AdaptiveMALATransition
    @test state isa AdaptiveMALAState
    @test length(trans.x) == 3
    @test isfinite(trans.logp)
    @test trans.step_size == sampler.epsilon_init
    @test trans.is_warmup == true
    @test state.step == 0
    @test state.epsilon == sampler.epsilon_init
    @test state.epsilon_bar == sampler.epsilon_init
    @test state.workspace isa MALA.MALAWorkspace
end

@testset "AdaptiveMALASampler initial step respects initial_params" begin
    rng = MersenneTwister(7)
    model = DensityModel(logp_adapt, gradlogp_adapt, 2)
    sampler = AdaptiveMALASampler(0.1)
    x0 = [1.0, -1.0]
    x0_copy = copy(x0)

    trans, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler; initial_params=x0)

    @test x0 == x0_copy   # not mutated
    @test trans.x == x0
end

@testset "step counter increments during warmup" begin
    rng = MersenneTwister(5)
    model = DensityModel(logp_adapt, gradlogp_adapt, 2)
    sampler = AdaptiveMALASampler(0.05; n_warmup=10)

    _, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler)

    for expected_step in 1:5
        _, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
        @test state.step == expected_step
        @test state.step <= sampler.n_warmup  # still in warmup
    end
end

@testset "step size changes during warmup but freezes after" begin
    rng = MersenneTwister(11)
    model = DensityModel(logp_adapt, gradlogp_adapt, 3)
    n_w = 20
    sampler = AdaptiveMALASampler(0.1; n_warmup=n_w)

    _, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler)

    epsilons_warmup = Float64[]
    for _ in 1:n_w
        t, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
        push!(epsilons_warmup, t.step_size)
        @test t.is_warmup == true
    end

    # At least some epsilon change during warmup
    @test !all(==(epsilons_warmup[1]), epsilons_warmup)

    # First post-warmup step: is_warmup should be false, step_size should be frozen ε̄
    epsilon_bar_final = state.epsilon_bar
    t_post, state_post = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
    @test t_post.is_warmup == false
    @test t_post.step_size ≈ epsilon_bar_final

    # Subsequent post-warmup steps keep the same step_size
    _, state_post2 = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state_post)
    @test state_post2.epsilon_bar ≈ epsilon_bar_final
end

@testset "adapted step size targets acceptance rate" begin
    D = 5
    model = DensityModel(logp_adapt, gradlogp_adapt, D)
    n_w = 2_000
    sampler = AdaptiveMALASampler(0.5; n_warmup=n_w, target_accept=0.574)

    # Use a single persistent RNG throughout warmup and post-warmup.
    rng = MersenneTwister(42)
    _, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler)
    for _ in 1:n_w
        _, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
    end

    #=
    After warmup, ε̄ should yield acceptance near target.
    Run 500 post-warmup steps and measure actual rate.
    =#
    n_accept = 0
    for _ in 1:500
        t, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
        n_accept += t.accepted
    end
    accept_rate = n_accept / 500

    @test 0.35 < accept_rate < 0.80   # wide tolerance; adaptation is approximate
end

@testset "sample() end-to-end with AdaptiveMALASampler" begin
    model = DensityModel(logp_adapt, gradlogp_adapt, 2)
    sampler = AdaptiveMALASampler(0.2; n_warmup=50)

    samples = sample(MersenneTwister(1), model, sampler, 100; progress=false)
    @test length(samples) == 100
end

@testset "sample() with chain_type=Chains" begin
    model = DensityModel(logp_adapt, gradlogp_adapt, 2)
    sampler = AdaptiveMALASampler(0.2; n_warmup=50)

    chain = sample(
        MersenneTwister(1),
        model,
        sampler,
        150;
        chain_type=SymChain,
        progress=false,
    )

    @test chain isa SymChain
    @test FlexiChains.niters(chain) == 150

    extras_names = FlexiChains.extras(chain)
    @test FlexiChains.Extra(:logp) in extras_names
    @test FlexiChains.Extra(:accepted) in extras_names
    @test FlexiChains.Extra(:step_size) in extras_names
    @test FlexiChains.Extra(:is_warmup) in extras_names

    @test all(isfinite, chain[:logp])
    @test all(x -> x == 0.0 || x == 1.0, chain[:accepted])
    @test all(s -> s > 0, chain[:step_size])

    # Warmup samples appear at the start
    @test chain[:is_warmup][1] == 1.0
    # Post-warmup samples have is_warmup == 0
    @test chain[:is_warmup][end] == 0.0
end

@testset "step_size is constant after warmup" begin
    model = DensityModel(logp_adapt, gradlogp_adapt, 3)
    n_w = 30
    sampler = AdaptiveMALASampler(0.1; n_warmup=n_w)

    chain = sample(
        MersenneTwister(3),
        model,
        sampler,
        n_w + 50;
        chain_type=VNChain,
        progress=false,
    )

    # Filter by is_warmup flag to avoid off-by-one from the init transition.
    is_wup = vec(chain[:is_warmup]) .== 1.0
    step_sizes = vec(chain[:step_size])
    post_warmup = step_sizes[.!is_wup]

    @test length(post_warmup) > 0
    # All post-warmup step sizes must be identical (frozen ε̄)
    @test all(≈(post_warmup[1]), post_warmup)
end

@testset "AdaptiveMALASampler stationary distribution" begin
    D = 3
    model = DensityModel(logp_adapt, gradlogp_adapt, D)
    n_w = 1_000
    sampler = AdaptiveMALASampler(0.5; n_warmup=n_w)

    chain = sample(
        MersenneTwister(2025),
        model,
        sampler,
        n_w + 5_000;
        chain_type=VNChain,
        progress=false,
    )

    # Discard warmup
    post = Array(chain)[(n_w + 1):end, :, :]   # 5000 × D

    mu = vec(mean(post; dims=1))
    vars = vec(var(post; dims=1))

    @test maximum(abs.(mu)) < 0.15
    @test maximum(abs.(vars .- 1.0)) < 0.25
end

@testset "AdaptiveMALASampler parallel chains via MCMCThreads" begin
    model = DensityModel(logp_adapt, gradlogp_adapt, 2)
    sampler = AdaptiveMALASampler(0.1; n_warmup=20)

    chains = sample(
        MersenneTwister(42),
        model,
        sampler,
        ParallelMCMC.AbstractMCMC.MCMCThreads(),
        60,
        2;
        chain_type=VNChain,
        progress=false,
    )

    @test chains isa VNChain
    @test FlexiChains.niters(chains) == 60
    @test FlexiChains.nchains(chains) == 2
end
