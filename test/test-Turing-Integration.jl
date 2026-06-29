using Test
using Random
using LinearAlgebra
using Statistics
using FlexiChains

using ParallelMCMC

using DynamicPPL
using LogDensityProblems
using ADTypes
using Enzyme
using ForwardDiff
using Distributions: Beta, Dirichlet, Normal, MvNormal, product_distribution

#=
A simple 1-D normal likelihood:  μ ~ N(0,1),  y | μ ~ N(μ, 0.5)
Posterior:  μ | y=1.5  is N(μ_post, σ_post²)
σ_post² = 1 / (1/1² + 1/0.5²) = 1 / (1 + 4) = 0.2
μ_post  = σ_post² * (y / 0.5²) = 0.2 * (1.5 / 0.25) = 0.2 * 6 = 1.2
=#
const TRUE_OBS = 1.5
const TRUE_MU_POST = 1.2
const TRUE_VAR_POST = 0.2

@model function normal_model(y)
    μ ~ Normal(0.0, 1.0)
    y ~ Normal(μ, 0.5)
end

@model function mv_model(y)
    c ~ Dirichlet(ones(3)) # to test constraints
    μ ~ product_distribution((a=Normal(), b=Normal()))
    y ~ MvNormal([μ.a, μ.b], 0.5 * I)
end

@model function beta_model()
    x ~ Beta(2, 2)
end

@model function mvnormal_2d_model()
    x ~ MvNormal(zeros(2), I)
end

@model function dirichlet_3_model()
    x ~ Dirichlet(ones(3))
end

@testset "directly passing LogDensityFunction" begin
    ld = DynamicPPL.LogDensityFunction(
        normal_model(TRUE_OBS),
        DynamicPPL.getlogjoint_internal,
        DynamicPPL.LinkAll();
        adtype=ADTypes.AutoEnzyme(),
    )

    model = DensityModel(ld)

    @test model.dim == 1

    chain = sample(
        MersenneTwister(1),
        model,
        AdaptiveMALASampler(0.3; n_warmup=200),
        600;
        chain_type=VNChain,
        progress=false,
    )
    @test only(FlexiChains.parameters(chain)) == @varname(μ)
end

@testset "DynamicPPLExt: convenience constructor" begin
    model = DensityModel(normal_model(TRUE_OBS))

    @test model.dim == 1
    @test isfinite(model.logdensity([0.0]))
    @test isfinite(model.grad_logdensity([0.0])[1])

    chain = sample(
        MersenneTwister(1),
        model,
        AdaptiveMALASampler(0.3; n_warmup=200),
        600;
        chain_type=VNChain,
        progress=false,
    )
    @test only(FlexiChains.parameters(chain)) == @varname(μ)

    @test_throws "SymChain is not supported" sample(
        MersenneTwister(1),
        model,
        AdaptiveMALASampler(0.3; n_warmup=200),
        600;
        chain_type=SymChain,
        progress=false,
    )
end

@testset "DynamicPPLExt: convenience constructor uses linked space for constrained models" begin
    model = DensityModel(beta_model())

    @test model.dim == 1
    @test isfinite(model.logdensity([-0.4]))
    @test isfinite(model.grad_logdensity([-0.4])[1])
end

@testset "DynamicPPLExt: generic Turing model works with ParallelMALA and default Enzyme HVP" begin
    model = DensityModel(normal_model(TRUE_OBS))

    @test model.hvp === nothing

    for jacobian in (:diag, :stoch_diag)
        sampler = ParallelMALASampler(
            0.03;
            T=4,
            maxiter=80,
            tol_abs=1e-5,
            tol_rel=1e-4,
            jacobian=jacobian,
            damping=0.5,
            backend=ADTypes.AutoEnzyme(),
        )

        trans, state = ParallelMCMC.AbstractMCMC.step(
            MersenneTwister(11), model, sampler; initial_params=[0.0]
        )

        @test trans isa ParallelMALATransition
        @test state isa ParallelMALAState
        @test isfinite(trans.logp)
        @test all(isfinite, state.trajectory)
    end
end

@testset "DynamicPPLExt: MvNormal(zeros(2), I) runs with ParallelMALA" begin
    model = DensityModel(mvnormal_2d_model())

    @test model.dim == 2
    @test isfinite(model.logdensity(zeros(2)))
    @test all(isfinite, model.grad_logdensity(zeros(2)))

    sampler = ParallelMALASampler(
        0.2; T=8, maxiter=80, tol_abs=1e-4, tol_rel=1e-3, backend=ADTypes.AutoEnzyme()
    )
    chain = sample(
        MersenneTwister(3),
        model,
        sampler,
        800;
        initial_params=zeros(2),
        chain_type=VNChain,
        progress=false,
    )
    samples = chain[@varname(x), stack=true]
    @test all(isfinite, samples)
    # Standard normal in 2-D: posterior mean should be near zero.
    posterior_means = mean(samples; dims=1)
    @test maximum(abs, posterior_means) < 0.25
end

@testset "DynamicPPLExt: Dirichlet(ones(3)) runs with ParallelMALA (linked space)" begin
    #=
    Dirichlet(ones(3)) lives on a 2-simplex, so its unconstrained
    representation has dim 2. Bijectors handles the link/unlink.
    =#
    model = DensityModel(dirichlet_3_model())

    @test model.dim == 2
    @test isfinite(model.logdensity(zeros(2)))
    @test all(isfinite, model.grad_logdensity(zeros(2)))

    sampler = ParallelMALASampler(
        0.2; T=8, maxiter=80, tol_abs=1e-4, tol_rel=1e-3, backend=ADTypes.AutoEnzyme()
    )
    chain = sample(
        MersenneTwister(4),
        model,
        sampler,
        800;
        initial_params=zeros(2),
        chain_type=VNChain,
        progress=false,
    )
    @test chain isa VNChain
    @test all(isfinite, Array(chain))
end

@testset "DynamicPPLExt: ParallelMALA bundle_samples fallback path (thinning)" begin
    #= A non-default kwarg (here `thinning`) forces ParallelMALA's `mcmcsample` override =#
    model = DensityModel(mvnormal_2d_model())
    sampler = ParallelMALASampler(
        0.2; T=8, maxiter=80, tol_abs=1e-4, tol_rel=1e-3, backend=ADTypes.AutoEnzyme()
    )
    chain = sample(
        MersenneTwister(3), model, sampler, 800;
        initial_params=zeros(2), chain_type=VNChain, thinning=2, progress=false,
    )
    @test chain isa VNChain
    @test only(FlexiChains.parameters(chain)) == @varname(x)
    @test all(isfinite, Array(chain))
end

@testset "DynamicPPLExt: named columns in Chains output" begin
    model = DensityModel(normal_model(TRUE_OBS))

    chain = sample(
        MersenneTwister(2),
        model,
        AdaptiveMALASampler(0.3; n_warmup=200),
        600;
        chain_type=VNChain,
        progress=false,
    )

    @test chain isa VNChain
    params = FlexiChains.parameters(chain)
    @test @varname(μ) in params
    @test !(@varname(x) in params)
end

@testset "discard_warmup=true removes warmup samples" begin
    model = DensityModel(normal_model(TRUE_OBS))
    n_warmup = 200
    n_total = 800
    sampler = AdaptiveMALASampler(0.3; n_warmup=n_warmup)

    chain_full = sample(
        MersenneTwister(3),
        model,
        sampler,
        n_total;
        chain_type=VNChain,
        progress=false,
    )
    chain_trimmed = sample(
        MersenneTwister(3),
        model,
        sampler,
        n_total;
        chain_type=VNChain,
        progress=false,
        discard_warmup=true,
    )

    @test FlexiChains.niters(chain_full) == n_total
    @test FlexiChains.niters(chain_trimmed) == n_total - n_warmup - 1
    @test all(==(false), chain_trimmed[:is_warmup])
end

@testset "posterior mean and variance match analytic solution" begin
    model = DensityModel(normal_model(TRUE_OBS))
    n_warmup = 2_000
    n_draw = 10_000
    sampler = AdaptiveMALASampler(0.3; n_warmup=n_warmup)

    chain = sample(
        MersenneTwister(2025),
        model,
        sampler,
        n_warmup + n_draw;
        chain_type=VNChain,
        progress=false,
        discard_warmup=true,
    )

    mu_samples = vec(chain[:μ])

    @test abs(mean(mu_samples) - TRUE_MU_POST) < 0.05
    @test abs(var(mu_samples) - TRUE_VAR_POST) < 0.05
end

@testset "multivariate model: named columns for each dimension" begin
    obs = [1.0, -1.0]
    model = DensityModel(mv_model(obs))

    # 2 from linked Dirichlet + 2 from product_distribution
    @test model.dim == 4

    chain = sample(
        MersenneTwister(7),
        model,
        AdaptiveMALASampler(0.2; n_warmup=100),
        300;
        chain_type=VNChain,
        progress=false,
    )

    # Check that the chain contains parameters in original space.
    # The Dirichlet parameter should have length 3.
    @test Set(FlexiChains.parameters(chain)) == Set([@varname(c), @varname(μ)])
    @test all(chain[@varname(c), stack=true] .>= 0.0)  # Dirichlet samples should be non-negative
end
