using Test
using Random
using LinearAlgebra
using Statistics
using MCMCChains

using ParallelMCMC

using DynamicPPL
using LogDensityProblems
using LogDensityProblemsAD
using ADTypes
using Distributions: Normal, MvNormal

# A simple 1-D normal likelihood:  μ ~ N(0,1),  y | μ ~ N(μ, 0.5)
# Posterior:  μ | y=1.5  is N(μ_post, σ_post²)
# σ_post² = 1 / (1/1² + 1/0.5²) = 1 / (1 + 4) = 0.2
# μ_post  = σ_post² * (y / 0.5²) = 0.2 * (1.5 / 0.25) = 0.2 * 6 = 1.2
const TRUE_OBS = 1.5
const TRUE_MU_POST = 1.2
const TRUE_VAR_POST = 0.2

@model function normal_model(y)
    μ ~ Normal(0.0, 1.0)
    y ~ Normal(μ, 0.5)
end

@model function mv_model(y)
    μ ~ MvNormal(zeros(2), I)
    y ~ MvNormal(μ, 0.5 * I)
end

@testset "LogDensityProblemsExt: param_names kwarg" begin
    ld = DynamicPPL.LogDensityFunction(normal_model(TRUE_OBS))
    ldg = LogDensityProblemsAD.ADgradient(ADTypes.AutoEnzyme(), ld)

    model = DensityModel(ldg; param_names=[:μ])

    @test model.dim == 1
    @test model.param_names == [:μ]

    chain = sample(
        MersenneTwister(1),
        model,
        AdaptiveMALASampler(0.3; n_warmup=200),
        600;
        chain_type=MCMCChains.Chains,
        progress=false,
    )
    @test :μ in names(chain, :parameters)
    @test !(Symbol("x[1]") in names(chain, :parameters))
end

@testset "DynamicPPLExt: convenience constructor" begin
    model = DensityModel(normal_model(TRUE_OBS))

    @test model.dim == 1
    @test model.param_names == [:μ]
    @test isfinite(model.logdensity([0.0]))
    @test isfinite(model.grad_logdensity([0.0])[1])
end

@testset "DynamicPPLExt: named columns in Chains output" begin
    model = DensityModel(normal_model(TRUE_OBS))

    chain = sample(
        MersenneTwister(2),
        model,
        AdaptiveMALASampler(0.3; n_warmup=200),
        600;
        chain_type=MCMCChains.Chains,
        progress=false,
    )

    @test chain isa MCMCChains.Chains
    @test :μ in names(chain, :parameters)
    @test !(Symbol("x[1]") in names(chain, :parameters))
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
        chain_type=MCMCChains.Chains,
        progress=false,
    )
    chain_trimmed = sample(
        MersenneTwister(3),
        model,
        sampler,
        n_total;
        chain_type=MCMCChains.Chains,
        progress=false,
        discard_warmup=true,
    )

    @test size(chain_full, 1) == n_total
    @test size(chain_trimmed, 1) == n_total - n_warmup - 1
    @test all(==(0.0), vec(chain_trimmed[:is_warmup]))
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
        chain_type=MCMCChains.Chains,
        progress=false,
        discard_warmup=true,
    )

    mu_samples = vec(Array(chain[:μ]))

    @test abs(mean(mu_samples) - TRUE_MU_POST) < 0.05
    @test abs(var(mu_samples) - TRUE_VAR_POST) < 0.05
end

@testset "multivariate model: named columns for each dimension" begin
    obs = [1.0, -1.0]
    model = DensityModel(mv_model(obs))

    @test model.dim == 2
    @test model.param_names == [Symbol("μ[1]"), Symbol("μ[2]")]

    chain = sample(
        MersenneTwister(7),
        model,
        AdaptiveMALASampler(0.2; n_warmup=100),
        300;
        chain_type=MCMCChains.Chains,
        progress=false,
    )
    @test Symbol("μ[1]") in names(chain, :parameters)
    @test Symbol("μ[2]") in names(chain, :parameters)
end
