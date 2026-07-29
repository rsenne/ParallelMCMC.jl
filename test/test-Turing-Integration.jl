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
    model = DensityModel(normal_model(TRUE_OBS); ad_backend=ADTypes.AutoForwardDiff())

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
    model = DensityModel(beta_model(); ad_backend=ADTypes.AutoForwardDiff())

    @test model.dim == 1
    @test isfinite(model.logdensity([-0.4]))
    @test isfinite(model.grad_logdensity([-0.4])[1])
end

@testset "DynamicPPLExt: generic Turing model works with ParallelMALA and default Enzyme HVP" begin
    model = DensityModel(normal_model(TRUE_OBS); ad_backend=ADTypes.AutoForwardDiff())

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

@testset "DynamicPPLExt: SecondOrder hvp on a Turing model" begin
    #= The gradient slot of a Turing model is DynamicPPL's own AD-prepared
    gradient, and its preparation rejects the tangents an outer pass would push
    through it — so a plain backend in `hvp` cannot differentiate it. A
    `SecondOrder` differentiates the log-density twice instead, bypassing that
    gradient, which is what makes an AD HVP reachable for a Turing model at all.

    normal_model(y) in unconstrained space is
      logp(μ) = logpdf(Normal(0,1), μ) + logpdf(Normal(μ, 0.5), y),
    so H = -1 - 1/0.5^2 = -5 and Hv = -5v. =#
    so = ParallelMCMC.DI.SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoForwardDiff())
    model = DensityModel(
        normal_model(TRUE_OBS); ad_backend=ADTypes.AutoForwardDiff(), hvp=so
    )
    @test model.hvp === so

    prepped = ParallelMCMC._prepare_model(model, [0.0], 8, nothing)
    @test prepped.hvp([0.0], [1.0]) ≈ [-5.0]
    # the model brought its own HVP, so no sampler backend is needed
    chain = sample(
        MersenneTwister(12),
        model,
        ParallelMALASampler(0.02; T=8),
        64;
        chain_type=VNChain,
        progress=false,
    )
    @test all(isfinite, vec(chain[@varname(μ)]))

    #= A plain backend is the case that cannot work. Preparing it succeeds — DI
    only builds the pushforward against the Float64 template — and it is the
    first call, pushing tangents into DynamicPPL's prepared gradient, that
    fails. Pinned as a test so that if DynamicPPL ever lifts this, the
    `SecondOrder`-only advice in the extension docstring gets revisited. =#
    model_plain = DensityModel(
        normal_model(TRUE_OBS);
        ad_backend=ADTypes.AutoForwardDiff(),
        hvp=ADTypes.AutoForwardDiff(),
    )
    prepped_plain = ParallelMCMC._prepare_model(model_plain, [0.0], 8, nothing)
    @test_throws Exception prepped_plain.hvp([0.0], [1.0])
end

@testset "DynamicPPLExt: batched slots reach the batched DEER path" begin
    #= DynamicPPL supplies no batched log-density, so the batched slots are the
    only way a Turing model reaches the batched update. Written out by hand for
    normal_model, including the normalizing constants so that the log-densities
    reported for a trajectory agree with `model.logdensity`. =#
    σ = 0.5
    logp_b(X) =
        vec(-0.5 .* X .^ 2 .- 0.5 .* ((TRUE_OBS .- X) ./ σ) .^ 2 .- log(2π) .- log(σ))
    grad_b(X) = -X .+ (TRUE_OBS .- X) ./ σ^2
    hvp_b(X, V) = (-1 - 1 / σ^2) .* V

    model = DensityModel(
        normal_model(TRUE_OBS);
        ad_backend=ADTypes.AutoForwardDiff(),
        hvp=(x, v) -> (-1 - 1 / σ^2) .* v,
        logdensity_batch=logp_b,
        grad_logdensity_batch=grad_b,
        hvp_batch=hvp_b,
    )

    @test model.logdensity_batch === logp_b
    @test model.grad_logdensity_batch === grad_b
    @test model.hvp_batch === hvp_b

    # the hand-written batched log-density agrees with the model's own, column by column
    X = reshape([-0.5, 0.0, 0.7, 1.4], 1, 4)
    @test logp_b(X) ≈ [model.logdensity(X[:, t]) for t in 1:size(X, 2)]

    prepped = ParallelMCMC._prepare_model(model, [0.0], 4, nothing)
    @test prepped.grad_logdensity_batch === grad_b
    @test prepped.hvp_batch === hvp_b

    chain = sample(
        MersenneTwister(13),
        model,
        ParallelMALASampler(0.02; T=8),
        64;
        chain_type=VNChain,
        progress=false,
    )
    @test all(isfinite, vec(chain[@varname(μ)]))
end

@testset "DynamicPPLExt: MvNormal(zeros(2), I) runs with ParallelMALA" begin
    model = DensityModel(mvnormal_2d_model(); ad_backend=ADTypes.AutoForwardDiff())

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
    samples = chain[@varname(x), stack = true]
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
    model = DensityModel(dirichlet_3_model(); ad_backend=ADTypes.AutoForwardDiff())

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
    model = DensityModel(mvnormal_2d_model(); ad_backend=ADTypes.AutoForwardDiff())
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
        thinning=2,
        progress=false,
    )
    @test chain isa VNChain
    @test only(FlexiChains.parameters(chain)) == @varname(x)
    @test all(isfinite, Array(chain))
end

@testset "DynamicPPLExt: named columns in Chains output" begin
    model = DensityModel(normal_model(TRUE_OBS); ad_backend=ADTypes.AutoForwardDiff())

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
    model = DensityModel(normal_model(TRUE_OBS); ad_backend=ADTypes.AutoForwardDiff())
    n_warmup = 200
    n_total = 800
    sampler = AdaptiveMALASampler(0.3; n_warmup=n_warmup)

    chain_full = sample(
        MersenneTwister(3), model, sampler, n_total; chain_type=VNChain, progress=false
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
    model = DensityModel(normal_model(TRUE_OBS); ad_backend=ADTypes.AutoForwardDiff())
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
    model = DensityModel(mv_model(obs); ad_backend=ADTypes.AutoForwardDiff())

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
    @test all(chain[@varname(c), stack = true] .>= 0.0)  # Dirichlet samples should be non-negative
end
