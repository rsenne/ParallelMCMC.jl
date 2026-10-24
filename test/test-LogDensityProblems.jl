using Test
using Random
using Statistics
using FlexiChains

using ParallelMCMC

using LogDensityProblems: LogDensityProblems
using ADTypes
using ForwardDiff: ForwardDiff

#=
μ ~ N(0, 1), y | μ ~ N(μ, 0.5), observed y = 1.5.
Posterior: σ_post² = 1 / (1 + 1/0.25) = 0.2, μ_post = 0.2 * 1.5 / 0.25 = 1.2.
The Hessian is constant: -1 - 1/0.25 = -5.
=#
const LDP_OBS = 1.5
const LDP_MU_POST = 1.2
const LDP_VAR_POST = 0.2

struct _NormalLD end
LogDensityProblems.capabilities(::Type{_NormalLD}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(::_NormalLD) = 1
function LogDensityProblems.logdensity(::_NormalLD, x)
    μ = x[1]
    return -0.5 * μ^2 - 0.5 * ((LDP_OBS - μ) / 0.5)^2
end
function LogDensityProblems.logdensity_and_gradient(ld::_NormalLD, x)
    μ = x[1]
    return LogDensityProblems.logdensity(ld, x), [-μ + (LDP_OBS - μ) / 0.25]
end

struct _NoGradLD end
LogDensityProblems.capabilities(::Type{_NoGradLD}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(::_NoGradLD) = 1

@testset "constructor" begin
    model = DensityModel(_NormalLD())
    @test model.dim == 1
    @test model.logdensity([0.0]) ≈ -4.5
    @test model.grad_logdensity([0.0]) ≈ [6.0]
    @test model.param_names === nothing

    @test_throws "must support gradients" DensityModel(_NoGradLD())
end

@testset "AdaptiveMALA recovers the posterior" begin
    model = DensityModel(_NormalLD(); param_names=[:μ])
    chain = sample(
        MersenneTwister(1),
        model,
        AdaptiveMALASampler(0.3; n_warmup=500),
        4_000;
        chain_type=VNChain,
        discard_warmup=true,
        progress=false,
    )
    @test only(FlexiChains.parameters(chain)) == @vn(μ)
    draws = vec(chain[@vn(μ)])
    @test abs(mean(draws) - LDP_MU_POST) < 0.1
    @test abs(var(draws) - LDP_VAR_POST) < 0.05
end

@testset "parameter is named x by default" begin
    chain = sample(
        MersenneTwister(2),
        DensityModel(_NormalLD()),
        MALASampler(0.3),
        20;
        chain_type=SymChain,
        progress=false,
    )
    @test only(FlexiChains.parameters(chain)) == :x
end

@testset "SecondOrder hvp drives ParallelMALA" begin
    so = ParallelMCMC.DI.SecondOrder(ADTypes.AutoForwardDiff(), ADTypes.AutoForwardDiff())
    model = DensityModel(_NormalLD(); param_names=[:μ], hvp=so)
    @test model.hvp === so

    prepped = ParallelMCMC._prepare_model(model, [0.0], 8, nothing)
    @test prepped.hvp([0.0], [1.0]) ≈ [-5.0]

    chain = sample(
        MersenneTwister(12),
        model,
        ParallelMALASampler(0.02; T=8),
        64;
        chain_type=SymChain,
        progress=false,
    )
    @test all(isfinite, vec(chain[:μ]))
end

@testset "batched slots reach the prepared model" begin
    logp_b(X) = vec(-0.5 .* X .^ 2 .- 2 .* (LDP_OBS .- X) .^ 2)
    grad_b(X) = -X .+ 4 .* (LDP_OBS .- X)
    hvp_b(X, V) = -5 .* V
    model = DensityModel(
        _NormalLD();
        param_names=[:μ],
        hvp=(x, v) -> -5 .* v,
        logdensity_batch=logp_b,
        grad_logdensity_batch=grad_b,
        hvp_batch=hvp_b,
    )

    prepped = ParallelMCMC._prepare_model(model, [0.0], 8, nothing)
    @test prepped.logdensity_batch === logp_b
    @test prepped.grad_logdensity_batch === grad_b
    @test prepped.hvp_batch === hvp_b

    chain = sample(
        MersenneTwister(13),
        model,
        ParallelMALASampler(0.02; T=8),
        64;
        chain_type=SymChain,
        progress=false,
    )
    @test all(isfinite, vec(chain[:μ]))
end
