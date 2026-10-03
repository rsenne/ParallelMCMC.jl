using Test
using Random
using LinearAlgebra
using Statistics

using ParallelMCMC
using ADTypes
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
import DifferentiationInterface as DI

#= This file only runs when PARALLELMCMC_TEST_WREN opts in (see runtests.jl).
WrenCore and WrenDists are not registered, so they have to be added to the test
environment by hand. The Reactant testset also needs PARALLELMCMC_TEST_REACTANT. =#
using WrenCore: WrenCore, @wren, @iid, @batch, @observe, VecModel
using WrenDists: WrenDists as WD

const _WREN_RUN_REACTANT =
    lowercase(get(ENV, "PARALLELMCMC_TEST_REACTANT", "false")) in ("1", "true", "yes")
if _WREN_RUN_REACTANT
    using Reactant: Reactant
end

@wren function wren_linreg(X, y)
    N, D = size(X)
    @iid β ~ WD.Normal() size=D
    σ ~ WD.Exponential()
    μ = X * β
    @batch for i in 1:N
        @observe y[i] ~ WD.Normal(μ[i], σ)
    end
end

const WREN_BETA = [1.0, -2.0, 0.5]

function wren_make_model(rng)
    N = 40
    X = randn(rng, N, length(WREN_BETA))
    y = X * WREN_BETA .+ 0.3 .* randn(rng, N)
    return wren_linreg(X, y)
end

# Per-column ForwardDiff references from the plain per-vector log-density.
wren_ref_lps(lp, P) = [lp(p) for p in eachcol(P)]
wren_ref_grads(lp, P) = reduce(hcat, [ForwardDiff.gradient(lp, p) for p in eachcol(P)])
function wren_ref_hvps(lp, P, V)
    return reduce(hcat, [ForwardDiff.hessian(lp, P[:, t]) * V[:, t] for t in axes(P, 2)])
end

function wren_check_derivatives(dm, P, V, backend, ref_lp)
    T = size(P, 2)
    m = ParallelMCMC._prepare_model(dm, P[:, 1], T, backend)
    @test m.logdensity(P[:, 1]) ≈ ref_lp(P[:, 1])
    @test m.logdensity_batch(P) ≈ wren_ref_lps(ref_lp, P)
    @test m.grad_logdensity(P[:, 1]) ≈ ForwardDiff.gradient(ref_lp, P[:, 1])
    @test m.grad_logdensity_batch(P) ≈ wren_ref_grads(ref_lp, P)
    @test m.hvp(P[:, 1], V[:, 1]) ≈ ForwardDiff.hessian(ref_lp, P[:, 1]) * V[:, 1]
    @test m.hvp_batch(P, V) ≈ wren_ref_hvps(ref_lp, P, V)
    return nothing
end

# The posterior is tight around the generating coefficients.
function wren_check_samples(samples, D)
    xs = reduce(hcat, [s.x for s in samples])
    @test size(xs, 1) == D
    @test all(isfinite, xs)
    @test all(s -> isfinite(s.logp), samples)
    @test isapprox(vec(mean(xs[1:length(WREN_BETA), :]; dims=2)), WREN_BETA; atol=0.3)
    return nothing
end

@testset "Wren extension" begin
    @test Base.get_extension(ParallelMCMC, :WrenCoreExt) !== nothing

    rng = Xoshiro(468)
    model = wren_make_model(rng)
    ref_vm = VecModel(model, AutoForwardDiff())
    ref_lp = Base.Fix1(WrenCore.lp, ref_vm)
    D = ref_vm.dimension
    T = 8
    P = randn(rng, D, T)
    V = randn(rng, D, T)
    x0 = zeros(D)

    @testset "constructors" begin
        dm = DensityModel(ref_vm)
        @test dm isa DensityModel
        @test dm.dim == D
        @test dm.grad_logdensity isa AutoForwardDiff
        @test dm.logdensity_batch isa WrenCore.BatchedLP

        dm_model = DensityModel(model; backend=AutoForwardDiff(), param_names=[:x => (D,)])
        @test dm_model.dim == D
        @test dm_model.param_names == [:x => (D,)]

        # Enzyme cannot prove the closed-over model read-only without `Const`.
        dm_enzyme = DensityModel(VecModel(model, AutoEnzyme(; mode=Enzyme.Reverse)))
        @test dm_enzyme.grad_logdensity isa AutoEnzyme{<:Any,Enzyme.Const}

        @test_throws ArgumentError DensityModel(VecModel(model, nothing))
        @test DensityModel(VecModel(model, nothing); backend=AutoForwardDiff()) isa
            DensityModel
    end

    @testset "ForwardDiff derivatives" begin
        dm = DensityModel(ref_vm)
        wren_check_derivatives(dm, P, V, AutoForwardDiff(), ref_lp)
    end

    @testset "Enzyme gradients with ForwardDiff HVPs" begin
        dm = DensityModel(
            VecModel(model, AutoEnzyme(; mode=Enzyme.Reverse));
            hvp=DI.SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
        )
        wren_check_derivatives(dm, P, V, nothing, ref_lp)
    end

    @testset "ParallelMALASampler (ForwardDiff)" begin
        dm = DensityModel(ref_vm)
        sampler = ParallelMALASampler(0.002; T=16, backend=AutoForwardDiff())
        samples = sample(
            Xoshiro(1),
            dm,
            sampler,
            300;
            initial_params=x0,
            discard_initial=100,
            progress=false,
        )
        @test length(samples) == 300
        wren_check_samples(samples, D)
    end

    if _WREN_RUN_REACTANT
        @testset "Reactant" begin
            @test Base.get_extension(ParallelMCMC, :ReactantExt) !== nothing

            for (name, make) in (
                ("Model", () -> DensityModel(model; backend=AutoReactant())),
                (
                    "VecModel in reverse mode",
                    () -> DensityModel(
                        VecModel(
                            model,
                            AutoReactant(; mode=AutoEnzyme(; mode=Enzyme.Reverse)),
                        ),
                    ),
                ),
            )
                @testset "$name" begin
                    dm = make()
                    @test dm.grad_logdensity === AutoReactant()
                    # Host calls go through WrenCore's compiled log-density.
                    @test dm.logdensity(P[:, 1]) ≈ ref_lp(P[:, 1])
                    @test dm.logdensity_batch(P) ≈ wren_ref_lps(ref_lp, P)
                    wren_check_derivatives(dm, P, V, AutoReactant(), ref_lp)
                end
            end

            @testset "ParallelMALASampler (AutoReactant)" begin
                dm = DensityModel(model; backend=AutoReactant())
                sampler = ParallelMALASampler(0.002; T=16, backend=AutoReactant())
                samples = sample(
                    Xoshiro(1),
                    dm,
                    sampler,
                    300;
                    initial_params=x0,
                    discard_initial=100,
                    progress=false,
                )
                @test length(samples) == 300
                wren_check_samples(samples, D)
            end
        end
    else
        @info "Skipping Wren Reactant tests (set PARALLELMCMC_TEST_REACTANT=true to opt in)"
    end
end
