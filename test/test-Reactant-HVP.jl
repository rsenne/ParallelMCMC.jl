using Test
using Random
using LinearAlgebra
using FlexiChains

using ParallelMCMC
using ADTypes
# Stands in for "some backend that is not Reactant" in the pairing tests below.
using ForwardDiff: ForwardDiff

#=
Reactant-compiled derivative paths (`AutoReactant`), see ext/ReactantExt.jl.

The quartic target keeps second-order structure honest: logp = -0.25‖x‖⁴ has
H = -(‖x‖² I + 2 x xᵀ), so an HVP that silently drops the second-order term
(the failure mode of `Enzyme.hvp` under `@compile`) is caught, unlike a
Gaussian where H is constant.
=#
logp_r(x) = -0.25 * sum(abs2, x)^2
gradlogp_r(x) = -sum(abs2, x) .* x
hvp_r(x, v) = -(sum(abs2, x) .* v .+ 2 .* dot(x, v) .* x)
logp_batch_r(X) = vec(-0.25 .* sum(abs2, X; dims=1) .^ 2)
gradlogp_batch_r(X) = -X .* sum(abs2, X; dims=1)

const D_R = 4
const CT_R = FlexiChains.FlexiChain{Symbol}

#= The pairing rule lives in `_resolve_hvp`, not in the extension, so its
dispatch table is checked whether or not Reactant loads. =#
@testset "Reactant does not pair with a DI backend" begin
    # Both slots Reactant, or a hand-written gradient (`nothing`), are accepted.
    @test ParallelMCMC._check_reactant_pair(AutoReactant(), AutoReactant()) === nothing
    @test ParallelMCMC._check_reactant_pair(nothing, AutoReactant()) === nothing
    @test ParallelMCMC._check_reactant_pair(AutoForwardDiff(), AutoForwardDiff()) ===
        nothing

    # One of each is refused in both directions.
    @test_throws ArgumentError ParallelMCMC._check_reactant_pair(
        AutoReactant(), AutoForwardDiff()
    )
    @test_throws ArgumentError ParallelMCMC._check_reactant_pair(
        AutoForwardDiff(), AutoReactant()
    )
end

reactant_ok = try
    using Reactant: Reactant
    using Enzyme: Enzyme
    true
catch err
    @warn "Reactant not available — skipping Reactant HVP tests" err
    false
end

if reactant_ok
    @testset "extension is loaded" begin
        @test Base.get_extension(ParallelMCMC, :ReactantExt) !== nothing
    end

    #= The same rule reached through `_prepare_model`, where the gradient slot
    resolves first: a mixed pair must still surface as an ArgumentError and not
    as whatever DI or Reactant would say downstream. =#
    @testset "mixed pairs are refused at preparation" begin
        x = zeros(D_R)

        reactant_grad = DensityModel(logp_r, AutoReactant(), D_R; hvp=AutoForwardDiff())
        @test_throws ArgumentError ParallelMCMC._prepare_model(reactant_grad, x, 8, nothing)

        reactant_hvp = DensityModel(logp_r, AutoForwardDiff(), D_R; hvp=AutoReactant())
        @test_throws ArgumentError ParallelMCMC._prepare_model(reactant_hvp, x, 8, nothing)
    end

    @testset "HVP matches analytic" begin
        rng = MersenneTwister(71)
        x = randn(rng, D_R)
        v = randn(rng, D_R)

        @testset "forward over user gradient (sampler backend)" begin
            model = DensityModel(logp_r, gradlogp_r, D_R)
            m_p = ParallelMCMC._prepare_model(model, x, 8, AutoReactant())
            @test m_p.hvp(x, v) ≈ hvp_r(x, v)
        end

        @testset "forward-over-reverse from logp alone (both slots AutoReactant)" begin
            model = DensityModel(logp_r, AutoReactant(), D_R; hvp=AutoReactant())
            m_p = ParallelMCMC._prepare_model(model, x, 8, nothing)
            @test m_p.grad_logdensity(x) ≈ gradlogp_r(x)
            @test m_p.hvp(x, v) ≈ hvp_r(x, v)
        end

        @testset "batched slots" begin
            T = 8
            X = randn(rng, D_R, T)
            V = randn(rng, D_R, T)
            Hv_cols = reduce(hcat, [hvp_r(X[:, t], V[:, t]) for t in 1:T])

            model = DensityModel(
                logp_r,
                AutoReactant(),
                D_R;
                logdensity_batch=logp_batch_r,
                grad_logdensity_batch=AutoReactant(),
                hvp=AutoReactant(),
                hvp_batch=AutoReactant(),
            )
            m_p = ParallelMCMC._prepare_model(model, X[:, 1], T, nothing)
            @test m_p.grad_logdensity_batch(X) ≈ gradlogp_batch_r(X)
            @test m_p.hvp_batch(X, V) ≈ Hv_cols
        end
    end

    @testset "end-to-end sampling with AutoReactant" begin
        model = DensityModel(logp_r, AutoReactant(), D_R)
        s = ParallelMALASampler(0.02; T=16, backend=AutoReactant())
        chain = sample(MersenneTwister(72), model, s, 64; chain_type=CT_R, progress=false)
        @test size(chain) == (64, 1)
        @test all(x -> all(isfinite, x), chain[:x])
    end

    reactant_gpu_ok = try
        using CUDA: CUDA
        CUDA.functional() && (CUDA.CuArray([1.0f0]); true)
    catch
        false
    end

    if !reactant_gpu_ok
        @info "Reactant HVP test: CUDA not functional — skipping CuArray boundary"
    else
        #=
        The compiled executable lives in Reactant's own (XLA) device memory;
        what's checked here is the CuArray <-> Reactant marshalling boundary:
        CuArray in, CuArray out, values matching the analytic HVP.
        =#
        logp_r32(x) = -0.25f0 * sum(abs2, x)^2

        @testset "CuArray boundary" begin
            rng = MersenneTwister(73)
            x_h = randn(rng, Float32, D_R)
            v_h = randn(rng, Float32, D_R)
            x_d = CUDA.CuArray(x_h)
            v_d = CUDA.CuArray(v_h)

            model = DensityModel(logp_r32, AutoReactant(), D_R; hvp=AutoReactant())
            m_p = ParallelMCMC._prepare_model(model, x_d, 8, nothing)

            g = m_p.grad_logdensity(x_d)
            @test g isa CUDA.CuArray
            @test Array(g) ≈ gradlogp_r(x_h)

            Hv = m_p.hvp(x_d, v_d)
            @test Hv isa CUDA.CuArray
            @test Array(Hv) ≈ hvp_r(x_h, v_h)
        end
    end
end
