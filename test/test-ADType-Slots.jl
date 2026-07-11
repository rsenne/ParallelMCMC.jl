using Test
using Random
using LinearAlgebra
using FlexiChains

using ParallelMCMC
using ADTypes
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff

#=
Backend-or-callable slots on `DensityModel` (issues #52 / #40). The Gaussian
target keeps everything analytically checkable: ∇logp = -x, hvp(x, v) = -v.
=#
logp_slots(x) = -0.5 * dot(x, x)
gradlogp_slots(x) = -x
logp_batch_slots(X) = vec(-0.5 .* sum(abs2, X; dims=1))
gradlogp_batch_slots(X) = -X

const D_SLOTS = 5
const CT_SLOTS = FlexiChains.FlexiChain{Symbol}

@testset "constructor validation" begin
    # primal slots cannot be backends — nothing to derive them from
    @test_throws ArgumentError DensityModel(AutoForwardDiff(), gradlogp_slots, D_SLOTS)
    @test_throws ArgumentError DensityModel(
        logp_slots, gradlogp_slots, D_SLOTS; logdensity_batch=AutoForwardDiff()
    )
    # batched AD slots need their primal/gradient counterpart
    @test_throws ArgumentError DensityModel(
        logp_slots, gradlogp_slots, D_SLOTS; grad_logdensity_batch=AutoForwardDiff()
    )
    @test_throws ArgumentError DensityModel(
        logp_slots, gradlogp_slots, D_SLOTS; hvp_batch=AutoForwardDiff()
    )
    # grad slot is mandatory (callable or backend)
    @test_throws ArgumentError DensityModel(logp_slots, nothing, D_SLOTS)
end

@testset "_prepare_model resolution" begin
    rng = MersenneTwister(11)
    x0 = randn(rng, D_SLOTS)

    @testset "returns a PreppedDensityModel; callable slots pass through" begin
        model = DensityModel(logp_slots, gradlogp_slots, D_SLOTS)
        m_r = ParallelMCMC._prepare_model(model, x0)
        @test m_r isa ParallelMCMC.PreppedDensityModel
        @test m_r.grad_logdensity === gradlogp_slots

        # DEER form: a user-supplied hvp callable passes through untouched
        hvp_fn = (x, v) -> -v
        model_hvp = DensityModel(logp_slots, gradlogp_slots, D_SLOTS; hvp=hvp_fn)
        m_deer = ParallelMCMC._prepare_model(model_hvp, x0, 8, nothing)
        @test m_deer isa ParallelMCMC.PreppedDensityModel
        @test m_deer.hvp === hvp_fn
    end

    @testset "gradient slot" begin
        model = DensityModel(logp_slots, AutoForwardDiff(), D_SLOTS)
        m_r = ParallelMCMC._prepare_model(model, x0)
        @test m_r.grad_logdensity(x0) ≈ -x0
        # inputs that don't match the preparation type take the unprepped fallback
        x32 = Float32.(x0)
        @test m_r.grad_logdensity(x32) ≈ -x32
    end

    @testset "hvp slot differentiates the (resolved) gradient" begin
        v = randn(rng, D_SLOTS)

        model = DensityModel(logp_slots, gradlogp_slots, D_SLOTS; hvp=AutoForwardDiff())
        m_r = ParallelMCMC._prepare_model(model, x0, 8, nothing)
        @test m_r.hvp(x0, v) ≈ -v

        model_ad = DensityModel(logp_slots, AutoForwardDiff(), D_SLOTS; hvp=AutoForwardDiff())
        m_ad = ParallelMCMC._prepare_model(model_ad, x0, 8, nothing)
        @test m_ad.hvp(x0, v) ≈ -v
    end

    @testset "sampler backend injected as hvp fallback" begin
        v = randn(rng, D_SLOTS)
        model = DensityModel(logp_slots, gradlogp_slots, D_SLOTS)

        m_r = ParallelMCMC._prepare_model(model, x0, 8, AutoForwardDiff())
        @test m_r.hvp(x0, v) ≈ -v

        # no hvp source anywhere → informative error
        @test_throws ArgumentError ParallelMCMC._prepare_model(model, x0, 8, nothing)
    end

    @testset "batched slots" begin
        T = 8
        model = DensityModel(
            logp_slots,
            AutoForwardDiff(),
            D_SLOTS;
            logdensity_batch=logp_batch_slots,
            grad_logdensity_batch=AutoForwardDiff(),
            hvp=AutoForwardDiff(),
            hvp_batch=AutoForwardDiff(),
        )
        m_r = ParallelMCMC._prepare_model(model, x0, T, nothing)
        X = randn(rng, D_SLOTS, T)
        V = randn(rng, D_SLOTS, T)
        @test m_r.grad_logdensity_batch(X) ≈ -X
        @test m_r.hvp_batch(X, V) ≈ -V

        # hvp_batch fallback from the sampler backend
        model_fb = DensityModel(
            logp_slots,
            gradlogp_slots,
            D_SLOTS;
            logdensity_batch=logp_batch_slots,
            grad_logdensity_batch=gradlogp_batch_slots,
        )
        m_fb = ParallelMCMC._prepare_model(model_fb, x0, T, AutoForwardDiff())
        @test m_fb.hvp_batch(X, V) ≈ -V
        @test_throws ArgumentError ParallelMCMC._prepare_model(model_fb, x0, T, nothing)
    end

    @testset "two-argument form leaves DEER-only slots untouched" begin
        model = DensityModel(logp_slots, gradlogp_slots, D_SLOTS; hvp=AutoForwardDiff())
        m_r = ParallelMCMC._prepare_model(model, x0)
        @test m_r isa ParallelMCMC.PreppedDensityModel
        @test m_r.hvp isa ADTypes.AbstractADType
    end
end

@testset "logp-only sampling matches analytic-gradient sampling" begin
    model_ad = DensityModel(logp_slots, AutoForwardDiff(), D_SLOTS)
    model_an = DensityModel(logp_slots, gradlogp_slots, D_SLOTS)

    @testset "MALASampler" begin
        c_ad = sample(
            MersenneTwister(21), model_ad, MALASampler(0.2), 100;
            chain_type=CT_SLOTS, progress=false,
        )
        c_an = sample(
            MersenneTwister(21), model_an, MALASampler(0.2), 100;
            chain_type=CT_SLOTS, progress=false,
        )
        @test c_ad[:x] ≈ c_an[:x]
    end

    @testset "AdaptiveMALASampler" begin
        c_ad = sample(
            MersenneTwister(22), model_ad, AdaptiveMALASampler(0.2; n_warmup=50), 100;
            chain_type=CT_SLOTS, progress=false,
        )
        c_an = sample(
            MersenneTwister(22), model_an, AdaptiveMALASampler(0.2; n_warmup=50), 100;
            chain_type=CT_SLOTS, progress=false,
        )
        @test c_ad[:x] ≈ c_an[:x]
    end

    @testset "ParallelMALASampler" begin
        s = ParallelMALASampler(0.05; T=16, backend=AutoForwardDiff())
        c_ad = sample(
            MersenneTwister(23), model_ad, s, 64; chain_type=CT_SLOTS, progress=false
        )
        c_an = sample(
            MersenneTwister(23), model_an, s, 64; chain_type=CT_SLOTS, progress=false
        )
        @test c_ad[:x] ≈ c_an[:x]
    end
end

@testset "ParallelMALASampler with every derivative slot as a backend" begin
    model = DensityModel(
        logp_slots,
        AutoForwardDiff(),
        D_SLOTS;
        logdensity_batch=logp_batch_slots,
        grad_logdensity_batch=AutoForwardDiff(),
        hvp=AutoForwardDiff(),
        hvp_batch=AutoForwardDiff(),
    )
    s = ParallelMALASampler(0.05; T=16, backend=AutoForwardDiff())
    chain = sample(MersenneTwister(31), model, s, 64; chain_type=CT_SLOTS, progress=false)
    @test size(chain) == (64, 1)
    @test all(x -> all(isfinite, x), chain[:x])
end

@testset "logp-only with Enzyme" begin
    model = DensityModel(logp_slots, AutoEnzyme(), D_SLOTS)
    rng = MersenneTwister(41)
    x0 = randn(rng, D_SLOTS)
    m_r = ParallelMCMC._prepare_model(model, x0)
    @test m_r.grad_logdensity(x0) ≈ -x0

    chain = sample(
        MersenneTwister(42), model, MALASampler(0.2), 100;
        chain_type=CT_SLOTS, progress=false,
    )
    @test all(x -> all(isfinite, x), chain[:x])
end

@testset "step interface carries the prepped model in the state" begin
    model = DensityModel(logp_slots, AutoForwardDiff(), D_SLOTS)
    rng = MersenneTwister(51)

    t, s = ParallelMCMC.AbstractMCMC.step(rng, model, MALASampler(0.2))
    @test s.model isa ParallelMCMC.PreppedDensityModel
    @test !(s.model.grad_logdensity isa ADTypes.AbstractADType)
    t2, s2 = ParallelMCMC.AbstractMCMC.step(rng, model, MALASampler(0.2), s)
    # the same resolved callable is reused, not re-prepared
    @test s2.model.grad_logdensity === s.model.grad_logdensity

    sp = ParallelMALASampler(0.05; T=8, backend=AutoForwardDiff())
    tp, spstate = ParallelMCMC.AbstractMCMC.step(rng, model, sp)
    @test spstate.model isa ParallelMCMC.PreppedDensityModel
    @test !(spstate.model.grad_logdensity isa ADTypes.AbstractADType)
    # sampler backend was injected: the prepped model always carries a callable hvp
    @test spstate.model.hvp !== nothing
end

@testset "sampler backend is optional when the model specifies hvp" begin
    model = DensityModel(
        logp_slots, AutoForwardDiff(), D_SLOTS; hvp=AutoForwardDiff()
    )
    s = ParallelMALASampler(0.05; T=16)   # no backend
    chain = sample(MersenneTwister(61), model, s, 64; chain_type=CT_SLOTS, progress=false)
    @test size(chain) == (64, 1)
    @test all(x -> all(isfinite, x), chain[:x])

    # neither model hvp nor sampler backend → informative error at sampling start
    bare = DensityModel(logp_slots, gradlogp_slots, D_SLOTS)
    @test_throws ArgumentError sample(
        MersenneTwister(62), bare, s, 32; chain_type=CT_SLOTS, progress=false
    )
end
