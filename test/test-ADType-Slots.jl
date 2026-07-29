using Test
using Random
using LinearAlgebra
using FlexiChains

using ParallelMCMC
using ADTypes
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff

#=
Backend-or-callable slots on `DensityModel` (#52, #40). A standard Gaussian
target so every resolved slot has a closed form: ∇logp = -x, hvp(x, v) = -v.
=#
logp_slots(x) = -0.5 * dot(x, x)
gradlogp_slots(x) = -x
logp_batch_slots(X) = vec(-0.5 .* sum(abs2, X; dims=1))
gradlogp_batch_slots(X) = -X

const D_SLOTS = 5
const CT_SLOTS = FlexiChains.FlexiChain{Symbol}
const DI_SLOTS = ParallelMCMC.DI

@testset "constructor validation" begin
    # primal slots cannot be backends: nothing to derive them from
    @test_throws ArgumentError DensityModel(AutoForwardDiff(), gradlogp_slots, D_SLOTS)
    @test_throws ArgumentError DensityModel(
        logp_slots, gradlogp_slots, D_SLOTS; logdensity_batch=AutoForwardDiff()
    )
    #= Batched derivative slots need a logdensity_batch, callables included: the
    batched update evaluates it, so without one the slot could never be reached. =#
    for slot in (AutoForwardDiff(), gradlogp_batch_slots)
        @test_throws ArgumentError DensityModel(
            logp_slots, gradlogp_slots, D_SLOTS; grad_logdensity_batch=slot
        )
    end
    for slot in (AutoForwardDiff(), (X, V) -> -V)
        @test_throws ArgumentError DensityModel(
            logp_slots, gradlogp_slots, D_SLOTS; hvp_batch=slot
        )
    end
    # a logdensity_batch on its own is allowed: it scores whole trajectories
    @test DensityModel(
        logp_slots, gradlogp_slots, D_SLOTS; logdensity_batch=logp_batch_slots
    ) isa DensityModel
    # grad slot is mandatory (callable or backend)
    @test_throws ArgumentError DensityModel(logp_slots, nothing, D_SLOTS)

    @test DensityModel(
        logp_slots,
        gradlogp_slots,
        D_SLOTS;
        logdensity_batch=logp_batch_slots,
        hvp_batch=AutoForwardDiff(),
    ) isa DensityModel
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

        model_ad = DensityModel(
            logp_slots, AutoForwardDiff(), D_SLOTS; hvp=AutoForwardDiff()
        )
        m_ad = ParallelMCMC._prepare_model(model_ad, x0, 8, nothing)
        @test m_ad.hvp(x0, v) ≈ -v
    end

    @testset "sampler backend injected as hvp fallback" begin
        v = randn(rng, D_SLOTS)
        model = DensityModel(logp_slots, gradlogp_slots, D_SLOTS)

        m_r = ParallelMCMC._prepare_model(model, x0, 8, AutoForwardDiff())
        @test m_r.hvp(x0, v) ≈ -v

        # no hvp source anywhere
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

    @testset "two-argument form drops the DEER-only slots" begin
        model = DensityModel(
            logp_slots,
            gradlogp_slots,
            D_SLOTS;
            hvp=AutoForwardDiff(),
            logdensity_batch=logp_batch_slots,
            grad_logdensity_batch=AutoForwardDiff(),
            hvp_batch=AutoForwardDiff(),
        )
        m_r = ParallelMCMC._prepare_model(model, x0)
        @test m_r isa ParallelMCMC.PreppedDensityModel
        # no slot of a prepped model is left holding a backend
        @test m_r.hvp === nothing
        @test m_r.grad_logdensity_batch === nothing
        @test m_r.hvp_batch === nothing
    end

    @testset "batched gradient derived from logdensity_batch" begin
        T = 8
        X = randn(rng, D_SLOTS, T)
        V = randn(rng, D_SLOTS, T)

        # grad_logdensity_batch omitted: taken from the gradient slot's backend
        model = DensityModel(
            logp_slots,
            AutoForwardDiff(),
            D_SLOTS;
            hvp=AutoForwardDiff(),
            logdensity_batch=logp_batch_slots,
        )
        m_r = ParallelMCMC._prepare_model(model, x0, T, nothing)
        @test m_r.grad_logdensity_batch(X) ≈ -X
        @test m_r.hvp_batch(X, V) ≈ -V

        #= A hand-written gradient is not opted into AD, so the sampler backend
        does not derive a batched gradient from `logdensity_batch` and the batched
        path stays off. Were it otherwise, `backend=` would decide which update
        path runs. =#
        model_an = DensityModel(
            logp_slots,
            gradlogp_slots,
            D_SLOTS;
            hvp=(x, v) -> -v,        # so the unbatched HVP is covered either way
            logdensity_batch=logp_batch_slots,
        )
        for spl_backend in (AutoForwardDiff(), nothing)
            m_an = ParallelMCMC._prepare_model(model_an, x0, T, spl_backend)
            @test m_an.grad_logdensity_batch === nothing
            @test m_an.hvp_batch === nothing
            # still kept, since `_trajectory_logps` scores a whole block with it
            @test m_an.logdensity_batch === logp_batch_slots
        end
    end

    @testset "hvp_batch backend over a derived batched gradient" begin
        T = 8
        X = randn(rng, D_SLOTS, T)
        V = randn(rng, D_SLOTS, T)

        # no grad_logdensity_batch: derived from the gradient slot's backend
        model_gs = DensityModel(
            logp_slots,
            AutoForwardDiff(),
            D_SLOTS;
            hvp=AutoForwardDiff(),
            logdensity_batch=logp_batch_slots,
            hvp_batch=AutoForwardDiff(),
        )
        m_gs = ParallelMCMC._prepare_model(model_gs, x0, T, nothing)
        @test m_gs.grad_logdensity_batch(X) ≈ -X
        @test m_gs.hvp_batch(X, V) ≈ -V

        #= Hand-written gradient, so nothing derives a batched one, and an
        `hvp_batch` was supplied explicitly: raise instead of silently running
        the unbatched update. The sampler backend does not rescue this. =#
        model_nd = DensityModel(
            logp_slots,
            gradlogp_slots,
            D_SLOTS;
            hvp=(x, v) -> -v,
            logdensity_batch=logp_batch_slots,
            hvp_batch=AutoForwardDiff(),
        )
        for spl_backend in (AutoForwardDiff(), nothing)
            err = try
                ParallelMCMC._prepare_model(model_nd, x0, T, spl_backend)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("no batched gradient", err.msg)
        end
    end

    @testset "model hvp backend feeds the batched HVP without a sampler backend" begin
        T = 8
        X = randn(rng, D_SLOTS, T)
        V = randn(rng, D_SLOTS, T)
        model = DensityModel(
            logp_slots,
            gradlogp_slots,
            D_SLOTS;
            hvp=AutoForwardDiff(),
            logdensity_batch=logp_batch_slots,
            grad_logdensity_batch=gradlogp_batch_slots,
        )
        m_r = ParallelMCMC._prepare_model(model, x0, T, nothing)
        @test m_r.hvp_batch(X, V) ≈ -V
    end
end

@testset "second-order HVP semantics" begin
    #= How an `hvp` backend gets its second derivative depends on the gradient
    slot, so the three cases are told apart with a gradient that is deliberately
    not ∇logp: a path that differentiates the slot sees `-2x` and reports `-2v`,
    one that differentiates `logdensity` twice reports `-v`. =#
    off_grad(x) = -2 .* x
    rng = MersenneTwister(61)
    x0 = randn(rng, D_SLOTS)
    v = randn(rng, D_SLOTS)
    T = 8
    X = randn(rng, D_SLOTS, T)
    V = randn(rng, D_SLOTS, T)

    @testset "plain backend over a hand-written gradient is one pass over it" begin
        #= Both `HVPStrategy` paths stay reachable end to end and keep routing on
        `DI.hvp_mode`: a hand-written gradient is differentiated once, never
        twice, whichever direction the backend reports. `-2v` rather than `-v` is
        what distinguishes that from the second-order paths below. =#
        for backend in (
            AutoForwardDiff(),                      # ForwardOnGrad
            AutoEnzyme(),                           # ForwardOnGrad
            AutoEnzyme(; mode=Enzyme.Reverse),      # ReverseOnGrad
        )
            model = DensityModel(logp_slots, off_grad, D_SLOTS; hvp=backend)
            m_r = ParallelMCMC._prepare_model(model, x0, T, nothing)
            @test m_r.hvp(x0, v) ≈ -2 .* v
        end
    end

    @testset "an explicit SecondOrder differentiates logdensity twice" begin
        #= Both passes are named, so the gradient slot is not the inner one even
        when hand-written, and the inner half is honoured rather than dropped. =#
        model = DensityModel(
            logp_slots,
            off_grad,
            D_SLOTS;
            hvp=DI_SLOTS.SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
        )
        m_r = ParallelMCMC._prepare_model(model, x0, T, nothing)
        @test m_r.hvp(x0, v) ≈ -v

        # same when it arrives as the sampler's fallback backend
        model_fb = DensityModel(logp_slots, off_grad, D_SLOTS)
        m_fb = ParallelMCMC._prepare_model(
            model_fb, x0, T, DI_SLOTS.SecondOrder(AutoForwardDiff(), AutoForwardDiff())
        )
        @test m_fb.hvp(x0, v) ≈ -v
    end

    @testset "a backend over an AD-derived gradient composes into SecondOrder" begin
        #= The gradient slot is a backend, so `hvp` becomes the outer half of a
        true second-order pair rather than an AD pass over the prepared DI
        gradient. Both readings agree numerically here (the derived gradient
        really is ∇logp), so the check is structural: the slot has to hold the
        closure the second-order factory builds, not the one a strategy builds. =#
        second_order = ParallelMCMC.DEER._make_hvp_fn_second_order(
            logp_slots, DI_SLOTS.SecondOrder(AutoForwardDiff(), AutoForwardDiff()), x0
        )

        model = DensityModel(logp_slots, AutoForwardDiff(), D_SLOTS; hvp=AutoForwardDiff())
        m_r = ParallelMCMC._prepare_model(model, x0, T, nothing)
        @test typeof(m_r.hvp) === typeof(second_order)
        @test m_r.hvp(x0, v) ≈ -v
        # the resolved gradient is still the prepared AD one
        @test m_r.grad_logdensity isa ParallelMCMC._ADGradient
        @test m_r.grad_logdensity(x0) ≈ -x0

        # a hand-written gradient keeps the single-pass strategy closure instead
        model_hand = DensityModel(logp_slots, off_grad, D_SLOTS; hvp=AutoForwardDiff())
        m_hand = ParallelMCMC._prepare_model(model_hand, x0, T, nothing)
        @test typeof(m_hand.hvp) !== typeof(second_order)
    end

    @testset "batched HVP follows the same three cases" begin
        off_grad_batch(X) = -2 .* X

        model_hand = DensityModel(
            logp_slots,
            off_grad,
            D_SLOTS;
            hvp=(x, vv) -> -vv,
            logdensity_batch=logp_batch_slots,
            grad_logdensity_batch=off_grad_batch,
            hvp_batch=AutoForwardDiff(),
        )
        m_hand = ParallelMCMC._prepare_model(model_hand, x0, T, nothing)
        @test m_hand.hvp_batch(X, V) ≈ -2 .* V

        model_so = DensityModel(
            logp_slots,
            off_grad,
            D_SLOTS;
            hvp=(x, vv) -> -vv,
            logdensity_batch=logp_batch_slots,
            grad_logdensity_batch=off_grad_batch,
            hvp_batch=DI_SLOTS.SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
        )
        m_so = ParallelMCMC._prepare_model(model_so, x0, T, nothing)
        @test m_so.hvp_batch(X, V) ≈ -V

        model_ad = DensityModel(
            logp_slots,
            AutoForwardDiff(),
            D_SLOTS;
            hvp=AutoForwardDiff(),
            logdensity_batch=logp_batch_slots,
            grad_logdensity_batch=AutoForwardDiff(),
            hvp_batch=AutoForwardDiff(),
        )
        m_ad = ParallelMCMC._prepare_model(model_ad, x0, T, nothing)
        @test m_ad.hvp_batch(X, V) ≈ -V
    end

    @testset "sampling with a SecondOrder hvp matches the analytic HVP" begin
        model_so = DensityModel(
            logp_slots,
            gradlogp_slots,
            D_SLOTS;
            hvp=DI_SLOTS.SecondOrder(AutoForwardDiff(), AutoForwardDiff()),
        )
        model_an = DensityModel(logp_slots, gradlogp_slots, D_SLOTS; hvp=(x, vv) -> -vv)
        s = ParallelMALASampler(0.05; T=16)
        c_so = sample(
            MersenneTwister(62), model_so, s, 64; chain_type=CT_SLOTS, progress=false
        )
        c_an = sample(
            MersenneTwister(62), model_an, s, 64; chain_type=CT_SLOTS, progress=false
        )
        @test c_so[:x] ≈ c_an[:x]
    end
end

@testset "logp-only sampling matches analytic-gradient sampling" begin
    model_ad = DensityModel(logp_slots, AutoForwardDiff(), D_SLOTS)
    model_an = DensityModel(logp_slots, gradlogp_slots, D_SLOTS)

    @testset "MALASampler" begin
        c_ad = sample(
            MersenneTwister(21),
            model_ad,
            MALASampler(0.2),
            100;
            chain_type=CT_SLOTS,
            progress=false,
        )
        c_an = sample(
            MersenneTwister(21),
            model_an,
            MALASampler(0.2),
            100;
            chain_type=CT_SLOTS,
            progress=false,
        )
        @test c_ad[:x] ≈ c_an[:x]
    end

    @testset "AdaptiveMALASampler" begin
        c_ad = sample(
            MersenneTwister(22),
            model_ad,
            AdaptiveMALASampler(0.2; n_warmup=50),
            100;
            chain_type=CT_SLOTS,
            progress=false,
        )
        c_an = sample(
            MersenneTwister(22),
            model_an,
            AdaptiveMALASampler(0.2; n_warmup=50),
            100;
            chain_type=CT_SLOTS,
            progress=false,
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
        MersenneTwister(42),
        model,
        MALASampler(0.2),
        100;
        chain_type=CT_SLOTS,
        progress=false,
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

@testset "a state from another model does not outrank the model passed to step" begin
    #= The prepped model rides along in the state so preparation is reused, but
    `initial_state` can hand a state to `step` on a different model. The model
    argument has to win, or the wrong target gets sampled silently. =#
    logp_tight(x) = -2.0 * dot(x, x)      # N(0, 1/2) rather than N(0, 1)
    model_a = DensityModel(logp_slots, AutoForwardDiff(), D_SLOTS)
    model_b = DensityModel(logp_tight, AutoForwardDiff(), D_SLOTS)

    @testset "MALASampler" begin
        spl = MALASampler(0.2)
        _, s_a = ParallelMCMC.AbstractMCMC.step(MersenneTwister(71), model_a, spl)
        @test s_a.model.source === model_a

        _, s_b = ParallelMCMC.AbstractMCMC.step(MersenneTwister(72), model_b, spl, s_a)
        @test s_b.model.source === model_b
        @test s_b.model.logdensity === logp_tight
        # re-prepared once, then reused
        _, s_b2 = ParallelMCMC.AbstractMCMC.step(MersenneTwister(73), model_b, spl, s_b)
        @test s_b2.model === s_b.model
    end

    @testset "AdaptiveMALASampler" begin
        spl = AdaptiveMALASampler(0.2; n_warmup=10)
        _, s_a = ParallelMCMC.AbstractMCMC.step(MersenneTwister(74), model_a, spl)
        _, s_b = ParallelMCMC.AbstractMCMC.step(MersenneTwister(75), model_b, spl, s_a)
        @test s_b.model.source === model_b
    end

    @testset "ParallelMALASampler" begin
        spl = ParallelMALASampler(0.05; T=8, backend=AutoForwardDiff())
        _, s_a = ParallelMCMC.AbstractMCMC.step(MersenneTwister(76), model_a, spl)
        @test s_a.t == 1
        #= What replaying model A's trajectory would have handed back. Snapshot
        it: a re-solve writes through the same workspace. =#
        x_replayed = copy(s_a.trajectory[:, 2])
        x_resume = copy(s_a.x)

        # the cached trajectory belongs to model A, so it is discarded, not replayed
        _, s_b = ParallelMCMC.AbstractMCMC.step(MersenneTwister(77), model_b, spl, s_a)
        @test s_b.model.source === model_b
        @test s_b.t == 1
        @test !(s_b.x ≈ x_replayed)

        # and what it solved instead is model B's own block from that point
        _, s_fresh = ParallelMCMC.AbstractMCMC.step(
            MersenneTwister(77), model_b, spl; initial_params=x_resume
        )
        @test s_b.x ≈ s_fresh.x
    end

    @testset "sample with initial_state follows the model it is given" begin
        spl = MALASampler(0.2)
        _, s_a = ParallelMCMC.AbstractMCMC.step(MersenneTwister(78), model_a, spl)

        # resuming from model A's state must match a fresh model B run from the
        # same point: the initial step draws no noise, so the streams line up
        c_resumed = sample(
            MersenneTwister(79),
            model_b,
            spl,
            4;
            initial_state=s_a,
            chain_type=CT_SLOTS,
            progress=false,
        )
        c_fresh = sample(
            MersenneTwister(79),
            model_b,
            spl,
            5;
            initial_params=copy(s_a.x),
            chain_type=CT_SLOTS,
            progress=false,
        )
        @test all(c_resumed[:x][i] ≈ c_fresh[:x][i + 1] for i in 1:4)
    end
end

@testset "sampler backend is optional when the model specifies hvp" begin
    model = DensityModel(logp_slots, AutoForwardDiff(), D_SLOTS; hvp=AutoForwardDiff())
    s = ParallelMALASampler(0.05; T=16)   # no backend
    chain = sample(MersenneTwister(61), model, s, 64; chain_type=CT_SLOTS, progress=false)
    @test size(chain) == (64, 1)
    @test all(x -> all(isfinite, x), chain[:x])

    # neither a model hvp nor a sampler backend: errors when sampling starts
    bare = DensityModel(logp_slots, gradlogp_slots, D_SLOTS)
    @test_throws ArgumentError sample(
        MersenneTwister(62), bare, s, 32; chain_type=CT_SLOTS, progress=false
    )
end
