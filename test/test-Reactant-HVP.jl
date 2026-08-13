using Test
using Random
using LinearAlgebra
using Statistics
using FlexiChains

using ParallelMCMC
using ADTypes
# Stands in for "some backend that is not Reactant" in the pairing tests below.
using ForwardDiff: ForwardDiff
using LogDensityProblems: LogDensityProblems

const DI_R = ParallelMCMC.DEER.DI

#=
Reactant-compiled derivative paths (`AutoReactant`), see ext/ReactantExt.jl.

The quartic target keeps the second-order structure honest: logp = -0.25‖x‖⁴ has
H = -(‖x‖² I + 2 x xᵀ), so an HVP that drops the second-order term — the failure
mode of `Enzyme.hvp` under `@compile` — is caught, where a Gaussian's constant H
would hide it. Every derivative-accuracy testset below uses it; do not swap in a
Gaussian.
=#
logp_r(x) = -0.25 * sum(abs2, x)^2
gradlogp_r(x) = -sum(abs2, x) .* x
hvp_r(x, v) = -(sum(abs2, x) .* v .+ 2 .* dot(x, v) .* x)
logp_batch_r(X) = vec(-0.25 .* sum(abs2, X; dims=1) .^ 2)
gradlogp_batch_r(X) = -X .* sum(abs2, X; dims=1)
logp_r32(x) = -0.25f0 * sum(abs2, x)^2

#= Standard Gaussian, constant Hessian (H = -I). Only for the end-to-end sampling
tests, which ask whether the sampler converged to the right posterior; whether
the HVP is second-order-correct is the quartic target's job. =#
logp_gauss(x) = -0.5 * sum(abs2, x)
gradlogp_gauss(x) = -x
hvp_gauss(x, v) = -v

const D_R = 4
const CT_R = FlexiChains.FlexiChain{Symbol}

#= The pairing rule lives in `_resolve_hvp` / `_prepare_model`, not in the
extension, so its dispatch table is checked whether or not Reactant loads —
none of these calls resolve a gradient or compile anything. =#
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

#= `SecondOrder(AutoReactant(), AutoReactant())`, or `AutoReactant()` paired with
a `SecondOrder` at all, is a natural thing to try given the pairing table in
10-getting-started.md, and would otherwise land in `DI.prepare_hvp` several
frames deep with no Reactant support. A dispatch-table property, so this runs
whether or not Reactant is loaded. =#
@testset "AutoReactant cannot appear inside a SecondOrder" begin
    # Sanity: an ordinary SecondOrder is unaffected.
    @test ParallelMCMC._check_reactant_pair(
        nothing, DI_R.SecondOrder(AutoForwardDiff(), AutoForwardDiff())
    ) === nothing

    @test_throws ArgumentError ParallelMCMC._check_reactant_pair(
        nothing, DI_R.SecondOrder(AutoReactant(), AutoReactant())
    )
    @test_throws ArgumentError ParallelMCMC._check_reactant_pair(
        AutoForwardDiff(), DI_R.SecondOrder(AutoReactant(), AutoForwardDiff())
    )
    # (AutoReactant, SecondOrder) exercises the disambiguating method directly.
    @test_throws ArgumentError ParallelMCMC._check_reactant_pair(
        AutoReactant(), DI_R.SecondOrder(AutoReactant(), AutoReactant())
    )
end

#= A `LogDensityProblemGradient` is a callable, so it clears the
`grad_backend === nothing` test that otherwise means "hand-written gradient",
but it is not Reactant-traceable. Checked on the type first, then through a real
LogDensityProblems model. Neither needs Reactant loaded: the check fires before
any gradient is resolved or anything compiled. =#
@testset "a LogDensityProblems gradient cannot pair with an AutoReactant hvp" begin
    @test ParallelMCMC._check_reactant_hvp_source(
        ParallelMCMC.LogDensityProblemGradient(nothing), AutoForwardDiff()
    ) === nothing
    @test_throws ArgumentError ParallelMCMC._check_reactant_hvp_source(
        ParallelMCMC.LogDensityProblemGradient(nothing), AutoReactant()
    )

    struct _FakeLD end
    LogDensityProblems.capabilities(::_FakeLD) = LogDensityProblems.LogDensityOrder{1}()
    LogDensityProblems.dimension(::_FakeLD) = D_R
    function LogDensityProblems.logdensity_and_gradient(::_FakeLD, x)
        return logp_r(x), gradlogp_r(x)
    end

    model = DensityModel(_FakeLD(); hvp=AutoReactant())
    @test_throws ArgumentError ParallelMCMC._prepare_model(model, zeros(D_R), 8, nothing)
end

#= Outside the `reactant_ok` guard below. `_check_reactant_pair` now runs above
gradient resolution in `_prepare_model`, so a mismatched pair is caught before
anything is resolved or compiled and no Reactant install is needed. This used to
pay a full XLA compile per `@test_throws`, ~18s, to demonstrate a config error. =#
@testset "mixed pairs are refused at preparation" begin
    x = zeros(D_R)

    reactant_grad = DensityModel(logp_r, AutoReactant(), D_R; hvp=AutoForwardDiff())
    @test_throws ArgumentError ParallelMCMC._prepare_model(reactant_grad, x, 8, nothing)

    reactant_hvp = DensityModel(logp_r, AutoForwardDiff(), D_R; hvp=AutoReactant())
    @test_throws ArgumentError ParallelMCMC._prepare_model(reactant_hvp, x, 8, nothing)
end

reactant_ok = try
    using Reactant: Reactant
    using Enzyme: Enzyme
    true
catch err
    @warn "Reactant not available — skipping Reactant HVP tests" err
    false
end

#= The `_REACTANT_LOAD_HINT` fallbacks (src/DEER/DEER.jl, and
`_reactant_resolve_gradient` / `_reactant_resolve_gradient_batch` in
src/interface.jl) can only be asserted in a session without Reactant: once
`ReactantExt` is loaded its more-specific methods shadow all of them and the
error can never fire. Hence the one testset here guarded on `!reactant_ok`. =#
if !reactant_ok
    @testset "clear load-hint error without Reactant loaded" begin
        #= Gradient slot: `_reactant_resolve_gradient`'s fallback. `backend` (the
        4th arg) is `AutoReactant()` too, as `_check_reactant_pair` requires, so
        `_prepare_model` gets as far as gradient resolution rather than failing
        first on the model having no HVP source — a correct failure, but not the
        one being pinned here. =#
        model_grad = DensityModel(logp_r, AutoReactant(), D_R)
        @test_throws "AutoReactant requires Reactant.jl" ParallelMCMC._prepare_model(
            model_grad, zeros(D_R), 8, AutoReactant()
        )

        # HVP slot over a hand-written gradient: `DEER._make_hvp_fn`'s
        # `ReactantHVP` fallback, reached via `_hvp_strategy(::AutoReactant)`.
        model_hvp = DensityModel(logp_r, gradlogp_r, D_R; hvp=AutoReactant())
        @test_throws "AutoReactant requires Reactant.jl" ParallelMCMC._prepare_model(
            model_hvp, zeros(D_R), 8, nothing
        )
    end
end

if reactant_ok
    @testset "extension is loaded" begin
        @test Base.get_extension(ParallelMCMC, :ReactantExt) !== nothing
    end

    #= The extension does not honour `AutoReactant.mode` (the wrapped
    `AutoEnzyme`): gradients always trace reverse, HVPs forward-over-that. A
    non-default mode is rejected rather than ignored. =#
    @testset "a non-default AutoReactant mode is rejected" begin
        bad = AutoReactant(; mode=AutoEnzyme(; mode=Enzyme.Forward))
        model_grad = DensityModel(logp_r, bad, D_R)
        @test_throws ArgumentError ParallelMCMC._prepare_model(
            model_grad, zeros(D_R), 8, nothing
        )

        model_hvp = DensityModel(logp_r, gradlogp_r, D_R; hvp=bad)
        @test_throws ArgumentError ParallelMCMC._prepare_model(
            model_hvp, zeros(D_R), 8, nothing
        )

        # The default still works.
        @test DensityModel(logp_r, AutoReactant(), D_R) isa DensityModel
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

        #= A hand-written batched gradient with `hvp_batch=AutoReactant()`, the
        batched analogue of the "forward over user gradient" case above. Routes
        through
        `DEER._make_hvp_batch_fn(::ReactantHVP, grad_batch, ::AutoReactant, ...)`,
        which nothing else here reaches. =#
        @testset "forward over user batched gradient (hvp_batch=AutoReactant())" begin
            T = 8
            X = randn(rng, D_R, T)
            V = randn(rng, D_R, T)
            Hv_cols = reduce(hcat, [hvp_r(X[:, t], V[:, t]) for t in 1:T])

            model = DensityModel(
                logp_r,
                gradlogp_r,
                D_R;
                hvp=hvp_r,
                logdensity_batch=logp_batch_r,
                grad_logdensity_batch=gradlogp_batch_r,
                hvp_batch=AutoReactant(),
            )
            m_p = ParallelMCMC._prepare_model(model, X[:, 1], T, nothing)
            @test m_p.hvp_batch(X, V) ≈ Hv_cols
        end

        #= Edge shapes. Both slots AutoReactant, as in the
        "forward-over-reverse from logp alone" case above, but at the smallest
        sizes DEER ever prepares. =#
        @testset "edge shapes" begin
            @testset "D=1" begin
                x1 = randn(rng, 1)
                v1 = randn(rng, 1)
                model = DensityModel(logp_r, AutoReactant(), 1; hvp=AutoReactant())
                m_p = ParallelMCMC._prepare_model(model, x1, 8, nothing)
                @test m_p.grad_logdensity(x1) ≈ gradlogp_r(x1)
                @test m_p.hvp(x1, v1) ≈ hvp_r(x1, v1)
            end

            @testset "T=1" begin
                T = 1
                X = reshape(x, D_R, T)
                V = reshape(v, D_R, T)
                model = DensityModel(
                    logp_r,
                    AutoReactant(),
                    D_R;
                    logdensity_batch=logp_batch_r,
                    grad_logdensity_batch=AutoReactant(),
                    hvp=AutoReactant(),
                    hvp_batch=AutoReactant(),
                )
                m_p = ParallelMCMC._prepare_model(model, x, T, nothing)
                @test m_p.grad_logdensity_batch(X) ≈ gradlogp_batch_r(X)
                @test m_p.hvp_batch(X, V) ≈ reshape(hvp_r(x, v), D_R, T)
            end
        end

        #= Float32 on the CPU path. Previously only exercised inside the
        CuArray-only block below, so it never ran without a functional CUDA. =#
        @testset "Float32 on CPU" begin
            x32 = Float32.(x)
            v32 = Float32.(v)
            model = DensityModel(logp_r32, AutoReactant(), D_R; hvp=AutoReactant())
            m_p = ParallelMCMC._prepare_model(model, x32, 8, nothing)

            g = m_p.grad_logdensity(x32)
            @test eltype(g) === Float32
            @test g ≈ gradlogp_r(x32)

            Hv = m_p.hvp(x32, v32)
            @test eltype(Hv) === Float32
            @test Hv ≈ hvp_r(x32, v32)
        end
    end

    #= Pins the "captured data is frozen at preparation time" limitation from
    ext/ReactantExt.jl's module docstring and docs/src/15-gpu.md: `@compile`
    bakes a captured plain `Array` in as a compile-time constant, so mutating it
    afterwards has NO effect on later calls, with no error and no warning. Here
    so that a future Reactant which does detect this gets noticed. Model code
    still should not close over mutable data.
    =#
    @testset "documented caveat: captured data is frozen at preparation time" begin
        data = [1.0, 1.0, 1.0, 1.0]
        f(x) = -0.5 * sum(abs2, x .- data)   # ∇f(x) = data - x
        model = DensityModel(f, AutoReactant(), D_R)
        # Two-argument `_prepare_model` only resolves `grad_logdensity`, which
        # is all this test needs (no HVP involved).
        m_p = ParallelMCMC._prepare_model(model, zeros(D_R))

        x0 = zeros(D_R)
        g_before = m_p.grad_logdensity(x0)
        @test g_before ≈ [1.0, 1.0, 1.0, 1.0]

        data .= 5.0   # mutate the captured array *after* preparation

        g_after = m_p.grad_logdensity(x0)
        #= Frozen at the pre-mutation value: NOT [5, 5, 5, 5], which is what a
        correct re-evaluation against the mutated `data` would give. =#
        @test g_after ≈ [1.0, 1.0, 1.0, 1.0]
    end

    #= `size(chain) == (N,1)` and `all(isfinite, ...)` pass even for a badly
    wrong HVP: DEER's Newton iteration just fails to converge and `DEER.solve`
    returns the non-converged trajectory, no NaNs involved. Assert the posterior
    mean against a target with a known mean instead (the convention in
    test-GPU-AD-HVP.jl), and cross-check against the same model driven by the
    analytic HVP on the same noise tape.
    =#
    @testset "end-to-end sampling with AutoReactant" begin
        @testset "posterior mean recovery (standard Gaussian, mean 0)" begin
            model = DensityModel(logp_gauss, AutoReactant(), D_R)
            s = ParallelMALASampler(0.3; T=16, backend=AutoReactant())
            n_samples, n_burn = 2000, 500
            chain = sample(
                MersenneTwister(72), model, s, n_samples; chain_type=CT_R, progress=false
            )
            @test size(chain) == (n_samples, 1)

            xs = chain[:x]
            @test all(x -> all(isfinite, x), xs)
            post_mean = vec(mean(reduce(hcat, xs[(n_burn + 1):end]); dims=2))
            @test maximum(abs, post_mean) < 0.3
        end

        @testset "matches analytic-HVP DEER on the same noise tape" begin
            s = ParallelMALASampler(0.1; T=16, backend=AutoReactant())
            model_r = DensityModel(logp_gauss, AutoReactant(), D_R)
            model_an = DensityModel(logp_gauss, gradlogp_gauss, D_R; hvp=hvp_gauss)

            c_r = sample(
                MersenneTwister(99), model_r, s, 64; chain_type=CT_R, progress=false
            )
            c_an = sample(
                MersenneTwister(99), model_an, s, 64; chain_type=CT_R, progress=false
            )
            @test c_r[:x] ≈ c_an[:x]
        end
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
        The compiled executable lives in Reactant's own (XLA) device memory, so
        what this checks is the CuArray <-> Reactant marshalling boundary:
        CuArray in, CuArray out, values matching the analytic HVP. It does NOT
        establish that the HVP executes on the GPU, which depends on Reactant's
        default XLA client (see docs/src/15-gpu.md) and is neither controlled
        nor asserted here.
        =#
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
