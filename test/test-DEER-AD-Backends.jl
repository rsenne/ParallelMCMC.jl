using Test
using Random
using LinearAlgebra

using ParallelMCMC
using ADTypes: ADTypes

_logp(x) = -0.5 * dot(x, x)
_gradlogp(x) = -x

function _make_tape_and_ref(rng, D, T, ε)
    ξs = [randn(rng, D) for _ in 1:T]
    us = rand(rng, T)
    x0 = randn(rng, D)
    xs_seq = ParallelMCMC.MALA.run_mala_sequential_taped(_logp, _gradlogp, x0, ε, ξs, us)
    tape = [(ξ=ξs[t], u=us[t]) for t in 1:T]
    return x0, tape, reduce(hcat, xs_seq[2:end])
end

function _make_rec(tape, ε, backend)
    step_fwd =
        (x, te) -> ParallelMCMC.MALA.mala_step_taped(_logp, _gradlogp, x, ε, te.ξ, te.u)
    step_lin =
        (x, te, a) -> ParallelMCMC.MALA.mala_step_surrogate(_logp, _gradlogp, x, ε, te.ξ, a)
    consts =
        (x, te) ->
            (ParallelMCMC.MALA.mala_accept_indicator(_logp, _gradlogp, x, ε, te.ξ, te.u),)
    return ParallelMCMC.DEER.TapedRecursion(
        step_fwd, step_lin, tape; consts=consts, const_example=(0.0,), backend=backend
    )
end

#=
Build the backend list dynamically.  Each backend is wrapped in try/catch so
a missing package (or DI extension not loaded) just emits a warning and skips
rather than failing the whole suite.
=#
_all_backends = Pair{String,ADTypes.AbstractADType}[]

# ForwardDiff — always in test deps
push!(_all_backends, "ForwardDiff" => ADTypes.AutoForwardDiff())

# Mooncake — always in project deps
try
    using Mooncake: Mooncake
    push!(_all_backends, "Mooncake" => ADTypes.AutoMooncake(; config=nothing))
catch e
    @warn "Mooncake not loadable, skipping" exception=e
end

#=
Enzyme on CPU has a known Windows issue: LLVM stack-frame unwinding triggers
EXCEPTION_ACCESS_VIOLATION in RtlVirtualUnwind2 (ntdll.dll).  This is a
platform bug in the pinned Enzyme version (0.13.131), not in our code.
Enzyme is still tested on GPU via test-GPU-DEER.jl (CUDA path).
=#
if !Sys.iswindows()
    try
        using Enzyme: Enzyme
        push!(_all_backends, "Enzyme" => ADTypes.AutoEnzyme())
    catch e
        @warn "Enzyme not loadable, skipping" exception=e
    end
end

# ReverseDiff
try
    using ReverseDiff: ReverseDiff
    push!(_all_backends, "ReverseDiff" => ADTypes.AutoReverseDiff())
catch e
    @warn "ReverseDiff not loadable, skipping" exception=e
end

# Zygote — DI's Zygote extension also loads ForwardDiff for its pushforward path
try
    using Zygote: Zygote
    push!(_all_backends, "Zygote" => ADTypes.AutoZygote())
catch e
    @warn "Zygote not loadable, skipping" exception=e
end

#=
For :stoch_diag mode, DEER calls DI.pushforward.  Not all backends expose this
natively; DI may or may not provide a fallback.  We probe each loaded backend
at test time: if prepare_pushforward succeeds, we include it in the list.
=#
D_PROBE, T_PROBE = 3, 4
ε_PROBE = 0.1
_probe_rng = MersenneTwister(0)
_probe_x0, _probe_tape, _ = _make_tape_and_ref(_probe_rng, D_PROBE, T_PROBE, ε_PROBE)

_pf_backends = Pair{String,ADTypes.AbstractADType}[]
for (bname, backend) in _all_backends
    try
        rec = _make_rec(_probe_tape, ε_PROBE, backend)
        ParallelMCMC.DEER.prepare_pushforward(rec, _probe_x0)
        push!(_pf_backends, bname => backend)
    catch e
        @warn "Backend $bname does not support pushforward via DI, skipping :stoch_diag tests" exception=e
    end
end

@testset "DEER.solve CPU backend=$bname (:diag)" for (bname, backend) in _all_backends
    rng = MersenneTwister(42)
    D, T = 4, 24
    ε = 0.1

    x0, tape, S_ref = _make_tape_and_ref(rng, D, T, ε)
    rec = _make_rec(tape, ε, backend)

    S, info = ParallelMCMC.DEER.solve(
        rec,
        x0;
        jacobian=:diag,
        damping=0.5,
        tol_abs=1e-8,
        tol_rel=1e-6,
        maxiter=200,
        return_info=true,
    )

    @test info.converged
    @test size(S) == (D, T)
    @test all(isfinite, S)
    @test S ≈ S_ref rtol=1e-4 atol=1e-5
end

@testset "DEER.solve CPU backend=$bname (:stoch_diag)" for (bname, backend) in _pf_backends
    rng = MersenneTwister(42)
    D, T = 4, 24
    ε = 0.1

    x0, tape, S_ref = _make_tape_and_ref(rng, D, T, ε)
    rec = _make_rec(tape, ε, backend)

    S, info = ParallelMCMC.DEER.solve(
        rec,
        x0;
        jacobian=:stoch_diag,
        probes=2,
        rng=MersenneTwister(123),
        damping=0.5,
        tol_abs=1e-6,
        tol_rel=1e-5,
        maxiter=400,
        return_info=true,
    )

    @test info.converged
    @test size(S) == (D, T)
    @test all(isfinite, S)
    @test S ≈ S_ref rtol=1e-3 atol=1e-4
end

@testset "DEER.solve CPU backend=$bname (:full)" for (bname, backend) in _all_backends
    rng = MersenneTwister(42)
    D, T = 4, 24
    ε = 0.1

    x0, tape, S_ref = _make_tape_and_ref(rng, D, T, ε)
    rec = _make_rec(tape, ε, backend)

    S, info = ParallelMCMC.DEER.solve(
        rec,
        x0;
        jacobian=:full,
        damping=0.5,
        tol_abs=1e-10,
        tol_rel=1e-8,
        maxiter=200,
        return_info=true,
    )

    @test info.converged
    @test size(S) == (D, T)
    @test all(isfinite, S)
    @test S ≈ S_ref rtol=1e-5 atol=1e-6
end

@testset "DEERSampler step CPU backend=$bname" for (bname, backend) in _all_backends
    rng = MersenneTwister(7)
    D = 3
    model = DensityModel(_logp, _gradlogp, D)
    sampler = DEERSampler(0.1; T=8, maxiter=200, jacobian=:diag, backend=backend)

    trans, state = ParallelMCMC.AbstractMCMC.step(
        rng, model, sampler; initial_params=randn(rng, D)
    )

    @test trans isa DEERTransition
    @test length(trans.x) == D
    @test isfinite(trans.logp)
    @test size(state.trajectory) == (D, 8)
end
