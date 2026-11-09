module ReactantExt

# Reactant treats captured arrays as compile-time constants. Traced model
# functions must not depend on later mutations to captured data.

using ParallelMCMC: ParallelMCMC
using ParallelMCMC.DEER: DEER
using ParallelMCMC.MALA: MALA
using ADTypes: ADTypes, AutoReactant, AutoEnzyme
using Reactant: Reactant, @compile
using Enzyme: Enzyme
using LinearAlgebra: LinearAlgebra
using Random: Random

_check_reactant_mode(::AutoReactant{AutoEnzyme{Nothing,Nothing}}) = nothing
function _check_reactant_mode(backend::AutoReactant)
    return throw(
        ArgumentError(
            "AutoReactant mode $(backend.mode) is not supported; use AutoReactant()"
        ),
    )
end

function _reactant_client_platform()
    return string(Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
end

function _warn_reactant_host_roundtrip(x::AbstractArray)
    if ParallelMCMC.needs_host_staging(x) && _reactant_client_platform() == "cpu"
        @warn "AutoReactant: preparing on a $(typeof(x)), but Reactant's default XLA " *
            "client targets \"cpu\". Every call will copy to the host and back. " *
            "Select a GPU client with `Reactant.set_default_backend(\"gpu\")` if one " *
            "is available." maxlog = 1
    end
    return nothing
end

#=
Every call uploads through `to_rarray`. Caching the XLA buffer instead would be
slower: Reactant's `copyto!` into an existing ConcreteRArray uploads to a new
buffer and then runs a compiled device copy on top of it.

XLA takes the host stage with `kImmutableOnlyDuringCall` semantics, so the stage
is free to overwrite once `to_rarray` returns. `to_rarray` wants an `Array`, so
views get collected first, as in `DEER._materialize_ad_array`.
=#
_upload(x::Array, ::Nothing) = Reactant.to_rarray(x)
_upload(x::AbstractArray, ::Nothing) = Reactant.to_rarray(Array(x))
#= XLA only ever sees the stage, so a short `x` would leave its tail holding the
previous call's values and the compiled function would run on them. Uploading
`x` directly would have failed XLA's shape check instead. =#
function _upload(x::AbstractArray, stage::AbstractArray)
    size(x) == size(stage) || throw(
        DimensionMismatch(
            "AutoReactant compiled for input size $(size(stage)), got $(size(x))"
        ),
    )
    copyto!(stage, x)
    return Reactant.to_rarray(stage)
end

# Host arrays upload directly and need no stage.
function _in_stage(template::AbstractArray)
    ParallelMCMC.needs_host_staging(template) || return nothing
    return ParallelMCMC._host_staging_buffer(template, eltype(template), size(template))
end

# Keep any promotion the compiled function performed. Callers hold on to
# gradients, so every call returns a fresh array.
function _promote_like(template::AbstractArray, out_h::Array)
    template isa Array && eltype(out_h) === eltype(template) && return out_h
    res = similar(template, eltype(out_h), size(out_h))
    copyto!(res, out_h)
    return res
end

# Device pointer to an XLA buffer, or `nothing` if it cannot be taken: pointer
# access needs an unsharded PJRT buffer that is not already on the host.
function _device_pointer(out)
    out isa Reactant.ConcretePJRTArray || return nothing
    Reactant.Sharding.is_sharded(out.sharding) && return nothing
    wait(out)
    buf = Reactant.get_buffer(out)
    Reactant.XLA.buffer_on_cpu(buf) && return nothing
    return Reactant.XLA.unsafe_buffer_pointer(buf)
end

function _download(template::AbstractArray, out, platform::AbstractString)
    ptr = ParallelMCMC.needs_host_staging(template) ? _device_pointer(out) : nothing
    if ptr !== nothing
        res = similar(template, eltype(out), size(out))
        GC.@preserve out begin
            ParallelMCMC._copy_from_device_pointer!(res, ptr, platform) && return res
        end
    end
    return _promote_like(template, Array(out))
end

# One per compiled function, owning a staging buffer per argument. `template`
# is the first argument, whose array type the result is rebuilt as.
# `platform` is read once, at compile time, from the default XLA client.
struct ReactantCall{F,S,T}
    compiled::F
    stages::S
    template::T
    platform::String
end

#=
XLA buffers are freed by GC finalizers. Reusing stages creates very little host
garbage, so GC may not run before device memory fills up. Finalize buffers as
soon as they are no longer needed; PJRT waits for any pending work before freeing
them. Finalization runs only once, even if a compiled function returns an input
wrapped in a new array. Donated inputs no longer own their buffers, so skip them.
Free on the error path too, or a model that throws on every call keeps its
buffers until GC.
=#
function _free!(x)
    x isa Reactant.ConcretePJRTArray || return nothing
    (x.donated || Reactant.Sharding.is_sharded(x.sharding)) && return nothing
    finalize(Reactant.get_buffer(x))
    return nothing
end

function (c::ReactantCall)(args::AbstractArray...)
    inputs = map(_upload, args, c.stages)
    out = nothing
    try
        out = c.compiled(inputs...)
        return _download(c.template, out, c.platform)
    finally
        foreach(_free!, inputs)
        _free!(out)
    end
end

#= XLA's DEFAULT precision lets the GPU run Float32 matrix products in TF32,
which puts a relative error of about 3e-4 into batched derivatives and stalls DEER
convergence (#83). The precision is read while tracing, so it covers the
products Enzyme adds for the derivative too. =#
function _compile_full_precision(core, inputs...)
    return Reactant.with_config(;
        dot_general_precision=Reactant.PrecisionConfig.HIGHEST,
        convolution_precision=Reactant.PrecisionConfig.HIGHEST,
    ) do
        @compile core(inputs...)
    end
end

function _compiled(core, templates::AbstractArray...)
    _warn_reactant_host_roundtrip(templates[1])
    stages = map(_in_stage, templates)
    inputs = map(_upload, templates, stages)
    compiled = _compile_full_precision(core, inputs...)
    return ReactantCall(compiled, stages, templates[1], _reactant_client_platform())
end

# For g = gradlogp, this JVP is the HVP. The callable and its captures are constant.
function _jvp(g, x, v)
    return only(Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(g), Enzyme.Duplicated(x, v)))
end

#= `Base.Fix1` only accepts trailing arguments on 1.12+; this is the same partial
application, spelled for the 1.10 compat floor. =#
struct JVPWith{G}
    g::G
end
(c::JVPWith)(x, v) = _jvp(c.g, x, v)

_rev_gradient(f, x) = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), x)[1]

# XLA shares `logdensity_batch`'s products with the gradient's forward pass.
struct ValueGradHVPBatch{F}
    logdensity_batch::F
end
function (c::ValueGradHVPBatch)(X, V)
    g = Base.Fix1(_rev_gradient, ParallelMCMC._BatchLogdensitySum(c.logdensity_batch))
    Hv, G = Enzyme.autodiff(
        Enzyme.ForwardWithPrimal, Enzyme.Const(g), Enzyme.Duplicated(X, V)
    )
    return c.logdensity_batch(X), G, Hv
end

function ParallelMCMC._reactant_resolve_gradient(
    logdensity, backend::AutoReactant, x_template::AbstractVector
)
    _check_reactant_mode(backend)
    core = Base.Fix1(_rev_gradient, logdensity)
    return _compiled(core, x_template)
end

function ParallelMCMC._reactant_resolve_gradient_batch(
    logdensity_batch, backend::AutoReactant, X_template::AbstractMatrix
)
    _check_reactant_mode(backend)
    core = Base.Fix1(_rev_gradient, ParallelMCMC._BatchLogdensitySum(logdensity_batch))
    return _compiled(core, X_template)
end

# AD-derived gradients differentiate the log-density; callable gradients use a JVP.
function DEER._make_hvp_fn_second_order(
    logdensity, backend::AutoReactant, x_template::AbstractVector
)
    _check_reactant_mode(backend)
    inner = Base.Fix1(_rev_gradient, logdensity)
    core = JVPWith(inner)
    return _compiled(core, x_template, x_template)
end

function DEER._make_hvp_fn(
    ::DEER.ReactantHVP, gradlogp, backend::AutoReactant, x_template::AbstractVector
)
    _check_reactant_mode(backend)
    core = JVPWith(gradlogp)
    return _compiled(core, x_template, x_template)
end

function DEER._make_hvp_batch_fn_second_order(
    logdensity_batch_sum, backend::AutoReactant, X_template::AbstractMatrix
)
    _check_reactant_mode(backend)
    inner = Base.Fix1(_rev_gradient, logdensity_batch_sum)
    core = JVPWith(inner)
    return _compiled(core, X_template, X_template)
end

function DEER._make_hvp_batch_fn(
    ::DEER.ReactantHVP, grad_batch, backend::AutoReactant, X_template::AbstractMatrix
)
    _check_reactant_mode(backend)
    core = JVPWith(grad_batch)
    return _compiled(core, X_template, X_template)
end

function ParallelMCMC._reactant_resolve_value_grad_hvp_batch(
    logdensity_batch, backend::AutoReactant, X_template::AbstractMatrix
)
    _check_reactant_mode(backend)
    return ValueGradHVPBatch(logdensity_batch)
end

# One XLA executable per DEER iteration: `deer_update!` and the batched MALA step,
# traced. The loop stays on the host (`DEER._deer_loop`) (#86).

# Probes from the traced RNG seeded by the iteration's input.
function DEER._probe_matrix!(
    Z::Reactant.AnyTracedRArray{T,2}, ::Random.AbstractRNG, _, dev::Reactant.ReactantRNG
) where {T}
    Random.rand!(dev, Z)
    Z .= ifelse.(Z .< T(0.5), -one(T), one(T))
    return Z
end

struct DEERIteration{V,E,C,K}
    evaluator::V
    epsilon::E
    cholM::C
    T::Int
    probes::Int
    jacobian_damping::K
    jacobian_clip::K
    damping::K
end

function DEERIteration(evaluator::ValueGradHVPBatch, sampler)
    return DEERIteration(
        evaluator,
        sampler.epsilon,
        _host_cholesky(sampler.cholM),
        sampler.T,
        sampler.probes,
        sampler.jacobian_damping,
        sampler.jacobian_clip,
        sampler.damping,
    )
end

# Reactant captures arrays as host constants, so a mass matrix on the device moves to the host.
_host_cholesky(::Nothing) = nothing
function _host_cholesky(C::LinearAlgebra.Cholesky)
    return LinearAlgebra.Cholesky(Array(C.factors), C.uplo, C.info)
end

# The fields a DEERIteration compiles in, read from it or from a sampler.
function _settings(x)
    return (x.epsilon, x.T, x.probes, x.jacobian_damping, x.jacobian_clip, x.damping)
end

#= A rebuilt sampler may carry an equal `cholM` in a new object; compare it by value then.
Only the triangle `uplo` names is set: `cholesky` leaves the other one uninitialized. =#
function _same_cholM(compiled, cholM)
    (compiled === nothing || cholM === nothing) && return compiled === cholM
    return compiled.U == _host_cholesky(cholM).U
end

# Returns `S_new` and `[Δ_max, S_max]`, one download for the convergence test.
function (it::DEERIteration)(S, s0, Xi, U, seed)
    probe_rng = Reactant.ReactantRNG(seed)
    ws = DEER.DEERWorkspace(S, s0; probe_rng=probe_rng)
    step = ParallelMCMC._mala_batched_fwd_and_jvp(
        S, it.evaluator, it.epsilon, Xi, U, it.cholM
    )
    rec = DEER.TapedRecursion(nothing, nothing, Base.OneTo(it.T); fwd_and_jvp_batch=step)
    S_new = similar(S)
    DEER.deer_update!(
        ws,
        S_new,
        rec,
        s0,
        S;
        damping=it.damping,
        probes=it.probes,
        rng=probe_rng,
        jacobian_damping=it.jacobian_damping,
        jacobian_clip=it.jacobian_clip,
    )
    maxima = vcat(
        vec(maximum(abs, S_new .- S; dims=(1, 2))), vec(maximum(abs, S_new; dims=(1, 2)))
    )
    return S_new, maxima, probe_rng.seed
end

struct ReactantDEERSolver{I,M,C,S,X}
    iteration::I
    cholM::M
    compiled::C
    stages::S
    template::X
    D::Int
    platform::String
end

function ParallelMCMC._mala_deer_workspace(
    evaluator::ValueGradHVPBatch, model, sampler::ParallelMCMC.ParallelMALASampler, x0, old
)
    sampler.jacobian === :stoch_diag ||
        return ParallelMCMC._mala_deer_workspace(nothing, model, sampler, x0, old)
    FP = typeof(sampler.epsilon)
    template = similar(x0, FP, 0, 0)
    if old isa ReactantDEERSolver &&
        old.iteration.evaluator === evaluator &&
        isequal(_settings(old.iteration), _settings(sampler)) &&
        old.D == length(x0) &&
        typeof(old.template) == typeof(template) &&
        (old.cholM === sampler.cholM || _same_cholM(old.iteration.cholM, sampler.cholM))
        return old
    end
    iteration = DEERIteration(evaluator, sampler)

    D, T = length(x0), sampler.T
    S0 = similar(x0, FP, D, T)
    s0 = similar(x0, FP)
    templates = (S0, s0, similar(S0), similar(s0, T))
    _warn_reactant_host_roundtrip(x0)
    stages = map(_in_stage, templates)
    inputs = map(_upload, templates, stages)
    seed = Reactant.to_rarray(zeros(UInt64, 2))
    compiled = try
        _compile_full_precision(iteration, inputs..., seed)
    finally
        foreach(_free!, inputs)
        _free!(seed)
    end
    return ReactantDEERSolver(
        iteration, sampler.cholM, compiled, stages, template, D, _reactant_client_platform()
    )
end

function ParallelMCMC._solve_mala_block(
    ws::ReactantDEERSolver,
    rng::Random.AbstractRNG,
    model::ParallelMCMC.PreppedDensityModel,
    sampler::ParallelMCMC.ParallelMALASampler,
    x0::AbstractVector,
    tape,
    Xi,
    U;
    warn::Bool=true,
)
    # Later blocks start from the downloaded trajectory, whose eltype can differ from x0's.
    FP = eltype(Xi)
    S0 = similar(Xi)
    S0 .= x0
    s0 = similar(Xi, FP, length(x0))
    s0 .= x0
    s0_r, Xi_r, U_r = map(_upload, (s0, Xi, U), ws.stages[2:4])
    S = _upload(S0, ws.stages[1])
    S_last = Ref{Any}(S)  # newest trajectory, so the error path frees it too
    S_spare = nothing
    seed = Ref(Reactant.to_rarray(rand(rng, UInt64, 2)))
    try
        step = function (S_old, S_in)
            _free!(S_old)
            S_next, maxima, seed_next = ws.compiled(S_in, s0_r, Xi_r, U_r, seed[])
            S_last[] = S_next
            # The RNG updates its seed in place, so `seed_next` may wrap the old seed's buffer.
            Reactant.get_buffer(seed_next) === Reactant.get_buffer(seed[]) ||
                _free!(seed[])
            seed[] = seed_next
            Δ_max, S_max = Array(maxima)
            _free!(maxima)
            return S_next, Δ_max, S_max
        end
        S, S_spare, converged, _, metric = DEER._deer_loop(
            step, S, nothing, sampler.tol_abs, sampler.tol_rel, sampler.maxiter
        )
        warn && !converged && DEER._warn_not_converged(sampler.maxiter, metric)
        return _download(ws.template, S, ws.platform), converged
    finally
        foreach(_free!, (s0_r, Xi_r, U_r, seed[], S_last[], S_spare))
    end
end

end # module
