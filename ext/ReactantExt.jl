module ReactantExt

# Reactant treats captured arrays as compile-time constants. Traced model
# functions must not depend on later mutations to captured data.

using ParallelMCMC: ParallelMCMC
using ParallelMCMC.DEER: DEER
using ADTypes: ADTypes, AutoReactant, AutoEnzyme
using Reactant: Reactant, @compile
using Enzyme: Enzyme

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

function (c::ReactantCall)(args::AbstractArray...)
    return _download(c.template, c.compiled(map(_upload, args, c.stages)...), c.platform)
end

function _compiled(core, templates::AbstractArray...)
    _warn_reactant_host_roundtrip(templates[1])
    stages = map(_in_stage, templates)
    compiled = @compile core(map(_upload, templates, stages)...)
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

end # module
