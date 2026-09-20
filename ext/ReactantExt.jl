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
Inputs go through `to_rarray` every call. Reactant's `copyto!` into an existing
ConcreteRArray uploads to a new buffer and then runs a compiled device copy on
top, so keeping XLA input buffers around would cost more, not less. The host
staging array is safe to reuse immediately: `ArrayFromHostBuffer` reads it
inside a `GC.@preserve` that ends when the call returns.

`to_rarray` wants an `Array`; views get collected first, as in
`DEER._materialize_ad_array`.
=#
_upload(x::Array, ::Nothing) = Reactant.to_rarray(x)
_upload(x::AbstractArray, ::Nothing) = Reactant.to_rarray(Array(x))
function _upload(x::AbstractArray, stage::AbstractArray)
    copyto!(stage, x)
    return Reactant.to_rarray(stage)
end

# Host arrays upload directly and need no stage.
function _in_stage(template::AbstractArray)
    ParallelMCMC.needs_host_staging(template) || return nothing
    return ParallelMCMC._host_staging_buffer(template, eltype(template), size(template))
end

# The output eltype can differ from the template's, so run the compiled
# function once on the template to size the stage. The template is x0, a valid
# point. The platform is fixed per compiled function.
function _prepare_output(template::AbstractArray, out_thunk)
    ParallelMCMC.needs_host_staging(template) || return nothing, ""
    out = out_thunk()
    platform = try
        string(Reactant.XLA.platform_name(Reactant.XLA.client(out)))
    catch
        _reactant_client_platform()
    end
    out_stage = ParallelMCMC._host_staging_buffer(template, eltype(out), size(out))
    return out_stage, platform
end

# Keep promotions performed by the compiled function. Callers hold on to
# gradients, so the result is a fresh array every call.
function _promote_like(template::AbstractArray, out_h::Array)
    template isa Array && eltype(out_h) === eltype(template) && return out_h
    res = similar(template, eltype(out_h), size(out_h))
    copyto!(res, out_h)
    return res
end

# Pointer access needs an unsharded PJRT buffer that is not already on the
# host. `nothing` means fall back to host staging.
function _device_view(template::AbstractArray, out, platform::AbstractString)
    out isa Reactant.ConcretePJRTArray || return nothing
    Reactant.Sharding.is_sharded(out.sharding) && return nothing
    buf = Reactant.get_buffer(out)
    Reactant.XLA.buffer_on_cpu(buf) && return nothing
    ptr = Reactant.XLA.unsafe_buffer_pointer(buf)
    return ParallelMCMC._device_array_from_pointer(
        template, eltype(out), ptr, size(out), platform
    )
end

# No stage: `Array(out)` already allocates a fresh host array.
function _download(template::AbstractArray, out, ::Nothing, ::AbstractString)
    return _promote_like(template, Array(out))
end

function _download(
    template::AbstractArray, out, out_stage::AbstractArray, platform::AbstractString
)
    wait(out)
    view = _device_view(template, out, platform)
    if view !== nothing
        res = similar(template, eltype(out), size(out))
        GC.@preserve out copyto!(res, view)
        return res
    end
    # CPU client, IFRT, sharded, or a device family without pointer wrapping.
    copyto!(out_stage, out)
    res = similar(template, eltype(out), size(out))
    copyto!(res, out_stage)
    return res
end

# One of these per compiled function, owning its staging buffers. `template` is
# the first argument; HVP outputs follow `x`, not `v`.
struct ReactantUnary{F,S1,T,OS}
    compiled::F
    in1_stage::S1
    template::T
    out_stage::OS
    platform::String
end

function (c::ReactantUnary)(x::AbstractArray)
    xr = _upload(x, c.in1_stage)
    return _download(c.template, c.compiled(xr), c.out_stage, c.platform)
end

struct ReactantBinary{F,S1,S2,T,OS}
    compiled::F
    in1_stage::S1
    in2_stage::S2
    template::T
    out_stage::OS
    platform::String
end

function (c::ReactantBinary)(x::AbstractArray, v::AbstractArray)
    xr = _upload(x, c.in1_stage)
    vr = _upload(v, c.in2_stage)
    return _download(c.template, c.compiled(xr, vr), c.out_stage, c.platform)
end

function _compiled(core, t1::AbstractArray)
    _warn_reactant_host_roundtrip(t1)
    in1_stage = _in_stage(t1)
    compiled = @compile core(_upload(t1, in1_stage))
    out_stage, platform = _prepare_output(t1, () -> compiled(_upload(t1, in1_stage)))
    return ReactantUnary(compiled, in1_stage, t1, out_stage, platform)
end

function _compiled(core, t1::AbstractArray, t2::AbstractArray)
    _warn_reactant_host_roundtrip(t1)
    in1_stage = _in_stage(t1)
    in2_stage = _in_stage(t2)
    compiled = @compile core(_upload(t1, in1_stage), _upload(t2, in2_stage))
    out_stage, platform = _prepare_output(
        t1, () -> compiled(_upload(t1, in1_stage), _upload(t2, in2_stage))
    )
    return ReactantBinary(compiled, in1_stage, in2_stage, t1, out_stage, platform)
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
