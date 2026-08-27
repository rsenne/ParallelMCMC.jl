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
            "client targets \"cpu\". Every call will round-trip to the host and back " *
            "instead of running where the array lives. Select a GPU client with " *
            "`Reactant.set_default_backend(\"gpu\")` if one is available." maxlog = 1
    end
    return nothing
end

_host(x::Array) = x
_host(x::AbstractArray) = Array(x)

# Preserve promotions performed by the compiled function.
function _from_host(template::AbstractArray, out)
    out_h = Array(out)
    res = similar(template, eltype(out_h), size(out_h))
    copyto!(res, out_h)
    return res
end

function _compiled(core, t1::AbstractArray)
    _warn_reactant_host_roundtrip(t1)
    r1 = Reactant.to_rarray(_host(t1))
    compiled = @compile core(r1)
    return x -> _from_host(x, compiled(Reactant.to_rarray(_host(x))))
end

function _compiled(core, t1::AbstractArray, t2::AbstractArray)
    _warn_reactant_host_roundtrip(t1)
    r1 = Reactant.to_rarray(_host(t1))
    r2 = Reactant.to_rarray(_host(t2))
    compiled = @compile core(r1, r2)
    return function (x, v)
        out = compiled(Reactant.to_rarray(_host(x)), Reactant.to_rarray(_host(v)))
        return _from_host(x, out)
    end
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
