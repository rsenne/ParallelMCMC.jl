module ReactantExt

#=
Reactant-compiled derivative paths, selected by `ADTypes.AutoReactant()` in any
derivative slot of `DensityModel`, or as the sampler `backend`. Derivatives are
traced with Enzyme-MLIR and compiled to XLA executables by `Reactant.@compile`,
which keeps them off Enzyme's LLVM pipeline and so off the GPU
`cuMemcpyDtoHAsync_v2` gc-transition abort, and off DifferentiationInterface
entirely. For a log-density-only model it yields a true second-order HVP.

Two silent failure modes, ahead of the ordinary limitations:

Captured data is frozen at compile time. `@compile` bakes any plain `Array` /
`Ref` reached through the traced closure into the executable as a constant. A
`logdensity` written as `x -> f(x, data)` whose `data` is mutated afterwards
keeps handing back the pre-mutation derivative from every executable compiled
before the mutation, with no error and no warning. Traced functions must be pure
with respect to what they capture; pass data that can change in as an argument.

The compiled program runs wherever Reactant's XLA client points, which need not
be a GPU. That client is a Reactant/`Reactant_jll`-wide setting
(`Reactant.set_default_backend`) and has nothing to do with where the package's
own `Vector`s / `CuArray`s live. Preparing on `CuArray` parameters while the
client targets `"cpu"` still compiles and still gives correct answers, but every
call round-trips `CuArray -> host -> XLA-CPU -> host -> CuArray`.
`_warn_reactant_host_roundtrip` below warns once per compiled slot on that
combination; it cannot fix it.

Requirements / limitations:
  - The traced function (`logdensity` / `gradlogp` / batched forms) must be
    Reactant-traceable: plain array ops. DynamicPPL-built log-densities do
    NOT trace as-is.
  - Executables are shape-specialized to the preparation templates, and
    arguments are marshalled to/from Reactant's own (XLA) device memory on
    every call. A boundary copy, not a fused in-place path — a target for
    later optimization. `@compile` also does not memoize across preparations:
    every `sample()` call recompiles every `AutoReactant` slot the model uses.
  - HVPs are explicit forward-over-(gradient) compositions; NEVER
    `Enzyme.hvp`, which silently returns zeros under `@compile`.
  - `AutoReactant.mode` (the wrapped `AutoEnzyme`) is not honoured: gradients
    always trace as Enzyme reverse, HVPs as forward-over-that. A non-default
    `mode` is rejected outright (`_check_reactant_mode`) rather than ignored.
=#

# Both `Reactant` and `Enzyme` trigger this extension (see Project.toml): the
# traced derivatives call `Enzyme.autodiff` / `Enzyme.gradient` themselves
# rather than reaching Enzyme-MLIR through Reactant.
using ParallelMCMC: ParallelMCMC
using ParallelMCMC.DEER: DEER
using ADTypes: ADTypes, AutoReactant, AutoEnzyme
using Reactant: Reactant, @compile
using Enzyme: Enzyme

#=
`AutoReactant()` defaults to `AutoReactant(; mode=AutoEnzyme())`, i.e.
`AutoReactant{AutoEnzyme{Nothing,Nothing}}` — the exact type matched below.
Anything else names an Enzyme mode or annotation, and this extension honours
neither.
=#
_check_reactant_mode(::AutoReactant{AutoEnzyme{Nothing,Nothing}}) = nothing
function _check_reactant_mode(backend::AutoReactant)
    return throw(
        ArgumentError(
            "AutoReactant(; mode=$(backend.mode)) is not supported: gradients always " *
            "trace as Enzyme reverse-mode and HVPs as forward-over-that, whatever " *
            "`mode` says. Use the default AutoReactant().",
        ),
    )
end

#=
Warn once per compiled slot when the template is not a plain `Array` (so looks
like it lives on a GPU) while Reactant's default XLA client targets "cpu": every
call then pays a host round trip on top of the usual marshalling. Guarded, so a
Reactant version without `XLA.platform_name` / `XLA.default_backend` degrades to
no warning instead of erroring out of `_prepare_model`.
=#
function _reactant_client_platform()
    return try
        string(Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
    catch
        nothing
    end
end

_warn_reactant_host_roundtrip(::Array) = nothing
function _warn_reactant_host_roundtrip(x::AbstractArray)
    if _reactant_client_platform() == "cpu"
        @warn "AutoReactant: preparing on a $(typeof(x)), but Reactant's default XLA " *
            "client targets \"cpu\". Every call will round-trip to the host and back " *
            "instead of running where the array lives, which is likely slower than not " *
            "using Reactant at all. Point Reactant at a GPU client with " *
            "`Reactant.set_default_backend(\"gpu\")` if one is available." maxlog = 1
    end
    return nothing
end

# Host-materialize for the Reactant boundary: Reactant manages its own (XLA)
# device memory, so we round-trip through a plain host Array regardless of
# where the package's arrays live (Vector / CuArray / SubArray).
_host(x::Array) = x
_host(x::AbstractArray) = Array(x)

#=
Eltype comes from `out`, what Reactant actually computed, not from `template`: a
traced computation that promotes internally would otherwise be narrowed back to
the template's eltype on the way out.
=#
function _from_host(template::AbstractArray, out)
    out_h = Array(out)
    res = similar(template, eltype(out_h), size(out_h))
    copyto!(res, out_h)
    return res
end

#=
Compile `core` for the template shapes once and return a closure that
marshals package arrays <-> Reactant arrays.
=#
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

# Forward-mode JVP of `g` in direction `v`, i.e. J(g)·v. For g = gradlogp
# this is the HVP H·v. `Const(g)` so Enzyme doesn't treat captures as active.
function _jvp(g, x, v)
    return only(Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(g), Enzyme.Duplicated(x, v)))
end

_rev_gradient(f, x) = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), x)[1]

#=
Gradient slots. The HVP factories below get `logdensity` from `_resolve_hvp`,
which already has it.
=#
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
    # `_BatchLogdensitySum` is the same sum-over-columns the DI-driven batched
    # gradient uses (src/interface.jl), so both batched paths differentiate the
    # same thing.
    core = Base.Fix1(_rev_gradient, ParallelMCMC._BatchLogdensitySum(logdensity_batch))
    return _compiled(core, X_template)
end

#=
HVP factories. `_resolve_hvp` / `_resolve_hvp_batch` (src/interface.jl) route to
one of two shapes, the same two every other backend gets.

  - Both `grad_logdensity` and the HVP source `AutoReactant`: the AD-derived
    gradient case, with `_second_order` collapsing the pair to one
    `AutoReactant()` since DI cannot form a `SecondOrder` from it.
    `_make_hvp_fn_second_order` here traces forward-over-reverse from
    `logdensity` as a single XLA program.
  - A hand-written `gradlogp` with an `AutoReactant` HVP source: routed by
    `_hvp_strategy(::AutoReactant) = ReactantHVP()` in `DEER.jl` to
    `_make_hvp_fn` below, a forward JVP over that callable.
=#
function DEER._make_hvp_fn_second_order(
    logdensity, backend::AutoReactant, x_template::AbstractVector
)
    _check_reactant_mode(backend)
    inner = Base.Fix1(_rev_gradient, logdensity)
    core(x, v) = _jvp(inner, x, v)
    return _compiled(core, x_template, x_template)
end

function DEER._make_hvp_fn(
    ::DEER.ReactantHVP, gradlogp, backend::AutoReactant, x_template::AbstractVector
)
    _check_reactant_mode(backend)
    core(x, v) = _jvp(gradlogp, x, v)
    return _compiled(core, x_template, x_template)
end

function DEER._make_hvp_batch_fn_second_order(
    logdensity_batch_sum, backend::AutoReactant, X_template::AbstractMatrix
)
    _check_reactant_mode(backend)
    inner = Base.Fix1(_rev_gradient, logdensity_batch_sum)
    core(X, V) = _jvp(inner, X, V)
    return _compiled(core, X_template, X_template)
end

function DEER._make_hvp_batch_fn(
    ::DEER.ReactantHVP, grad_batch, backend::AutoReactant, X_template::AbstractMatrix
)
    # Column-independent batched gradient ⇒ forward JVP is the columnwise HVP.
    _check_reactant_mode(backend)
    core(X, V) = _jvp(grad_batch, X, V)
    return _compiled(core, X_template, X_template)
end

end # module
