module WrenCoreExt

using ParallelMCMC
using ADTypes: ADTypes, AbstractADType, AutoReactant
using EnzymeCore: EnzymeCore
using WrenCore: WrenCore

#=
A `VecModel` built with `AutoReactant` stores the model arguments as Reactant
arrays, so its log-density only runs when traced. ParallelMCMC's AutoReactant
path traces these callables inside `@compile`, which hands them traced arrays,
whose elements are not `AbstractFloat`s. Every other call (e.g. the batched MALA
step and trajectory log-densities) passes concrete arrays, which go through
WrenCore's compiled `lp` / `lp_batch` on the host.
=#
const ReactantVecModel = WrenCore.VecModel{<:WrenCore.Model,<:AutoReactant}

_is_concrete(x::AbstractArray) = eltype(x) <: AbstractFloat
_host(x::Array) = x
_host(x::AbstractArray) = Array(x)

struct ReactantWrenLogDensity{V<:ReactantVecModel}
    vm::V
end
function (f::ReactantWrenLogDensity)(x::AbstractVector)
    _is_concrete(x) && return WrenCore.lp(f.vm, _host(x))
    return f.vm.lp_func(x)
end

struct ReactantWrenLogDensityBatch{V<:ReactantVecModel}
    vm::V
end
function (f::ReactantWrenLogDensityBatch)(X::AbstractMatrix)
    _is_concrete(X) || return WrenCore.batched_lp_function(f.vm)(X)
    lps = WrenCore.lp_batch(f.vm, _host(X))
    X isa Array && return lps
    out = similar(X, eltype(lps), length(lps))
    return copyto!(out, lps)
end

_logdensity(vm::WrenCore.VecModel) = vm.lp_func
_logdensity(vm::ReactantVecModel) = ReactantWrenLogDensity(vm)
_logdensity_batch(vm::WrenCore.VecModel) = WrenCore.batched_lp_function(vm)
_logdensity_batch(vm::ReactantVecModel) = ReactantWrenLogDensityBatch(vm)

# The log-density closes over the model, so Enzyme has to treat it as constant.
_grad_backend(backend) = backend
function _grad_backend(backend::ADTypes.AutoEnzyme{M,Nothing}) where {M}
    return ADTypes.AutoEnzyme(; mode=backend.mode, function_annotation=EnzymeCore.Const)
end
# ParallelMCMC only supports the default AutoReactant mode, which is reverse mode.
function _grad_backend(backend::AutoReactant)
    ADTypes.mode(backend.mode) isa ADTypes.ForwardMode && return backend
    return AutoReactant()
end
#= Enzyme forward-over-reverse needs runtime activity on a Wren model: the
log-density is a constant closure holding the model's data, which Enzyme's static
activity analysis rejects when nested. Only the HVP pays for this; an explicit
`SecondOrder` bypasses the gradient slot, which keeps the faster mode. =#
_default_hvp(backend) = nothing
function _default_hvp(::ADTypes.AutoEnzyme)
    rt = EnzymeCore.set_runtime_activity
    return ParallelMCMC.DI.SecondOrder(
        ADTypes.AutoEnzyme(;
            mode=rt(EnzymeCore.Forward), function_annotation=EnzymeCore.Const
        ),
        ADTypes.AutoEnzyme(;
            mode=rt(EnzymeCore.Reverse), function_annotation=EnzymeCore.Const
        ),
    )
end

function _grad_backend(::Nothing)
    return throw(
        ArgumentError(
            "this VecModel has no AD backend; pass `backend=` to choose one for " *
            "ParallelMCMC",
        ),
    )
end

"""
    DensityModel(vm::WrenCore.VecModel; backend=vm.adtype, param_names=nothing,
                 hvp=..., logdensity_batch=..., grad_logdensity_batch=nothing,
                 hvp_batch=...)

Wrap a Wren `VecModel` as a `DensityModel`. Sampling happens in the `VecModel`'s
(unconstrained) parameter space.

`backend` fills the `grad_logdensity` slot, so ParallelMCMC derives gradients
and HVPs from the model's log-density itself. It defaults to the `VecModel`'s
AD type. An `AutoEnzyme` backend without a `function_annotation` gets
`Enzyme.Const`, and an `AutoReactant` reverse mode becomes `AutoReactant()`.
`logdensity_batch` defaults to `WrenCore.batched_lp_function(vm)`, which
Reactant lowers to a single batched program. The remaining keywords are
forwarded to `DensityModel`.

For `backend=AutoReactant()`, build `vm` with `AutoReactant` as well, so the
model's data enter the compiled programs as inputs rather than constants.

With an `AutoEnzyme` backend, `hvp` and `hvp_batch` default to Enzyme
forward-over-reverse with runtime activity, which Enzyme needs to nest through a
Wren model. The gradient itself does not use runtime activity. Other backends
leave both as `nothing`, so ParallelMCMC builds HVPs from the sampler's
`backend`.

# Example
```julia
using ParallelMCMC, WrenCore, ForwardDiff
using WrenDists: WrenDists as WD

@wren function demo(y)
    μ ~ WD.Normal()
    @observe y ~ WD.Normal(μ, 0.5)
end

model = DensityModel(VecModel(demo(1.5), AutoForwardDiff()))
chain = sample(model, ParallelMALASampler(0.1; T=32, backend=AutoForwardDiff()), 500)
```
"""
function ParallelMCMC.DensityModel(
    vm::WrenCore.VecModel;
    backend=vm.adtype,
    param_names=nothing,
    hvp=_default_hvp(backend),
    logdensity_batch=_logdensity_batch(vm),
    grad_logdensity_batch=nothing,
    hvp_batch=_default_hvp(backend),
)
    return ParallelMCMC.DensityModel(
        _logdensity(vm),
        _grad_backend(backend),
        WrenCore.LogDensityProblems.dimension(vm);
        param_names=param_names,
        hvp=hvp,
        logdensity_batch=logdensity_batch,
        grad_logdensity_batch=grad_logdensity_batch,
        hvp_batch=hvp_batch,
    )
end

"""
    DensityModel(model::WrenCore.Model; backend, kwargs...)

Build `WrenCore.VecModel(model, backend)` and wrap it as above. With an
`AutoReactant` backend the `VecModel` is compiled with Reactant too.
"""
function ParallelMCMC.DensityModel(model::WrenCore.Model; backend, kwargs...)
    return ParallelMCMC.DensityModel(WrenCore.VecModel(model, backend); kwargs...)
end

end # module
