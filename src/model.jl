"""
    DensityModel(logdensity, grad_logdensity, dim;
                 param_names=nothing, hvp=nothing,
                 logdensity_batch=nothing,
                 grad_logdensity_batch=nothing, hvp_batch=nothing)

Describe a target density and the derivatives used by ParallelMCMC's samplers.

For a vector `x` of length `dim`, `logdensity(x)` must return a real scalar and
`grad_logdensity(x)` must return a vector like `x`. The log-density may omit its
normalizing constant. Instead of a gradient function, `grad_logdensity` may be
an `ADTypes.AbstractADType` backend:

```julia
model = DensityModel(logdensity, AutoForwardDiff(), dim)
```

Sequential MALA needs only the gradient. Parallel MALA also needs
Hessian-vector products. Supply `hvp(x, v)`, put an AD backend in `hvp`, or pass
a `backend` to [`ParallelMALASampler`](@ref). If none of these is given and
`grad_logdensity` is an AD backend, the HVP runs that backend's forward-mode
counterpart over the gradient. AD backends are prepared when sampling starts,
after the element type and storage of the initial state are known.

The optional batched functions operate on independent columns:

- `logdensity_batch(X)` returns one log-density per column of `X`.
- `grad_logdensity_batch(X)` returns a matrix shaped like `X`.
- `hvp_batch(X, V)` returns `H(X[:, j]) * V[:, j]` in column `j`.

Both batched derivative slots require `logdensity_batch`. A derivative slot may
hold a callable or an AD backend. When a batched gradient is derived with AD,
each output of `logdensity_batch` must depend only on the matching input column.

`param_names` controls the parameter keys in the returned `FlexiChain`. With
the default `nothing`, the chain contains one vector-valued parameter named
`:x`. See [Defining models](@ref) for naming, batching, Turing integration, and
the rules for combining derivative backends.
"""
struct DensityModel{F,G,H,FB,GB,HB,PN} <: AbstractMCMC.AbstractModel
    logdensity::F
    grad_logdensity::G
    hvp::H
    logdensity_batch::FB
    grad_logdensity_batch::GB
    hvp_batch::HB
    dim::Int
    param_names::PN
end

function DensityModel(
    logdensity,
    grad_logdensity,
    dim::Int;
    param_names=nothing,
    hvp=nothing,
    logdensity_batch=nothing,
    grad_logdensity_batch=nothing,
    hvp_batch=nothing,
)
    logdensity isa AbstractADType &&
        throw(ArgumentError("logdensity must be callable, not an AD backend"))
    logdensity_batch isa AbstractADType &&
        throw(ArgumentError("logdensity_batch must be callable, not an AD backend"))
    grad_logdensity === nothing && throw(
        ArgumentError(
            "grad_logdensity must be a callable or an ADTypes.AbstractADType backend"
        ),
    )
    _batch_needs_logp(name) = throw(ArgumentError("$name requires logdensity_batch"))
    if logdensity_batch === nothing
        grad_logdensity_batch === nothing || _batch_needs_logp("grad_logdensity_batch")
        hvp_batch === nothing || _batch_needs_logp("hvp_batch")
    end
    return DensityModel(
        logdensity,
        grad_logdensity,
        hvp,
        logdensity_batch,
        grad_logdensity_batch,
        hvp_batch,
        dim,
        param_names,
    )
end

"""
A [`DensityModel`](@ref) whose AD backends have been resolved to callables.
`source` prevents a cached preparation from being reused with another model.
"""
struct PreppedDensityModel{F,G,H,FB,GB,HB,PN,SM<:DensityModel}
    logdensity::F
    grad_logdensity::G
    hvp::H
    logdensity_batch::FB
    grad_logdensity_batch::GB
    hvp_batch::HB
    dim::Int
    param_names::PN
    source::SM
end

# Identity is required: reusing another model's preparation would sample the wrong target.
_prepped_for(prepped::PreppedDensityModel, model::DensityModel) = prepped.source === model

# DI keys preparations on function identity. Other input types use an unprepared call.
struct _ADGradient{F,B<:AbstractADType,P,TX}
    logdensity::F
    backend::B
    prep::P
end
function (g::_ADGradient{F,B,P,TX})(x) where {F,B,P,TX}
    if x isa TX
        return DI.gradient(g.logdensity, g.prep, g.backend, x)
    else
        return DI.gradient(g.logdensity, g.backend, x)
    end
end

# Columns of X are independent samples, so ∇_X sum(logdensity_batch(X))
# stacks the per-column gradients.
struct _BatchLogdensitySum{F}
    logdensity_batch::F
end
(c::_BatchLogdensitySum)(X) = sum(c.logdensity_batch(X))

struct _ADGradientBatch{C<:_BatchLogdensitySum,B<:AbstractADType,P,TX}
    closure::C
    backend::B
    prep::P
end
function (g::_ADGradientBatch{C,B,P,TX})(X) where {C,B,P,TX}
    if X isa TX
        return DI.gradient(g.closure, g.prep, g.backend, X)
    else
        return DI.gradient(g.closure, g.backend, X)
    end
end

function _resolve_gradient(logdensity, backend::AbstractADType, x_template::AbstractVector)
    prep = DI.prepare_gradient(logdensity, backend, x_template)
    return _ADGradient{typeof(logdensity),typeof(backend),typeof(prep),typeof(x_template)}(
        logdensity, backend, prep
    )
end

function _resolve_gradient_batch(
    logdensity_batch, backend::AbstractADType, X_template::AbstractMatrix
)
    closure = _BatchLogdensitySum(logdensity_batch)
    prep = DI.prepare_gradient(closure, backend, X_template)
    return _ADGradientBatch{typeof(closure),typeof(backend),typeof(prep),typeof(X_template)}(
        closure, backend, prep
    )
end

function _resolve_gradient(
    logdensity, backend::ADTypes.AutoReactant, x_template::AbstractVector
)
    return _reactant_resolve_gradient(logdensity, backend, x_template)
end
function _reactant_resolve_gradient(logdensity, backend, x_template)
    return error(_REACTANT_LOAD_HINT)
end

function _resolve_gradient_batch(
    logdensity_batch, backend::ADTypes.AutoReactant, X_template::AbstractMatrix
)
    return _reactant_resolve_gradient_batch(logdensity_batch, backend, X_template)
end
function _reactant_resolve_gradient_batch(logdensity_batch, backend, X_template)
    return error(_REACTANT_LOAD_HINT)
end

# An explicit SecondOrder differentiates logdensity and bypasses the gradient slot.
function _resolve_hvp(logdensity, grad, grad_backend, hvp_backend, x_template)
    if hvp_backend isa DI.SecondOrder
        return DEER._make_hvp_fn_second_order(logdensity, hvp_backend, x_template)
    elseif grad_backend !== nothing
        return DEER._make_hvp_fn_second_order(
            logdensity, _second_order(hvp_backend, grad_backend), x_template
        )
    else
        return DEER._make_hvp_fn(
            DEER._hvp_strategy(hvp_backend), grad, hvp_backend, x_template
        )
    end
end

function _resolve_hvp_batch(
    logdensity_batch, grad_batch, grad_batch_backend, hvp_backend, X_template
)
    if hvp_backend isa DI.SecondOrder
        return DEER._make_hvp_batch_fn_second_order(
            _BatchLogdensitySum(logdensity_batch), hvp_backend, X_template
        )
    elseif grad_batch_backend !== nothing
        return DEER._make_hvp_batch_fn_second_order(
            _BatchLogdensitySum(logdensity_batch),
            _second_order(hvp_backend, grad_batch_backend),
            X_template,
        )
    else
        return DEER._make_hvp_batch_fn(
            DEER._hvp_strategy(hvp_backend), grad_batch, hvp_backend, X_template
        )
    end
end

# With no HVP source, run forward mode over an AD-derived gradient.
# AutoReactant gradient maps to itself, so the Reactant pairing still holds.
_default_hvp_backend(::Nothing) = nothing
_default_hvp_backend(grad_backend::AbstractADType) = DI.forward_counterpart(grad_backend)

# Reactant implements the two passes itself; DI composes all other backends.
_second_order(hvp_backend, grad_backend) = DI.SecondOrder(hvp_backend, grad_backend)
function _second_order(hvp_backend::ADTypes.AutoReactant, ::ADTypes.AutoReactant)
    return hvp_backend
end

_check_reactant_pair(grad_backend, hvp_backend) = nothing
_check_reactant_pair(::ADTypes.AutoReactant, ::ADTypes.AutoReactant) = nothing
_check_reactant_pair(::Nothing, ::ADTypes.AutoReactant) = nothing

function _check_reactant_pair(grad_backend::ADTypes.AutoReactant, hvp_backend)
    return throw(
        ArgumentError(
            "an AutoReactant gradient requires an AutoReactant HVP backend; " *
            "got $(hvp_backend)",
        ),
    )
end

function _check_reactant_pair(grad_backend, hvp_backend::ADTypes.AutoReactant)
    return throw(
        ArgumentError(
            "an AutoReactant HVP requires an AutoReactant or callable gradient; " *
            "got backend $(grad_backend)",
        ),
    )
end

function _check_reactant_pair(::ADTypes.AutoReactant, hvp_backend::DI.SecondOrder)
    return throw(
        ArgumentError(
            "an AutoReactant gradient requires a bare AutoReactant HVP backend; " *
            "got $(hvp_backend)",
        ),
    )
end

function _check_reactant_pair(grad_backend, hvp_backend::DI.SecondOrder)
    _second_order_has_reactant(hvp_backend) || return nothing
    return throw(
        ArgumentError(
            "AutoReactant cannot be used inside DifferentiationInterface.SecondOrder; " *
            "use a bare AutoReactant() HVP backend",
        ),
    )
end

function _second_order_has_reactant(so::DI.SecondOrder)
    return DI.outer(so) isa ADTypes.AutoReactant || DI.inner(so) isa ADTypes.AutoReactant
end

_check_reactant_hvp_source(grad, hvp_backend) = nothing

"""
    _prepare_model(model, x_template)                    -> PreppedDensityModel
    _prepare_model(model, x_template, T::Int, backend)   -> PreppedDensityModel

Resolve AD backends into prepared callables at `x_template`. The two-argument
form prepares only the gradient. The four-argument form also prepares DEER's
HVP and batched slots at shape `(model.dim, T)`.
"""
function _prepare_model(model::DensityModel, x_template::AbstractVector)
    grad = if model.grad_logdensity isa AbstractADType
        _resolve_gradient(model.logdensity, model.grad_logdensity, x_template)
    else
        model.grad_logdensity
    end
    return PreppedDensityModel(
        model.logdensity,
        grad,
        nothing,
        model.logdensity_batch,
        nothing,
        nothing,
        model.dim,
        model.param_names,
        model,
    )
end

function _prepare_model(model::DensityModel, x_template::AbstractVector, T::Int, backend)
    grad_backend =
        model.grad_logdensity isa AbstractADType ? model.grad_logdensity : nothing

    # Validate the pair before an AutoReactant gradient triggers compilation.
    needs_hvp = model.hvp === nothing || model.hvp isa AbstractADType
    hvp_backend = if needs_hvp
        hb = if model.hvp !== nothing
            model.hvp
        elseif backend !== nothing
            backend
        else
            _default_hvp_backend(grad_backend)
        end
        hb === nothing && throw(
            ArgumentError(
                "ParallelMALASampler needs a Hessian-vector product: supply `hvp` " *
                "on the DensityModel (callable or AD backend), pass `backend=` " *
                "to ParallelMALASampler, or give `grad_logdensity` as an AD backend",
            ),
        )
        _check_reactant_pair(grad_backend, hb)
        _check_reactant_hvp_source(model.grad_logdensity, hb)
        hb
    else
        nothing
    end

    grad = if grad_backend !== nothing
        _resolve_gradient(model.logdensity, grad_backend, x_template)
    else
        model.grad_logdensity
    end

    hvp = if needs_hvp
        _resolve_hvp(model.logdensity, grad, grad_backend, hvp_backend, x_template)
    else
        model.hvp
    end

    grad_batch = model.grad_logdensity_batch
    if grad_batch === nothing && model.logdensity_batch !== nothing
        grad_batch = grad_backend
    end
    grad_batch_backend = grad_batch isa AbstractADType ? grad_batch : nothing

    hvp_batch = model.hvp_batch
    batch_active = model.logdensity_batch !== nothing && grad_batch !== nothing

    if batch_active
        # Prepare on x0 in every column; zeros need not be in the support.
        X_template = similar(x_template, length(x_template), T)
        X_template .= x_template

        needs_hvp_batch = hvp_batch === nothing || hvp_batch isa AbstractADType
        hvp_batch_backend = if needs_hvp_batch
            hbb = if hvp_batch !== nothing
                hvp_batch
            elseif model.hvp isa AbstractADType
                model.hvp
            elseif backend !== nothing
                backend
            else
                _default_hvp_backend(grad_batch_backend)
            end
            hbb === nothing && throw(
                ArgumentError("batched DEER requires `hvp_batch` or a sampler `backend`"),
            )
            _check_reactant_pair(grad_batch_backend, hbb)
            hbb
        else
            nothing
        end

        if grad_batch_backend !== nothing
            grad_batch = _resolve_gradient_batch(
                model.logdensity_batch, grad_batch_backend, X_template
            )
        end

        if needs_hvp_batch
            hvp_batch = _resolve_hvp_batch(
                model.logdensity_batch,
                grad_batch,
                grad_batch_backend,
                hvp_batch_backend,
                X_template,
            )
        end
    elseif hvp_batch !== nothing
        # Do not silently ignore an explicitly supplied HVP.
        throw(
            ArgumentError(
                "hvp_batch has no batched gradient; supply grad_logdensity_batch or an AD-backed grad_logdensity",
            ),
        )
    end

    return PreppedDensityModel(
        model.logdensity,
        grad,
        hvp,
        model.logdensity_batch,
        grad_batch,
        hvp_batch,
        model.dim,
        model.param_names,
        model,
    )
end

#= Wrappers for LogDensityProblems objects. They are defined here rather than in
LogDensityProblemsExt so `_check_reactant_hvp_source` can dispatch on them. The call
methods live in the extension. =#
struct LogDensityProblemPrimal{L}
    ld::L
end
struct LogDensityProblemGradient{L}
    ld::L
end

function _check_reactant_hvp_source(
    ::LogDensityProblemGradient, hvp_backend::ADTypes.AutoReactant
)
    return throw(
        ArgumentError(
            "AutoReactant cannot differentiate a LogDensityProblems-derived gradient; " *
            "supply a Reactant-traceable gradient callable instead",
        ),
    )
end
