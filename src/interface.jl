#=
AbstractMCMC interface for ParallelMCMC samplers.

Defines model/sampler/state/transition types and implements
`AbstractMCMC.step` so that `sample(model, sampler, N)` works out of the box.
=#

"""
    DensityModel(logdensity, grad_logdensity, dim; param_names, logdensity_batch, grad_logdensity_batch, hvp, hvp_batch)

Wraps a log-density function, its gradient, and optional Hessian-vector
product helpers for use with ParallelMCMC samplers.

Each derivative slot (`grad_logdensity`, `hvp`, `grad_logdensity_batch`,
`hvp_batch`) takes a callable or an `ADTypes.AbstractADType`, so a model can be
built from the log-density alone:

    DensityModel(logp, AutoForwardDiff(), dim)

Backends become prepared DifferentiationInterface callables when sampling starts,
and an AD failure surfaces there. `ADTypes.AutoReactant()` is the one backend DI
cannot drive; it is traced with Enzyme-MLIR and compiled to an XLA executable by
Reactant.jl instead, which needs `using Reactant` and brings requirements of its
own — see the GPU guide, `docs/src/15-gpu.md`, and `ext/ReactantExt.jl`'s module
docstring.

A backend in `hvp` / `hvp_batch` over a hand-written gradient is a single AD pass
across your own code. Over an AD-derived one it is
`DifferentiationInterface.SecondOrder(hvp_backend, grad_backend)`, taken through
DI's second-order operator. Passing a `SecondOrder` yourself always means the
latter, and bypasses the gradient slot even when you wrote it by hand.
`AutoReactant` sits outside both: it cannot go inside a `SecondOrder`, and
`grad_logdensity` and `hvp`/`hvp_batch` must either both be `AutoReactant()` or
neither, since mixing it with a DifferentiationInterface backend across the two
passes of an HVP raises an `ArgumentError` at preparation time.

- `logdensity(x::AbstractVector) -> Real`
- `grad_logdensity` — callable `x -> AbstractVector`, or a backend to
  differentiate `logdensity` with.
- `hvp` — optional callable `(x, v) -> AbstractVector`, or a backend. If
  `nothing`, DEER builds the HVP from the sampler's `backend`.
- `logdensity_batch(X::AbstractMatrix) -> AbstractVector` — optional batched
  log-density over columns (callable only). Columns must be independent:
  element `t` of the result may depend on column `t` of `X` and nothing else.
  A batched gradient derived from this is one gradient of its sum, so coupling
  between columns would go unnoticed and give wrong derivatives.
- `grad_logdensity_batch` — optional callable `X -> AbstractMatrix`, or a
  backend to differentiate `logdensity_batch` with. Left out alongside a
  `logdensity_batch`, it is derived when `grad_logdensity` is a backend.
- `hvp_batch` — optional callable `(X, V) -> AbstractMatrix`, or a backend,
  resolved against `grad_logdensity_batch` the same way `hvp` is against
  `grad_logdensity`.
- `dim::Int` — dimensionality of the parameter space
- `param_names` — optional collection of parameter names used in `FlexiChains` output. If
  `nothing` (the default), uses a single vector-valued parameter `:x` with shape `(dim,)`.
  See the [`Parameter names`](@ref parameter-names) section of the docs for more
  information.

Both batched derivative slots require `logdensity_batch`, which the batched
update evaluates directly. `ParallelMALASampler` runs that update once it has a
`logdensity_batch` and a batched gradient — `grad_logdensity_batch`, or one
derived from `logdensity_batch` when `grad_logdensity` is a backend. A
`logdensity_batch` on its own is fine too: it scores whole trajectories at once
and leaves the batched update off.
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

"""
Primary constructor — accepts optional batched functions as keyword arguments.
"""
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
    logdensity isa AbstractADType && throw(
        ArgumentError(
            "logdensity must be a callable, not an AD backend; there is nothing to derive it from",
        ),
    )
    logdensity_batch isa AbstractADType && throw(
        ArgumentError(
            "logdensity_batch must be a callable, not an AD backend; there is nothing to derive it from",
        ),
    )
    grad_logdensity === nothing && throw(
        ArgumentError(
            "grad_logdensity must be a callable or an ADTypes.AbstractADType backend"
        ),
    )
    #= The batched DEER update evaluates `logdensity_batch` itself, so without one
    a batched derivative slot is unusable either way: a backend has nothing to
    differentiate and a callable is never reached. Rejected rather than ignored.
    A `logdensity_batch` on its own is fine; `_prepare_model` decides from the
    gradient slot whether the batched path can run. =#
    _batch_needs_logp(name) = throw(
        ArgumentError(
            "$name requires logdensity_batch, which the batched DEER update evaluates"
        ),
    )
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
A [`DensityModel`](@ref) with its backend slots resolved to prepared callables —
DifferentiationInterface ones, or a compiled Reactant/XLA executable for
`AutoReactant()`. `_prepare_model` builds these, and the sampler internals take
them rather than a `DensityModel`, so no slot here ever holds an
`AbstractADType`. Which slots are filled depends on the sampler: the sequential
samplers only need `grad_logdensity` and get `nothing` for the DEER-only slots,
DEER fills the rest.

`source` is the `DensityModel` this was prepared from. A sampler state carries
a prepped model so the preparation is reused across steps, and `initial_state`
can hand such a state to a `step` called on a different model; `source` is how
that `step` tells the two apart (see `_prepped_for`).
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

#=
Whether a prepped model carried in a state was built from the model a `step`
was handed. Identity, not equality: an equal-but-distinct `DensityModel` just
costs one re-preparation, whereas treating a different model as a match would
silently sample the wrong target.
=#
_prepped_for(prepped::PreppedDensityModel, model::DensityModel) = prepped.source === model

#=
Resolved gradient wrappers. Structs rather than anonymous closures since DI keys
preparations on function identity. `TX` is the input type the prep was made for;
anything else falls back to an unprepared `DI.gradient` rather than failing.
=#
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

#=
`AutoReactant` gradients bypass DI (which cannot drive Reactant yet) and go
through hook functions that `ReactantExt` fills in with strictly more specific
methods; the untyped fallbacks give a clear load-order error.
=#
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

#=
Resolve an HVP slot given as a backend. `grad_backend` is the backend that
produced `grad`, or nothing when the gradient slot held a callable. Dispatch is
on types alone, so the branch folds and the returned closure type stays
statically known.

A `SecondOrder` bypasses the gradient slot even when that slot is hand-written:
naming both passes asks for two derivatives of `logdensity`. The slot is still
the drift term the MALA step uses.

`AutoReactant` takes the same three branches as anything else. `_second_order`
below collapses an `AutoReactant` pair to a single `AutoReactant()` rather than a
`DI.SecondOrder`, and `_hvp_strategy(::AutoReactant)` sends the
hand-written-gradient case to `ReactantHVP`; `ReactantExt` supplies both matching
methods. The pairing is already checked by `_prepare_model` before either path
here pays for a gradient resolution or an HVP compile.
=#
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

# Batched counterpart, on `sum(logdensity_batch(X))` for the second-order paths.
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

#= The HVP backend composed with the backend that produced the gradient under it.
For a DI-driven pair that is literally `DI.SecondOrder(hvp_backend,
grad_backend)`, taken through `DI.hvp`. Two `AutoReactant`s are not a pair DI
could run at all, so they collapse to the one backend that traces
forward-over-reverse from `logdensity` itself (`ReactantExt`'s
`_make_hvp_fn_second_order`). A mixed pair is already out by the time this runs,
via `_check_reactant_pair`. =#
_second_order(hvp_backend, grad_backend) = DI.SecondOrder(hvp_backend, grad_backend)
_second_order(::ADTypes.AutoReactant, ::ADTypes.AutoReactant) = ADTypes.AutoReactant()

#=
Reactant does not pair with a DI backend across the two passes of an HVP: the
compiled gradient is an opaque XLA executable DI cannot differentiate, and a
DI-prepared gradient is not Reactant-traceable. Both slots take `AutoReactant`
or neither does; a hand-written gradient pairs with either.

Dispatch rather than a runtime `isa` chain, so the check folds away with the rest
of `_resolve_hvp`'s branching.
=#
_check_reactant_pair(grad_backend, hvp_backend) = nothing
_check_reactant_pair(::ADTypes.AutoReactant, ::ADTypes.AutoReactant) = nothing
_check_reactant_pair(::Nothing, ::ADTypes.AutoReactant) = nothing

function _check_reactant_pair(grad_backend::ADTypes.AutoReactant, hvp_backend)
    return throw(
        ArgumentError(
            "an AutoReactant gradient needs an AutoReactant Hessian-vector product: " *
            "got hvp backend $(hvp_backend). Reactant compiles the gradient to an XLA " *
            "executable, which DifferentiationInterface cannot differentiate. Set the " *
            "model's `hvp` (or the sampler's `backend`) to AutoReactant() as well.",
        ),
    )
end

function _check_reactant_pair(grad_backend, hvp_backend::ADTypes.AutoReactant)
    return throw(
        ArgumentError(
            "an AutoReactant Hessian-vector product needs an AutoReactant or " *
            "hand-written gradient: got gradient backend $(grad_backend). Reactant " *
            "traces the HVP from the log-density (or from your gradient) and cannot " *
            "trace a DifferentiationInterface-prepared gradient. Set " *
            "`grad_logdensity` to AutoReactant() or supply a callable.",
        ),
    )
end

#= Would otherwise be ambiguous between the two methods above, and wants its own
message anyway: "an AutoReactant Hessian-vector product" does not describe a
`SecondOrder`. =#
function _check_reactant_pair(::ADTypes.AutoReactant, hvp_backend::DI.SecondOrder)
    return throw(
        ArgumentError(
            "an AutoReactant gradient needs a bare AutoReactant Hessian-vector " *
            "product: got hvp backend $(hvp_backend). An AutoReactant gradient " *
            "compiles to an XLA executable, which DifferentiationInterface's " *
            "SecondOrder cannot drive. Set the model's `hvp` (or the sampler's " *
            "`backend`) to AutoReactant(), unwrapped.",
        ),
    )
end

#= `SecondOrder(AutoReactant(), AutoReactant())` is a natural thing to try, given
the pairing table in `10-getting-started.md`, and would otherwise land in
`DI.prepare_hvp` several frames deep with no Reactant support. Checked whatever
the gradient slot holds, since a `SecondOrder` bypasses it anyway (see
`_resolve_hvp`). =#
function _check_reactant_pair(grad_backend, hvp_backend::DI.SecondOrder)
    _second_order_has_reactant(hvp_backend) || return nothing
    return throw(
        ArgumentError(
            "AutoReactant cannot go inside a DifferentiationInterface SecondOrder: " *
            "got hvp backend $(hvp_backend). DifferentiationInterface has no Reactant " *
            "support at all. Set `hvp` (or the sampler's `backend`) to a bare " *
            "AutoReactant() instead.",
        ),
    )
end

function _second_order_has_reactant(so::DI.SecondOrder)
    return DI.outer(so) isa ADTypes.AutoReactant || DI.inner(so) isa ADTypes.AutoReactant
end

#= A `LogDensityProblemGradient` (defined below) is a callable, so it clears the
`grad_backend === nothing` test that otherwise means "hand-written gradient" —
but it dispatches into DynamicPPL/LogDensityProblems and is not
Reactant-traceable. `_check_reactant_pair` only sees the backend, which is
`nothing` for both, so this checks `grad`'s type instead. The specialization has
to wait for `LogDensityProblemGradient` to exist and sits further down. =#
_check_reactant_hvp_source(grad, hvp_backend) = nothing

"""
    _prepare_model(model, x_template)                    -> PreppedDensityModel
    _prepare_model(model, x_template, T::Int, backend)   -> PreppedDensityModel

Resolve the backend slots of `model` into prepared DI callables, preparing
at `x_template`. The two-argument form only does `grad_logdensity`, which is
all the sequential samplers use, and leaves the DEER-only slots `nothing`.

The four-argument form also does the HVP and batched slots, preparing those at
a `(dim, T)` template. A missing `hvp` comes from the sampler's `backend`, and
a missing `hvp_batch` from the model's own `hvp` backend if it has one and the
sampler's otherwise. A missing `grad_logdensity_batch` is derived only when
`grad_logdensity` is a backend, never from the sampler's `backend`, which would
let it decide whether the batched update runs.

The batched slots are filled only when `logdensity_batch` is present and a
batched gradient is reachable; otherwise the batched path stays off and the
unbatched update covers it.
"""
function _prepare_model(model::DensityModel, x_template::AbstractVector)
    grad = if model.grad_logdensity isa AbstractADType
        _resolve_gradient(model.logdensity, model.grad_logdensity, x_template)
    else
        model.grad_logdensity
    end
    #= The DEER-only slots are dropped rather than passed along: a sequential
    sampler never reads them, and an unresolved backend sitting in one would
    break the invariant that no slot here holds an `AbstractADType`. =#
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

    #= Settle the HVP backend and check its pairing with `grad_backend` before
    resolving the gradient. Otherwise a mismatched `AutoReactant` pair, or the
    LogDensityProblems-gradient case, surfaces only once `_resolve_gradient` has
    paid for an XLA compile: 18+ seconds to report a config error. Neither check
    needs the resolved gradient. `grad_backend` is known already, and
    `_check_reactant_hvp_source` only looks at `model.grad_logdensity`'s type,
    which is `grad` unchanged whenever `grad_backend` is `nothing`. =#
    needs_hvp = model.hvp === nothing || model.hvp isa AbstractADType
    hvp_backend = if needs_hvp
        hb = model.hvp === nothing ? backend : model.hvp
        hb === nothing && throw(
            ArgumentError(
                "ParallelMALASampler needs a Hessian-vector product: supply `hvp` " *
                "on the DensityModel (callable or AD backend), or pass `backend=` " *
                "to ParallelMALASampler",
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

    #= A batched log-density with no batched gradient gets one from the model's
    own gradient backend, never the sampler's: a hand-written gradient has not
    opted into AD, and deriving one anyway would let `backend=` decide which
    update path runs. Not deriving one leaves the batched path off rather than
    raising, since `_trajectory_logps` uses `logdensity_batch` either way. =#
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

        # Same reasoning as the unbatched case above: check before compiling.
        needs_hvp_batch = hvp_batch === nothing || hvp_batch isa AbstractADType
        hvp_batch_backend = if needs_hvp_batch
            # The model's own HVP backend if it has one, else the sampler's.
            hbb = if hvp_batch === nothing
                model.hvp isa AbstractADType ? model.hvp : backend
            else
                hvp_batch
            end
            hbb === nothing && throw(
                ArgumentError(
                    "the batched DEER path needs a batched Hessian-vector product: " *
                    "supply `hvp_batch` on the DensityModel (callable or AD backend), " *
                    "or pass `backend=` to ParallelMALASampler",
                ),
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
        #= Raise rather than drop it: `hvp_batch` was supplied explicitly, and
        the alternative is silently running the unbatched update. =#
        throw(
            ArgumentError(
                "hvp_batch has no batched gradient to go with it: supply " *
                "`grad_logdensity_batch` on the DensityModel (a callable, or a backend " *
                "to derive one from `logdensity_batch`). A backend in `grad_logdensity` " *
                "also derives one; `backend=` on ParallelMALASampler does not.",
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

# Callable structs that allow us to dispatch on the type of the LogDensityProblems object in
# the postprocessing stage. Ideally these would be defined in the LogDensityProblemsExt.
# However, structs defined in extensions are hard to get hold of so we define them here.
# The callable behaviour itself is implemented in LogDensityProblemsExt.
struct LogDensityProblemPrimal{L}
    ld::L
end
struct LogDensityProblemGradient{L}
    ld::L
end

# The `_check_reactant_hvp_source` specialization promised further up.
function _check_reactant_hvp_source(
    ::LogDensityProblemGradient, hvp_backend::ADTypes.AutoReactant
)
    return throw(
        ArgumentError(
            "an AutoReactant Hessian-vector product needs an AutoReactant or " *
            "hand-written gradient: got a LogDensityProblems-derived gradient. Reactant " *
            "cannot trace DynamicPPL/LogDensityProblems machinery. Set `grad_logdensity` " *
            "to AutoReactant(), or supply a Reactant-traceable callable.",
        ),
    )
end

"""
    MALASampler(epsilon; cholM=nothing)

Metropolis-Adjusted Langevin Algorithm sampler with step size `epsilon`.

Optionally pass `cholM = cholesky(M)` to use a mass matrix `M` as a
preconditioner.  The proposal becomes `y = x + ε M ∇logp(x) + √(2ε) L ξ`
where `L` is the Cholesky factor of `M`.
"""
struct MALASampler{FP<:AbstractFloat,CM} <: AbstractMCMC.AbstractSampler
    epsilon::FP
    cholM::CM
end

function MALASampler(epsilon::Real; cholM=nothing)
    epsilon > 0 || throw(ArgumentError("epsilon must be > 0, got $epsilon"))
    eps_f = float(epsilon)
    return MALASampler{typeof(eps_f),typeof(cholM)}(eps_f, cholM)
end

"""
State for a `MALASampler` chain. Holds the prepped model so AD preparation
happens once per chain.
"""
struct MALAState{V<:AbstractVector,L<:Real,W,NV<:AbstractVector,H,DM<:PreppedDensityModel}
    x::V
    logp::L
    workspace::W
    noise::NV
    noise_host::H
    model::DM
end

"""
One `MALASampler` sample: parameter vector `x`, its log-density `logp`, and an
accept/reject flag.
"""
struct MALATransition{V<:AbstractVector,L<:Real}
    x::V
    logp::L
    accepted::Bool
end

function _make_noise_buffer(x::AbstractVector, ::Type{FP}, D::Int) where {FP}
    ξ = similar(x, FP, D)
    host = ξ isa CUDA.CuArray ? Vector{FP}(undef, D) : nothing
    return ξ, host
end

function _randn_like!(
    rng::Random.AbstractRNG, ξ::AbstractVector{FP}, host::Union{Nothing,AbstractVector{FP}}
) where {FP}
    if ξ isa CUDA.CuArray
        host === nothing && error("CuArray normal noise requires a reusable host buffer")
        randn!(rng, host)
        copyto!(ξ, host)
    else
        randn!(rng, ξ)
    end
    return ξ
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::MALASampler{FP};
    initial_params=nothing,
    kwargs...,
) where {FP}
    x = if initial_params !== nothing
        copy(initial_params)
    else
        randn(rng, FP, model.dim)
    end
    model = _prepare_model(model, x)
    logp_val = model.logdensity(x)
    ws = MALA.MALAWorkspace(x)
    noise, noise_host = _make_noise_buffer(x, FP, model.dim)
    t = MALATransition(x, logp_val, true)
    s = MALAState(x, logp_val, ws, noise, noise_host, model)
    return t, s
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    user_model::DensityModel,
    sampler::MALASampler,
    state::MALAState;
    kwargs...,
)
    #= Reuse the state's preparation, unless the state came from a different
    model (via `initial_state`), in which case the model we were handed is the
    one to sample. =#
    model = if _prepped_for(state.model, user_model)
        state.model
    else
        _prepare_model(user_model, state.x)
    end
    x = state.x
    ϵ = sampler.epsilon
    D = model.dim

    ξ = _randn_like!(rng, state.noise, state.noise_host)
    u = rand(rng)

    x_next = similar(x)
    x_next, accepted, _ = MALA.mala_step_with_logα!(
        x_next,
        state.workspace,
        model.logdensity,
        model.grad_logdensity,
        x,
        ϵ,
        ξ,
        u;
        cholM=sampler.cholM,
    )

    logp_val = accepted ? model.logdensity(x_next) : state.logp
    t = MALATransition(x_next, logp_val, accepted)
    s = MALAState(x_next, logp_val, state.workspace, state.noise, state.noise_host, model)
    return t, s
end

for TKey in (Symbol, VarName)
    @eval function AbstractMCMC.bundle_samples(
        samples::Vector{<:MALATransition},
        model::DensityModel,
        sampler::MALASampler,
        state::MALAState,
        ::Type{FlexiChains.FlexiChain{$TKey}};
        param_names=nothing,
        kwargs...,
    )
        N = length(samples)
        D = model.dim
        # Follow the sampler's working precision instead of pinning `Float64`, so a
        # Float32 (e.g. GPU) run yields a Float32 chain rather than a silently widened one.
        FP = typeof(sampler.epsilon)

        vals = Matrix{FP}(undef, N, D)
        logp = Vector{FP}(undef, N)
        accepted = Vector{Bool}(undef, N)

        for i in 1:N
            s = samples[i]
            vals[i, :] .= s.x
            logp[i] = s.logp
            accepted[i] = s.accepted
        end

        internals = (logp=logp, accepted=accepted)
        return _construct_flexichain($TKey, vals, internals, param_names, model)
    end
end

"""
    MALATapeElement(ξ, u)

One element of the MALA noise tape: a noise vector `ξ ~ N(0,I)` and a uniform
scalar `u ~ Uniform(0,1)`.  Stored with a concrete vector type `V` for type
stability inside `DEER.TapedRecursion`.
"""
struct MALATapeElement{FP<:AbstractFloat,V<:AbstractVector{FP}}
    ξ::V
    u::FP
end

"""
    ParallelMALASampler(epsilon; T, maxiter, tol_abs, tol_rel, jacobian, damping, probes, cholM, backend)

DEER-parallelized MALA sampler.

Supported Jacobian modes are `:stoch_diag` (the default Hutchinson diagonal
estimator) and `:diag` (exact diagonal via `D` JVPs).

`backend` supplies Hessian-vector products when the `DensityModel` brings no
`hvp` / `hvp_batch` of its own, and does nothing else. It never supplies a
gradient, so it cannot decide which update path runs, nor put AD on a function
the model had no backend for. Leave it out for a model that carries its own HVPs.

`backend = ADTypes.AutoReactant()` constrains the model too, since
`AutoReactant` does not pair with a DifferentiationInterface backend across the
two passes of an HVP: `grad_logdensity` must then be `AutoReactant()` as well, or
a hand-written callable. Mixing raises an `ArgumentError` at preparation time.
"""
struct ParallelMALASampler{FP<:AbstractFloat,CM,AD} <: AbstractMCMC.AbstractSampler
    epsilon::FP
    T::Int
    maxiter::Int
    tol_abs::FP
    tol_rel::FP
    jacobian::Symbol
    damping::FP
    probes::Int
    cholM::CM
    backend::AD
end

function ParallelMALASampler(
    epsilon::Real;
    T::Int=64,
    maxiter::Int=200,
    tol_abs::Real=1e-6,
    tol_rel::Real=1e-5,
    jacobian::Symbol=:stoch_diag,
    damping::Real=0.5,
    probes::Int=1,
    cholM=nothing,
    backend=nothing,
)
    epsilon > 0 || throw(ArgumentError("epsilon must be > 0, got $epsilon"))
    (jacobian === :stoch_diag || jacobian === :diag) ||
        throw(ArgumentError("jacobian must be :stoch_diag or :diag"))
    eps_f = float(epsilon)
    FP = typeof(eps_f)
    return ParallelMALASampler{FP,typeof(cholM),typeof(backend)}(
        eps_f,
        T,
        maxiter,
        FP(tol_abs),
        FP(tol_rel),
        jacobian,
        FP(damping),
        probes,
        cholM,
        backend,
    )
end

"""
State for a `ParallelMALASampler` chain. Holds the prepped model so AD
preparation happens once per chain.
"""
struct ParallelMALAState{
    V<:AbstractVector,L<:Real,M<:AbstractMatrix,LV<:AbstractVector,W,DM<:PreppedDensityModel
}
    x::V
    logp::L
    trajectory::M
    logps::LV
    workspace::W
    tape::Vector{<:MALATapeElement}
    t::Int
    model::DM
end

"""
One `ParallelMALASampler` sample: parameter vector `x` and its log-density `logp`.
"""
struct ParallelMALATransition{V<:AbstractVector,L<:Real}
    x::V
    logp::L
end

struct ParallelMALABlockSamples{B<:AbstractVector,L<:AbstractVector} <:
       AbstractVector{ParallelMALATransition}
    blocks::B
    logps::L
    n::Int
    T::Int
end

Base.size(samples::ParallelMALABlockSamples) = (samples.n,)
Base.length(samples::ParallelMALABlockSamples) = samples.n
Base.IndexStyle(::Type{<:ParallelMALABlockSamples}) = IndexLinear()

function Base.getindex(samples::ParallelMALABlockSamples, i::Int)
    1 <= i <= samples.n || throw(BoundsError(samples, i))
    block_idx = fld(i - 1, samples.T) + 1
    t = mod(i - 1, samples.T) + 1
    return ParallelMALATransition(
        samples.blocks[block_idx][:, t], samples.logps[block_idx][t]
    )
end

function _make_mala_tape_block(
    rng::Random.AbstractRNG, x0::AbstractVector, ::Type{FP}, D::Int, T::Int
) where {FP}
    Xi = similar(x0, FP, D, T)
    U_host = Vector{FP}(undef, T)

    if Xi isa CUDA.CuArray
        Xi_host = Matrix{FP}(undef, D, T)
        for t in 1:T
            randn!(rng, view(Xi_host, :, t))
            U_host[t] = FP(rand(rng))
        end
        copyto!(Xi, Xi_host)
    else
        for t in 1:T
            randn!(rng, view(Xi, :, t))
            U_host[t] = FP(rand(rng))
        end
    end

    U = similar(x0, FP, T)
    copyto!(U, U_host)
    tape = [MALATapeElement(view(Xi, :, t), U_host[t]) for t in 1:T]
    return tape, Xi, U
end

function _build_mala_deer_rec(
    model::PreppedDensityModel,
    ε::Real,
    tape::Vector{<:MALATapeElement},
    x0_like::AbstractVector;
    cholM=nothing,
    tape_noise=nothing,
    tape_uniforms=nothing,
)
    logp = model.logdensity
    gradlogp = model.grad_logdensity

    # `hvp` / `hvp_batch` were resolved to callables in `_prepare_model`.
    hvp_fn = model.hvp

    # Exact forward step.
    step_fwd =
        (x, te) -> MALA.mala_step_taped(logp, gradlogp, x, ε, te.ξ, te.u; cholM=cholM)

    # Explicit analytical JVP of the surrogate step.
    jvp =
        (x, te, v) -> MALA.mala_step_surrogate_sigmoid_jvp(
            logp, gradlogp, x, ε, te.ξ, te.u, v, hvp_fn; cholM=cholM
        )

    # Fused forward step + JVP.
    fwd_and_jvp =
        (x, te, v) -> MALA.mala_step_taped_and_jvp(
            logp, gradlogp, x, ε, te.ξ, te.u, v, hvp_fn; cholM=cholM
        )

    fwd_and_jvp_batch =
        if model.logdensity_batch !== nothing && model.grad_logdensity_batch !== nothing
            D = length(x0_like)
            T = length(tape)

            (tape_noise === nothing) == (tape_uniforms === nothing) || throw(
                ArgumentError("tape_noise and tape_uniforms must be provided together")
            )
            Xi, U = if tape_noise === nothing
                Xi_local = similar(x0_like, D, T)
                U_host = Vector{typeof(tape[1].u)}(undef, T)
                U_local = similar(x0_like, typeof(tape[1].u), T)

                for t in 1:T
                    copyto!(Xi_local, (t - 1) * D + 1, tape[t].ξ, 1, D)
                    U_host[t] = tape[t].u
                end
                copyto!(U_local, U_host)
                Xi_local, U_local
            else
                size(tape_noise) == (D, T) ||
                    throw(DimensionMismatch("tape_noise must have size (D, T)"))
                length(tape_uniforms) == T ||
                    throw(DimensionMismatch("tape_uniforms must have length T"))
                tape_noise, tape_uniforms
            end

            X_template = similar(x0_like, D, T)
            fill!(X_template, zero(eltype(X_template)))
            batch_ws = MALA.MALABatchedWorkspace(X_template)
            FT = similar(X_template)
            Jt = similar(X_template)
            hvp_batch = model.hvp_batch

            (Xbar, Z) -> MALA.mala_step_batched_fwd_and_jvp!(
                FT,
                Jt,
                batch_ws,
                model.logdensity_batch,
                model.grad_logdensity_batch,
                hvp_batch,
                Xbar,
                ε,
                Xi,
                U,
                Z;
                cholM=cholM,
            )
        else
            nothing
        end

    return DEER.TapedRecursion(
        step_fwd, jvp, tape; fwd_and_jvp=fwd_and_jvp, fwd_and_jvp_batch=fwd_and_jvp_batch
    )
end

function _deer_solve_new_tape(
    rng::Random.AbstractRNG,
    model::PreppedDensityModel,
    sampler::ParallelMALASampler,
    x0::AbstractVector;
    workspace=nothing,
)
    D = model.dim
    T = sampler.T
    FP = typeof(sampler.epsilon)

    tape, Xi, U = _make_mala_tape_block(rng, x0, FP, D, T)

    rec = _build_mala_deer_rec(
        model,
        sampler.epsilon,
        tape,
        x0;
        cholM=sampler.cholM,
        tape_noise=Xi,
        tape_uniforms=U,
    )

    ws = workspace === nothing ? DEER.DEERWorkspace(x0, T) : workspace

    S = DEER.solve(
        rec,
        x0;
        tol_abs=sampler.tol_abs,
        tol_rel=sampler.tol_rel,
        maxiter=sampler.maxiter,
        jacobian=sampler.jacobian,
        damping=sampler.damping,
        probes=sampler.probes,
        rng=rng,
        workspace=ws,
        copy_result=false,
    )
    return S, tape, ws
end

function _trajectory_logps(model::PreppedDensityModel, S::AbstractMatrix)
    if model.logdensity_batch !== nothing
        return Array(model.logdensity_batch(S))
    end

    T = size(S, 2)
    return [model.logdensity(S[:, t]) for t in 1:T]
end

function _parallel_mala_initial_x(
    rng::Random.AbstractRNG, model::DensityModel, ::ParallelMALASampler{FP}, initial_params
) where {FP}
    return initial_params !== nothing ? copy(initial_params) : randn(rng, FP, model.dim)
end

function _parallel_mala_progress(progress, progressname)
    progress === true && return AbstractMCMC.CreateNewProgressBar(progressname)
    progress === false && return AbstractMCMC.NoLogging()
    return progress
end

function _parallel_mala_update_progress!(
    progress, nsteps::Int, Ntotal::Int, next_update::Real, threshold::Real
)
    if nsteps >= next_update && next_update <= Ntotal
        AbstractMCMC.update_progress!(progress, min(nsteps / Ntotal, 1))
        next_update += threshold
    end
    return next_update
end

function _copy_trajectory_rows!(
    vals::AbstractMatrix{<:Real}, first_row::Int, S::AbstractMatrix, ncols::Int
)
    rows = first_row:(first_row + ncols - 1)
    S_host = Array(view(S, :, 1:ncols))
    vals[rows, :] .= transpose(S_host)
    return vals
end

function _sample_parallel_mala_chain(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::ParallelMALASampler,
    N::Int,
    ::Type{FlexiChains.FlexiChain{TKey}};
    initial_params=nothing,
    param_names=nothing,
    progress=AbstractMCMC.PROGRESS[],
    progressname="Sampling",
) where {TKey}
    D = model.dim
    FP = typeof(sampler.epsilon)

    vals = Matrix{FP}(undef, N, D)
    logp = Vector{FP}(undef, N)

    progress = _parallel_mala_progress(progress, progressname)
    x0 = _parallel_mala_initial_x(rng, model, sampler, initial_params)
    # Postprocessing dispatches on the user's model type (e.g. from the
    # DynamicPPL extension), so `model` itself stays unprepped.
    prepped = _prepare_model(model, x0, sampler.T, sampler.backend)
    ws = nothing
    nsteps = 0
    next_update = N / AbstractMCMC.get_n_updates(progress)
    threshold = next_update

    AbstractMCMC.@maybewithricherlogger begin
        AbstractMCMC.init_progress!(progress)
        try
            while nsteps < N
                S, tape, ws = _deer_solve_new_tape(rng, prepped, sampler, x0; workspace=ws)
                logps = _trajectory_logps(prepped, S)
                nkeep = min(sampler.T, N - nsteps)
                first_row = nsteps + 1
                rows = first_row:(first_row + nkeep - 1)

                _copy_trajectory_rows!(vals, first_row, S, nkeep)
                logp[rows] .= view(logps, 1:nkeep)

                x0 = copy(view(S, :, nkeep))
                nsteps += nkeep
                next_update = _parallel_mala_update_progress!(
                    progress, nsteps, N, next_update, threshold
                )
            end
        finally
            AbstractMCMC.finish_progress!(progress)
        end
    end

    internals = (logp=logp,)
    return _construct_flexichain(TKey, vals, internals, param_names, model)
end

function _construct_flexichain(
    ::Type{TKey},
    vals::AbstractMatrix{<:Real},
    internals::NamedTuple,
    param_names::Any,
    model::DensityModel,
) where {TKey}
    #= Wrap user-supplied names in `Parameter`. This allows people to specify, e.g.,
    `param_names=(:x, :y, :z=>(2,))` without faffing with `Parameter` themselves. Also
    'upgrade' symbol parameter names to VarNames if the user requested a VNChain. =#
    to_parameter(vn::VarName) = FlexiChains.Parameter(vn)
    to_parameter(s::Symbol) = FlexiChains.Parameter(TKey <: VarName ? VarName{s}() : s)

    N, D = size(vals)

    if param_names === nothing
        param_names = model.param_names
    end
    wrapped_param_names = if param_names === nothing
        # Wasn't defined either as `param_names` or `model.param_names`, so make some up
        (to_parameter(:x) => (D,),)
    else
        map(param_names) do n
            if n isa Pair
                to_parameter(n.first) => n.second
            elseif n isa TKey || n isa Symbol
                to_parameter(n)
            else
                throw(
                    ArgumentError(
                        "param_names must be a collection of Pairs, Symbols, or $TKey, got $(typeof(n))",
                    ),
                )
            end
        end
    end

    arr = reshape(vals, N, 1, D)
    param_chain = FlexiChains.FlexiChain{TKey}(arr, Tuple(wrapped_param_names))

    isempty(internals) && return param_chain

    #=
    The internals have mixed element types (e.g. `Bool` for `accepted`/`is_warmup`,
    `FP` for `logp`), so they cannot share a single stacked array without widening.
    Store each in its own column via the dict-of-arrays constructor, which preserves
    the natural element type (see issue #49), then merge with the parameters.
    =#
    extras = OrderedDict{FlexiChains.ParameterOrExtra{<:TKey},AbstractArray}(
        FlexiChains.Extra(name) => internals[name] for name in keys(internals)
    )
    extra_chain = FlexiChains.FlexiChain{TKey}(N, 1, extras)

    return merge(param_chain, extra_chain)
end

function _sample_parallel_mala_blocks(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::ParallelMALASampler,
    N::Int;
    initial_params=nothing,
    progress=AbstractMCMC.PROGRESS[],
    progressname="Sampling",
)
    blocks = Vector{AbstractMatrix}(undef, 0)
    logp_blocks = Vector{AbstractVector}(undef, 0)
    sizehint!(blocks, cld(N, sampler.T))
    sizehint!(logp_blocks, cld(N, sampler.T))

    progress = _parallel_mala_progress(progress, progressname)
    x0 = _parallel_mala_initial_x(rng, model, sampler, initial_params)
    prepped = _prepare_model(model, x0, sampler.T, sampler.backend)
    ws = nothing
    nsteps = 0
    next_update = N / AbstractMCMC.get_n_updates(progress)
    threshold = next_update
    final_state = nothing

    AbstractMCMC.@maybewithricherlogger begin
        AbstractMCMC.init_progress!(progress)
        try
            while nsteps < N
                S, tape, ws = _deer_solve_new_tape(rng, prepped, sampler, x0; workspace=ws)
                logps = _trajectory_logps(prepped, S)
                nkeep = min(sampler.T, N - nsteps)
                S_keep = copy(view(S, :, 1:nkeep))
                logps_keep = collect(view(logps, 1:nkeep))
                x0 = copy(view(S_keep, :, nkeep))

                push!(blocks, S_keep)
                push!(logp_blocks, logps_keep)
                final_state = ParallelMALAState(
                    x0, logps_keep[nkeep], S_keep, logps_keep, ws, tape, nkeep, prepped
                )

                nsteps += nkeep
                next_update = _parallel_mala_update_progress!(
                    progress, nsteps, N, next_update, threshold
                )
            end
        finally
            AbstractMCMC.finish_progress!(progress)
        end
    end

    return ParallelMALABlockSamples(blocks, logp_blocks, N, sampler.T), final_state
end

function _default_parallel_mala_mcmcsample(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::ParallelMALASampler,
    N::Integer;
    kwargs...,
)
    return invoke(
        AbstractMCMC.mcmcsample,
        Tuple{
            Random.AbstractRNG,
            AbstractMCMC.AbstractModel,
            AbstractMCMC.AbstractSampler,
            Integer,
        },
        rng,
        model,
        sampler,
        N;
        kwargs...,
    )
end

function AbstractMCMC.mcmcsample(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::ParallelMALASampler,
    N::Integer;
    progress=AbstractMCMC.PROGRESS[],
    progressname="Sampling",
    callback=nothing,
    num_warmup::Int=0,
    discard_initial::Int=num_warmup,
    thinning=1,
    chain_type::Type=Any,
    initial_state=nothing,
    initial_params=nothing,
    param_names=nothing,
    kwargs...,
)
    if callback !== nothing ||
        num_warmup != 0 ||
        discard_initial != 0 ||
        thinning != 1 ||
        initial_state !== nothing
        return _default_parallel_mala_mcmcsample(
            rng,
            model,
            sampler,
            N;
            progress=progress,
            progressname=progressname,
            callback=callback,
            num_warmup=num_warmup,
            discard_initial=discard_initial,
            thinning=thinning,
            chain_type=chain_type,
            initial_state=initial_state,
            initial_params=initial_params,
            param_names=param_names,
            kwargs...,
        )
    end

    N > 0 || error("the number of samples must be ≥ 1")
    N_int = Int(N)

    if chain_type <: FlexiChains.FlexiChain
        return _sample_parallel_mala_chain(
            rng,
            model,
            sampler,
            N_int,
            chain_type;
            initial_params=initial_params,
            param_names=param_names,
            progress=progress,
            progressname=progressname,
        )
    end

    samples, state = _sample_parallel_mala_blocks(
        rng,
        model,
        sampler,
        N_int;
        initial_params=initial_params,
        progress=progress,
        progressname=progressname,
    )
    chain_type === Any && return samples

    sample_vec = [samples[i] for i in 1:length(samples)]
    return AbstractMCMC.bundle_samples(
        sample_vec, model, sampler, state, chain_type; param_names=param_names, kwargs...
    )
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::ParallelMALASampler{FP};
    initial_params=nothing,
    kwargs...,
) where {FP}
    x0 = if initial_params !== nothing
        copy(initial_params)
    else
        randn(rng, FP, model.dim)
    end
    model = _prepare_model(model, x0, sampler.T, sampler.backend)

    S, tape, ws = _deer_solve_new_tape(rng, model, sampler, x0)
    logps = _trajectory_logps(model, S)
    x1 = S[:, 1]
    logp1 = logps[1]
    trans = ParallelMALATransition(x1, logp1)
    state = ParallelMALAState(x1, logp1, S, logps, ws, tape, 1, model)
    return trans, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    user_model::DensityModel,
    sampler::ParallelMALASampler,
    state::ParallelMALAState;
    kwargs...,
)
    #= A state from a different model (via `initial_state`) brings a trajectory
    solved under that model, so the rest of it can't be replayed either: pick up
    from the current position with a fresh tape under the model we were handed. =#
    reuse = _prepped_for(state.model, user_model)
    model = if reuse
        state.model
    else
        _prepare_model(user_model, state.x, sampler.T, sampler.backend)
    end
    T = sampler.T
    t_next = state.t + 1

    if reuse && t_next <= T
        x_new = state.trajectory[:, t_next]
        logp_new = state.logps[t_next]
        trans = ParallelMALATransition(x_new, logp_new)
        new_state = ParallelMALAState(
            x_new,
            logp_new,
            state.trajectory,
            state.logps,
            state.workspace,
            state.tape,
            t_next,
            model,
        )
        return trans, new_state
    else
        x0 = reuse ? state.trajectory[:, T] : copy(state.x)
        S_new, tape, ws = _deer_solve_new_tape(
            rng, model, sampler, x0; workspace=state.workspace
        )
        logps = _trajectory_logps(model, S_new)
        x_new = S_new[:, 1]
        logp_new = logps[1]
        trans = ParallelMALATransition(x_new, logp_new)
        new_state = ParallelMALAState(x_new, logp_new, S_new, logps, ws, tape, 1, model)
        return trans, new_state
    end
end

for TKey in (Symbol, VarName)
    @eval function AbstractMCMC.bundle_samples(
        samples::Vector{<:ParallelMALATransition},
        model::DensityModel,
        sampler::ParallelMALASampler,
        state::ParallelMALAState,
        ::Type{FlexiChains.FlexiChain{$TKey}};
        param_names=nothing,
        kwargs...,
    )
        N = length(samples)
        D = model.dim
        FP = typeof(sampler.epsilon)

        vals = Matrix{FP}(undef, N, D)
        logp = Vector{FP}(undef, N)

        for i in 1:N
            vals[i, :] .= samples[i].x
            logp[i] = samples[i].logp
        end

        internals = (logp=logp,)
        return _construct_flexichain($TKey, vals, internals, param_names, model)
    end
end

"""
    AdaptiveMALASampler(epsilon_init; n_warmup, target_accept, gamma, t0, kappa, cholM)
"""
struct AdaptiveMALASampler{FP<:AbstractFloat,CM} <: AbstractMCMC.AbstractSampler
    epsilon_init::FP
    n_warmup::Int
    target_accept::FP
    gamma::FP
    t0::FP
    kappa::FP
    cholM::CM
end

function AdaptiveMALASampler(
    epsilon_init::Real;
    n_warmup::Int=1000,
    target_accept::Real=0.574,
    gamma::Real=0.05,
    t0::Real=10.0,
    kappa::Real=0.75,
    cholM=nothing,
)
    epsilon_init > 0 || throw(ArgumentError("epsilon_init must be > 0, got $epsilon_init"))
    0 < target_accept < 1 ||
        throw(ArgumentError("target_accept must be in (0,1), got $target_accept"))
    gamma > 0 || throw(ArgumentError("gamma must be > 0, got $gamma"))
    t0 > 0 || throw(ArgumentError("t0 must be > 0, got $t0"))
    0.5 < kappa <= 1.0 || throw(ArgumentError("kappa must be in (0.5, 1], got $kappa"))

    eps_f = float(epsilon_init)
    FP = typeof(eps_f)
    return AdaptiveMALASampler{FP,typeof(cholM)}(
        eps_f, n_warmup, FP(target_accept), FP(gamma), FP(t0), FP(kappa), cholM
    )
end

"""
State for an `AdaptiveMALASampler` chain, including dual-averaging adaptation
statistics. Holds the prepped model so AD preparation happens once per chain.
"""
struct AdaptiveMALAState{
    V<:AbstractVector,FP<:AbstractFloat,W,NV<:AbstractVector,H,DM<:PreppedDensityModel
}
    x::V
    logp::FP
    epsilon::FP
    epsilon_bar::FP
    H_bar::FP
    step::Int
    workspace::W
    noise::NV
    noise_host::H
    model::DM
end

"""
One `AdaptiveMALASampler` sample: parameter vector `x`, its log-density `logp`,
the step size used for that transition, and whether the sample came from warmup.
"""
struct AdaptiveMALATransition{V<:AbstractVector,FP<:AbstractFloat}
    x::V
    logp::FP
    accepted::Bool
    step_size::FP
    is_warmup::Bool
end

function _dual_average_update(
    epsilon_init::FP,
    epsilon_bar::FP,
    H_bar::FP,
    m::Int,
    logα::FP,
    sampler::AdaptiveMALASampler{FP},
) where {FP<:AbstractFloat}
    α = min(one(FP), exp(logα))
    δ = sampler.target_accept
    γ = sampler.gamma
    t0 = sampler.t0
    κ = sampler.kappa
    μ = log(10 * epsilon_init)

    inv_mt0 = one(FP) / (FP(m) + t0)
    H_bar_new = (one(FP) - inv_mt0) * H_bar + inv_mt0 * (δ - α)
    log_ε = μ - sqrt(FP(m)) / γ * H_bar_new
    mk = FP(m)^(-κ)
    log_ε_bar_new = mk * log_ε + (one(FP) - mk) * log(epsilon_bar)

    return exp(log_ε), exp(log_ε_bar_new), H_bar_new
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::AdaptiveMALASampler{FP};
    initial_params=nothing,
    kwargs...,
) where {FP}
    x = if initial_params !== nothing
        copy(initial_params)
    else
        randn(rng, FP, model.dim)
    end
    model = _prepare_model(model, x)
    logp_val = FP(model.logdensity(x))
    ws = MALA.MALAWorkspace(x)
    noise, noise_host = _make_noise_buffer(x, FP, model.dim)
    trans = AdaptiveMALATransition(x, logp_val, true, sampler.epsilon_init, true)
    state = AdaptiveMALAState(
        x,
        logp_val,
        sampler.epsilon_init,
        sampler.epsilon_init,
        zero(FP),
        0,
        ws,
        noise,
        noise_host,
        model,
    )
    return trans, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    user_model::DensityModel,
    sampler::AdaptiveMALASampler{FP},
    state::AdaptiveMALAState;
    kwargs...,
) where {FP}
    model = if _prepped_for(state.model, user_model)
        state.model
    else
        _prepare_model(user_model, state.x)
    end
    D = model.dim
    in_warmup = state.step < sampler.n_warmup
    ε = in_warmup ? state.epsilon : state.epsilon_bar

    ξ = _randn_like!(rng, state.noise, state.noise_host)
    u = rand(rng)

    x_next = similar(state.x)
    x_next, accepted, logα = MALA.mala_step_with_logα!(
        x_next,
        state.workspace,
        model.logdensity,
        model.grad_logdensity,
        state.x,
        ε,
        ξ,
        u;
        cholM=sampler.cholM,
    )

    logp_next = accepted ? FP(model.logdensity(x_next)) : state.logp

    m_new = state.step + 1
    ε_new, ε_bar_new, H_bar_new = if in_warmup
        _dual_average_update(
            sampler.epsilon_init,
            state.epsilon_bar,
            state.H_bar,
            m_new,
            FP(logα),
            sampler,
        )
    else
        state.epsilon, state.epsilon_bar, state.H_bar
    end

    trans = AdaptiveMALATransition(x_next, logp_next, accepted, ε, in_warmup)
    new_state = AdaptiveMALAState(
        x_next,
        logp_next,
        ε_new,
        ε_bar_new,
        H_bar_new,
        m_new,
        state.workspace,
        state.noise,
        state.noise_host,
        model,
    )
    return trans, new_state
end

for TKey in (Symbol, VarName)
    @eval function AbstractMCMC.bundle_samples(
        samples::Vector{<:AdaptiveMALATransition},
        model::DensityModel,
        sampler::AdaptiveMALASampler,
        state::AdaptiveMALAState,
        ::Type{FlexiChains.FlexiChain{$TKey}};
        param_names=nothing,
        discard_warmup=false,
        kwargs...,
    )
        filtered = discard_warmup ? filter(s -> !s.is_warmup, samples) : samples
        N = length(filtered)
        D = model.dim
        FP = typeof(sampler.epsilon_init)

        vals = Matrix{FP}(undef, N, D)
        logp = Vector{FP}(undef, N)
        accepted = Vector{Bool}(undef, N)
        step_size = Vector{FP}(undef, N)
        is_warmup = Vector{Bool}(undef, N)

        for i in 1:N
            s = filtered[i]
            vals[i, :] .= s.x
            logp[i] = s.logp
            accepted[i] = s.accepted
            step_size[i] = s.step_size
            is_warmup[i] = s.is_warmup
        end

        internals = (logp=logp, accepted=accepted, step_size=step_size, is_warmup=is_warmup)
        return _construct_flexichain($TKey, vals, internals, param_names, model)
    end
end
