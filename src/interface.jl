#=
AbstractMCMC interface for ParallelMCMC samplers.

Defines model/sampler/state/transition types and implements
`AbstractMCMC.step` so that `sample(model, sampler, N)` works out of the box.
=#

"""
    DensityModel(logdensity, grad_logdensity, dim; param_names, logdensity_batch, grad_logdensity_batch, hvp, hvp_batch)

Wraps a log-density function, its gradient, and optional Hessian-vector
product helpers for use with ParallelMCMC samplers.

The derivative slots (`grad_logdensity`, `hvp`, `grad_logdensity_batch`,
`hvp_batch`) take either a callable or an `ADTypes.AbstractADType`. Backends
are turned into prepared DifferentiationInterface callables when sampling
starts, and any AD failure surfaces there. So a model needs nothing beyond
the log-density:

    DensityModel(logp, AutoForwardDiff(), dim)

How a backend in `hvp` / `hvp_batch` gets its second derivative depends on the
gradient slot. Over a hand-written gradient it is a single AD pass across your
own code. Over an AD-derived one it is
`DifferentiationInterface.SecondOrder(hvp_backend, grad_backend)`, taken through
DI's second-order operator. Passing a `SecondOrder` yourself always means the
latter, and bypasses the gradient slot even when you wrote it by hand.

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
update evaluates directly. `logdensity_batch` alone is allowed and scores whole
trajectories at once without switching the batched update on.

`ParallelMALASampler` runs the batched DEER update once it has a
`logdensity_batch` and a batched gradient: either `grad_logdensity_batch`, or one
derived from `logdensity_batch` when `grad_logdensity` is a backend.
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
    #= The batched DEER update evaluates `logdensity_batch` itself, so neither
    batched derivative is usable without one: a backend would have nothing to
    differentiate, and a callable would never be reached. Rejected here rather
    than silently ignored. A `logdensity_batch` on its own is allowed, and
    `_prepare_model` decides from the gradient slot whether the path can run. =#
    _batch_needs_logp(name) = throw(
        ArgumentError(
            "$name requires logdensity_batch: the batched DEER path evaluates the " *
            "batched log-density, so it cannot run without one",
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
A [`DensityModel`](@ref) with its backend slots resolved to prepared
DifferentiationInterface callables. `_prepare_model` builds these, and the
sampler internals take them rather than a `DensityModel`, so no slot of one
of these ever holds an `AbstractADType`. Which slots are filled depends on
the sampler: the sequential samplers only need `grad_logdensity` and get
`nothing` for the DEER-only slots, DEER fills the rest.

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
anything else falls back to unprepared `DI.gradient` rather than failing. In a
normal run the prepared branch is the one that fires.
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
Resolve an HVP slot given as a backend. `grad_backend` is the backend that
produced `grad`, or nothing when the gradient slot held a callable. Dispatch is
on types alone, so the branch folds and the returned closure type stays
statically known.

A `SecondOrder` bypasses the gradient slot even when that slot is hand-written:
naming both passes asks for two derivatives of `logdensity`. The slot is still
the drift term the MALA step uses.
=#
function _resolve_hvp(logdensity, grad, grad_backend, hvp_backend, x_template)
    if hvp_backend isa DI.SecondOrder
        return DEER._make_hvp_fn_second_order(logdensity, hvp_backend, x_template)
    elseif grad_backend !== nothing
        return DEER._make_hvp_fn_second_order(
            logdensity, DI.SecondOrder(hvp_backend, grad_backend), x_template
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
            DI.SecondOrder(hvp_backend, grad_batch_backend),
            X_template,
        )
    else
        return DEER._make_hvp_batch_fn(
            DEER._hvp_strategy(hvp_backend), grad_batch, hvp_backend, X_template
        )
    end
end

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
    #= The derivative slots DEER alone reads are dropped rather than passed
    along: a sequential sampler never looks at them, and carrying an
    unresolved backend would break the invariant that no slot here holds an
    `AbstractADType`. =#
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
    grad = if grad_backend !== nothing
        _resolve_gradient(model.logdensity, grad_backend, x_template)
    else
        model.grad_logdensity
    end

    hvp = if model.hvp === nothing || model.hvp isa AbstractADType
        hvp_backend = model.hvp === nothing ? backend : model.hvp
        hvp_backend === nothing && throw(
            ArgumentError(
                "ParallelMALASampler needs a Hessian-vector product: supply `hvp` " *
                "on the DensityModel (callable or AD backend), or pass `backend=` " *
                "to ParallelMALASampler",
            ),
        )
        _resolve_hvp(model.logdensity, grad, grad_backend, hvp_backend, x_template)
    else
        model.hvp
    end

    #= A batched log-density with no batched gradient gets one from the model's
    own gradient backend, never the sampler's: a model with a hand-written
    gradient has not opted into AD, and deriving one anyway would let `backend=`
    decide which update path runs. Failing to derive leaves the path off rather
    than raising, since `_trajectory_logps` uses `logdensity_batch` regardless. =#
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

        if grad_batch_backend !== nothing
            grad_batch = _resolve_gradient_batch(
                model.logdensity_batch, grad_batch_backend, X_template
            )
        end

        if hvp_batch === nothing || hvp_batch isa AbstractADType
            # The model's own HVP backend if it has one, else the sampler's.
            hvp_batch_backend = if hvp_batch === nothing
                model.hvp isa AbstractADType ? model.hvp : backend
            else
                hvp_batch
            end
            hvp_batch_backend === nothing && throw(
                ArgumentError(
                    "the batched DEER path needs a batched Hessian-vector product: " *
                    "supply `hvp_batch` on the DensityModel (callable or AD backend), " *
                    "or pass `backend=` to ParallelMALASampler",
                ),
            )
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

#= Callable structs that allow us to dispatch on the type of the LogDensityProblems object in
the postprocessing stage. Ideally these would be defined in the LogDensityProblemsExt.
However, structs defined in extensions are hard to get hold of so we define them here.
The callable behaviour itself is implemented in LogDensityProblemsExt =#
struct LogDensityProblemPrimal{L}
    ld::L
end
struct LogDensityProblemGradient{L}
    ld::L
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
    host = needs_host_staging(ξ) ? Vector{FP}(undef, D) : nothing
    return ξ, host
end

function _randn_like!(
    rng::Random.AbstractRNG, ξ::AbstractVector{FP}, host::Union{Nothing,AbstractVector{FP}}
) where {FP}
    if needs_host_staging(ξ)
        host === nothing &&
            error("normal noise for $(typeof(ξ)) requires a reusable host buffer")
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

`backend` is the fallback source of Hessian-vector products, used when the
`DensityModel` brings no `hvp` / `hvp_batch` of its own. That is all it does: it
never supplies a gradient, so it cannot change which update path runs or put AD
on a function the model did not already have a backend for. A model carrying its
own HVPs does not need it.
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

    if needs_host_staging(Xi)
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
