#=
AbstractMCMC interface for ParallelMCMC samplers.

Defines model/sampler/state/transition types and implements
`AbstractMCMC.step` so that `sample(model, sampler, N)` works out of the box.
=#

"""
    DensityModel(logdensity, grad_logdensity, dim; param_names=nothing)
    DensityModel(logdensity, grad_logdensity, dim, param_names)

Wraps a log-density function and its gradient for use with ParallelMCMC samplers.

- `logdensity(x::AbstractVector) -> Real`
- `grad_logdensity(x::AbstractVector) -> AbstractVector`
- `dim::Int` — dimensionality of the parameter space
- `param_names` — optional `Vector{Symbol}` of parameter names used in `MCMCChains` output.
  If `nothing` (the default), names fall back to `x[1], x[2], ...`.

Parameter names can also be overridden at sampling time via the `param_names` keyword
argument of `sample(...)`.
"""
struct DensityModel{F,G} <: AbstractMCMC.AbstractModel
    logdensity::F
    grad_logdensity::G
    dim::Int
    param_names::Union{Nothing, Vector{Symbol}}
end

# Backward-compatible 3-argument constructor
DensityModel(logdensity, grad_logdensity, dim) =
    DensityModel(logdensity, grad_logdensity, dim, nothing)

"""
    MALASampler(epsilon; cholM=nothing)

Metropolis-Adjusted Langevin Algorithm sampler with step size `epsilon`.

Optionally pass `cholM = cholesky(M)` to use a mass matrix `M` as a
preconditioner.  The proposal becomes `y = x + ε M ∇logp(x) + √(2ε) L ξ`
where `L` is the Cholesky factor of `M`.
"""
struct MALASampler{FP<:AbstractFloat, CM} <: AbstractMCMC.AbstractSampler
    epsilon::FP
    cholM::CM
end

function MALASampler(epsilon::Real; cholM=nothing)
    epsilon > 0 || throw(ArgumentError("epsilon must be > 0, got $epsilon"))
    eps_f = float(epsilon)
    return MALASampler{typeof(eps_f), typeof(cholM)}(eps_f, cholM)
end

struct MALAState{V<:AbstractVector, L<:Real}
    x::V
    logp::L
end

struct MALATransition{V<:AbstractVector, L<:Real}
    x::V
    logp::L
    accepted::Bool
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
    logp_val = model.logdensity(x)
    t = MALATransition(x, logp_val, true)
    s = MALAState(x, logp_val)
    return t, s
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::MALASampler,
    state::MALAState;
    kwargs...,
)
    x = state.x
    ϵ = sampler.epsilon
    D = model.dim

    ξ = randn(rng, eltype(x), D)
    u = rand(rng)

    x_next, accepted = MALA.mala_step_full(
        model.logdensity, model.grad_logdensity, x, ϵ, ξ, u;
        cholM=sampler.cholM,
    )

    logp_val = accepted ? model.logdensity(x_next) : state.logp
    t = MALATransition(x_next, logp_val, accepted)
    s = MALAState(x_next, logp_val)
    return t, s
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:MALATransition},
    model::DensityModel,
    sampler::MALASampler,
    state::MALAState,
    ::Type{MCMCChains.Chains};
    param_names=nothing,
    kwargs...,
)
    N = length(samples)
    D = model.dim

    names = if param_names !== nothing
        param_names
    elseif model.param_names !== nothing
        model.param_names
    else
        [Symbol("x[$i]") for i in 1:D]
    end

    internal_names = [:logp, :accepted]

    vals      = Matrix{Float64}(undef, N, D)
    internals = Matrix{Float64}(undef, N, 2)

    for i in 1:N
        s = samples[i]
        vals[i, :] .= s.x
        internals[i, 1] = s.logp
        internals[i, 2] = s.accepted ? 1.0 : 0.0
    end

    return MCMCChains.Chains(
        hcat(vals, internals),
        vcat(names, internal_names),
        Dict(:parameters => names, :internals => internal_names),
    )
end

"""
    MALATapeElement(ξ, u)

One element of the MALA noise tape: a noise vector `ξ ~ N(0,I)` and a uniform
scalar `u ~ Uniform(0,1)`.  Stored with a concrete vector type `V` for type
stability inside `DEER.TapedRecursion`.
"""
struct MALATapeElement{FP<:AbstractFloat, V<:AbstractVector{FP}}
    ξ::V
    u::FP
end

"""
    DEERSampler(epsilon; T, maxiter, tol_abs, tol_rel, jacobian, damping, probes, cholM, backend)

DEER-accelerated MALA sampler.

DEER solves for a trajectory of `T` steps in parallel (O(log T) sweep levels),
then the AbstractMCMC interface returns samples from that trajectory
sequentially.  When the trajectory is exhausted a new tape is drawn and DEER
re-solves starting from the last state.

# Arguments
- `epsilon` — MALA step size.
- `T` — trajectory length per DEER solve (default 64).
- `maxiter` — maximum DEER iterations per solve (default 200).
- `tol_abs`, `tol_rel` — convergence tolerances (default 1e-6, 1e-5).
- `jacobian` — Jacobian mode: `:diag`, `:stoch_diag`, or `:full` (default `:diag`).
- `damping` — DEER damping in (0,1] (default 0.5; helps convergence).
- `probes` — Hutchinson probes for `:stoch_diag` mode (default 1).
- `cholM` — optional Cholesky factor of a mass matrix `M` (default `nothing` = identity).
- `backend` — AD backend for Jacobian computation (default `AutoMooncake()`).
  For GPU execution pass a GPU-compatible backend, e.g. `AutoEnzyme()`.

# GPU use
The parallel-prefix scan (`solve_affine_scan_diag`) runs as pure array broadcasts
and is array-type-agnostic.  To run DEER on GPU:
1. Implement `logdensity` and `grad_logdensity` using GPU-compatible operations.
2. Pass `backend = AutoEnzyme()` (or another GPU-compatible `ADTypes` backend).
3. Pass a GPU vector as `initial_params` to `sample`.

# Parallel chains
Both `MALASampler` and `DEERSampler` are compatible with
`AbstractMCMC.sample(model, sampler, MCMCThreads(), N, nchains)`.  Each chain
has its own immutable state so there is no shared mutable data.
"""
struct DEERSampler{FP<:AbstractFloat, CM, AD} <: AbstractMCMC.AbstractSampler
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

function DEERSampler(
    epsilon::Real;
    T::Int=64,
    maxiter::Int=200,
    tol_abs::Real=1e-6,
    tol_rel::Real=1e-5,
    jacobian::Symbol=:diag,
    damping::Real=0.5,
    probes::Int=1,
    cholM=nothing,
    backend=DEER.DEFAULT_BACKEND,
)
    epsilon > 0 || throw(ArgumentError("epsilon must be > 0, got $epsilon"))
    eps_f = float(epsilon)
    FP    = typeof(eps_f)
    return DEERSampler{FP, typeof(cholM), typeof(backend)}(
        eps_f, T, maxiter,
        FP(tol_abs), FP(tol_rel),
        jacobian, FP(damping), probes, cholM, backend,
    )
end

"""
State for a `DEERSampler` chain.

- `x` — current position (= `trajectory[:, t]`, the last returned sample).
- `logp` — log-density at `x`.
- `trajectory` — D×T matrix produced by the most recent DEER solve.
- `tape` — noise tape used for that solve.
- `t` — index within `trajectory` of the last returned sample (1-indexed).
"""
struct DEERState{V<:AbstractVector, L<:Real, M<:AbstractMatrix}
    x::V
    logp::L
    trajectory::M
    tape::Vector{<:MALATapeElement}
    t::Int
end

"""
One DEER sample: parameter vector `x` and its log-density `logp`.
"""
struct DEERTransition{V<:AbstractVector, L<:Real}
    x::V
    logp::L
end

function _build_mala_deer_rec(
    model::DensityModel, ε::Real, tape::Vector{<:MALATapeElement};
    cholM=nothing, backend=DEER.DEFAULT_BACKEND,
)
    logp     = model.logdensity
    gradlogp = model.grad_logdensity

    step_fwd = (x, te) -> MALA.mala_step_taped(logp, gradlogp, x, ε, te.ξ, te.u; cholM=cholM)
    step_lin = (x, te, a) -> MALA.mala_step_surrogate(logp, gradlogp, x, ε, te.ξ, a; cholM=cholM)
    consts   = (x, te) -> (MALA.mala_accept_indicator(logp, gradlogp, x, ε, te.ξ, te.u; cholM=cholM),)

    FP = typeof(float(ε))
    return DEER.TapedRecursion(
        step_fwd, step_lin, tape;
        consts=consts, const_example=(zero(FP),), backend=backend,
    )
end

function _deer_solve_new_tape(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::DEERSampler,
    x0::AbstractVector,
)
    D    = model.dim
    T    = sampler.T
    FP   = typeof(sampler.epsilon)
    # Generate noise on the same device as x0: generate on CPU then copy via copyto!.
    # copyto!(CuVector, Vector) performs a host-to-device transfer in CUDA.jl.
    tape = map(1:T) do _
        ξ = copyto!(similar(x0, D), randn(rng, FP, D))
        MALATapeElement(ξ, FP(rand(rng)))
    end
    rec  = _build_mala_deer_rec(model, sampler.epsilon, tape; cholM=sampler.cholM, backend=sampler.backend)
    S    = DEER.solve(
        rec, x0;
        tol_abs  = sampler.tol_abs,
        tol_rel  = sampler.tol_rel,
        maxiter  = sampler.maxiter,
        jacobian = sampler.jacobian,
        damping  = sampler.damping,
        probes   = sampler.probes,
        rng      = rng,
    )
    return S, tape
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::DEERSampler{FP};
    initial_params=nothing,
    kwargs...,
) where {FP}
    x0 = if initial_params !== nothing
        copy(initial_params)
    else
        randn(rng, FP, model.dim)
    end

    S, tape  = _deer_solve_new_tape(rng, model, sampler, x0)
    x1       = S[:, 1]
    logp1    = model.logdensity(x1)
    trans    = DEERTransition(x1, logp1)
    state    = DEERState(x1, logp1, S, tape, 1)
    return trans, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::DEERSampler,
    state::DEERState;
    kwargs...,
)
    T      = sampler.T
    t_next = state.t + 1

    if t_next <= T
        # Consume the next cached sample from the trajectory.
        x_new    = state.trajectory[:, t_next]
        logp_new = model.logdensity(x_new)
        trans     = DEERTransition(x_new, logp_new)
        new_state = DEERState(x_new, logp_new, state.trajectory, state.tape, t_next)
        return trans, new_state
    else
        # Trajectory exhausted — re-solve with a fresh tape.
        x0          = state.trajectory[:, T]
        S_new, tape = _deer_solve_new_tape(rng, model, sampler, x0)
        x_new    = S_new[:, 1]
        logp_new = model.logdensity(x_new)
        trans     = DEERTransition(x_new, logp_new)
        new_state = DEERState(x_new, logp_new, S_new, tape, 1)
        return trans, new_state
    end
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:DEERTransition},
    model::DensityModel,
    sampler::DEERSampler,
    state::DEERState,
    ::Type{MCMCChains.Chains};
    param_names=nothing,
    kwargs...,
)
    N = length(samples)
    D = model.dim

    names = if param_names !== nothing
        param_names
    elseif model.param_names !== nothing
        model.param_names
    else
        [Symbol("x[$i]") for i in 1:D]
    end

    internal_names = [:logp]

    vals      = Matrix{Float64}(undef, N, D)
    internals = Matrix{Float64}(undef, N, 1)

    for i in 1:N
        vals[i, :]   .= samples[i].x
        internals[i, 1] = samples[i].logp
    end

    return MCMCChains.Chains(
        hcat(vals, internals),
        vcat(names, internal_names),
        Dict(:parameters => names, :internals => internal_names),
    )
end

"""
    AdaptiveMALASampler(epsilon_init; n_warmup, target_accept, gamma, t0, kappa, cholM)

MALA sampler with automatic step-size adaptation via dual averaging
(Nesterov 2009, as used in NUTS — Hoffman & Gelman 2014).

During the first `n_warmup` steps the step size `ε` is adapted online to drive
the Metropolis acceptance rate toward `target_accept` (default 0.574, which is
the asymptotically optimal rate for MALA in high dimensions).  After warmup the
smoothed estimate `ε̄` is frozen and used for all remaining steps.

# Algorithm
At warmup step `m`, given current log-acceptance ratio `logα`:

    α   = min(1, exp(logα))
    H̄_m = (1 − 1/(m+t₀)) H̄_{m−1} + (1/(m+t₀)) (δ − α)
    log ε_m  = μ − √m/γ · H̄_m               (instantaneous)
    log ε̄_m  = m^(−κ) log ε_m + (1−m^(−κ)) log ε̄_{m−1}   (smoothed)

where μ = log(10 ε₀) is a fixed target.  After warmup `ε̄` is used.

# Keyword arguments
- `n_warmup` — adaptation steps (default 1000).
- `target_accept` — δ, desired acceptance rate (default 0.574).
- `gamma` — γ, regularisation strength (default 0.05).
- `t0` — stability offset (default 10.0).
- `kappa` — shrinkage exponent κ ∈ (0.5, 1] (default 0.75).
- `cholM` — optional Cholesky factor of a mass matrix `M` (default `nothing` = identity).

# MCMCChains output
The `Chains` object includes internals `[:logp, :accepted, :step_size, :is_warmup]`.
After warmup, `step_size` is constant (the frozen `ε̄`).

# Parallel chains
Works with `MCMCThreads()`.  R-hat and ESS are computed automatically by
`MCMCChains` from multi-chain output.

# Turing.jl / LogDensityProblems
Load `LogDensityProblems` (and optionally `LogDensityProblemsAD`) then use the
`DensityModel(ld)` constructor to wrap any `LogDensityProblems`-compatible model
(including Turing/DynamicPPL models) directly.
"""
struct AdaptiveMALASampler{FP<:AbstractFloat, CM} <: AbstractMCMC.AbstractSampler
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
    epsilon_init > 0   || throw(ArgumentError("epsilon_init must be > 0, got $epsilon_init"))
    0 < target_accept < 1 || throw(ArgumentError("target_accept must be in (0,1), got $target_accept"))
    gamma > 0          || throw(ArgumentError("gamma must be > 0, got $gamma"))
    t0 > 0             || throw(ArgumentError("t0 must be > 0, got $t0"))
    0.5 < kappa <= 1.0 || throw(ArgumentError("kappa must be in (0.5, 1], got $kappa"))

    eps_f = float(epsilon_init)
    FP    = typeof(eps_f)
    return AdaptiveMALASampler{FP, typeof(cholM)}(
        eps_f, n_warmup,
        FP(target_accept), FP(gamma), FP(t0), FP(kappa),
        cholM,
    )
end

struct AdaptiveMALAState{V<:AbstractVector, FP<:AbstractFloat}
    x::V
    logp::FP
    epsilon::FP        # instantaneous step size ε_m
    epsilon_bar::FP    # smoothed step size ε̄_m  (frozen after warmup)
    H_bar::FP          # dual-average statistic H̄_m
    step::Int          # warmup step counter (0 = initialisation)
end

struct AdaptiveMALATransition{V<:AbstractVector, FP<:AbstractFloat}
    x::V
    logp::FP
    accepted::Bool
    step_size::FP   # ε used for this step
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
    α   = min(one(FP), exp(logα))
    δ   = sampler.target_accept
    γ   = sampler.gamma
    t0  = sampler.t0
    κ   = sampler.kappa
    μ   = log(10 * epsilon_init)   # fixed shrinkage target

    inv_mt0   = one(FP) / (FP(m) + t0)
    H_bar_new = (one(FP) - inv_mt0) * H_bar + inv_mt0 * (δ - α)
    log_ε     = μ - sqrt(FP(m)) / γ * H_bar_new
    mk        = FP(m)^(-κ)
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
    logp_val = FP(model.logdensity(x))
    trans    = AdaptiveMALATransition(x, logp_val, true, sampler.epsilon_init, true)
    state    = AdaptiveMALAState(x, logp_val, sampler.epsilon_init, sampler.epsilon_init, zero(FP), 0)
    return trans, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::AdaptiveMALASampler{FP},
    state::AdaptiveMALAState;
    kwargs...,
) where {FP}
    D         = model.dim
    in_warmup = state.step < sampler.n_warmup
    ε         = in_warmup ? state.epsilon : state.epsilon_bar

    ξ = randn(rng, eltype(state.x), D)
    u = rand(rng)

    x_next, accepted, logα = MALA.mala_step_with_logα(
        model.logdensity, model.grad_logdensity, state.x, ε, ξ, u;
        cholM=sampler.cholM,
    )

    logp_next = accepted ? FP(model.logdensity(x_next)) : state.logp

    # Dual-average adaptation (only during warmup)
    m_new = state.step + 1
    ε_new, ε_bar_new, H_bar_new = if in_warmup
        _dual_average_update(
            sampler.epsilon_init, state.epsilon_bar, state.H_bar,
            m_new, FP(logα), sampler,
        )
    else
        state.epsilon, state.epsilon_bar, state.H_bar
    end

    trans     = AdaptiveMALATransition(x_next, logp_next, accepted, ε, in_warmup)
    new_state = AdaptiveMALAState(x_next, logp_next, ε_new, ε_bar_new, H_bar_new, m_new)
    return trans, new_state
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:AdaptiveMALATransition},
    model::DensityModel,
    sampler::AdaptiveMALASampler,
    state::AdaptiveMALAState,
    ::Type{MCMCChains.Chains};
    param_names=nothing,
    discard_warmup=false,
    kwargs...,
)
    # Optionally remove warmup samples before building the chain.
    filtered = discard_warmup ? filter(s -> !s.is_warmup, samples) : samples
    N = length(filtered)
    D = model.dim

    names = if param_names !== nothing
        param_names
    elseif model.param_names !== nothing
        model.param_names
    else
        [Symbol("x[$i]") for i in 1:D]
    end

    internal_names = [:logp, :accepted, :step_size, :is_warmup]

    vals      = Matrix{Float64}(undef, N, D)
    internals = Matrix{Float64}(undef, N, 4)

    for i in 1:N
        s = filtered[i]
        vals[i, :]  .= s.x
        internals[i, 1] = s.logp
        internals[i, 2] = s.accepted  ? 1.0 : 0.0
        internals[i, 3] = s.step_size
        internals[i, 4] = s.is_warmup ? 1.0 : 0.0
    end

    return MCMCChains.Chains(
        hcat(vals, internals),
        vcat(names, internal_names),
        Dict(:parameters => names, :internals => internal_names),
    )
end

"""
    DensityModel(ld)

Construct a `DensityModel` from any object `ld` that implements the
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl) interface,
i.e. provides `LogDensityProblems.logdensity`, `LogDensityProblems.logdensity_and_gradient`,
and `LogDensityProblems.dimension`.

Requires the `LogDensityProblems` package to be loaded:

```julia
using LogDensityProblems          # gradient-free: only logdensity used
using LogDensityProblemsAD        # or this, for AD-based gradients
```

# Turing.jl example
```julia
using Turing, LogDensityProblems, LogDensityProblemsAD, Mooncake

@model function mymodel(data)
    μ ~ Normal(0, 1)
    data ~ Normal(μ, 1)
end

ld  = DynamicPPL.LogDensityFunction(mymodel(obs))
ldg = LogDensityProblemsAD.ADgradient(ADTypes.AutoMooncake(; config=nothing), ld)
model = DensityModel(ldg)

chain = sample(model, AdaptiveMALASampler(0.1; n_warmup=500), 2000;
               chain_type=MCMCChains.Chains, progress=true)
```

This method is defined in the `LogDensityProblemsExt` extension and is only
available when `LogDensityProblems` has been loaded.
"""
function DensityModel end   # extended by LogDensityProblemsExt
