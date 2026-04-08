#=
AbstractMCMC interface for ParallelMCMC samplers.

Defines model/sampler/state/transition types and implements
`AbstractMCMC.step` so that `sample(model, sampler, N)` works out of the box.
=#

"""
    DensityModel(logdensity, grad_logdensity, dim; param_names, logdensity_batch, grad_logdensity_batch)

Wraps a log-density function and its gradient for use with ParallelMCMC samplers.

- `logdensity(x::AbstractVector) -> Real`
- `grad_logdensity(x::AbstractVector) -> AbstractVector`
- `dim::Int` — dimensionality of the parameter space
- `param_names` — optional `Vector{Symbol}` of parameter names used in `MCMCChains` output.
  If `nothing` (the default), names fall back to `x[1], x[2], ...`.

# Optional batched functions (required for GPU-efficient `ParallelMALASampler`)

- `logdensity_batch(X::AbstractMatrix) -> AbstractVector` — evaluates `logdensity` on
  every column of `X` (D×N) and returns a length-N vector.
- `grad_logdensity_batch(X::AbstractMatrix) -> AbstractMatrix` — evaluates
  `grad_logdensity` on every column of `X` and returns a D×N matrix.

When both batched functions are provided, `ParallelMALASampler` switches to a
**batched DEER update** that processes all T trajectory steps simultaneously using
D×T matrix operations instead of a sequential loop.  This is the key for GPU
efficiency: it replaces 6T sequential gradient calls with ~8 batched calls over
the full matrix, mirroring JAX's `jax.vmap` pattern.

For GPU (CuArray inputs), implement the batched functions via vectorised operations:

```julia
logp_batch(X) = map(t -> logp(X[:,t]), 1:size(X,2))           # CPU only
gradlogp_batch(X) = hcat([gradlogp(X[:,t]) for t in 1:size(X,2)]...)   # CPU only

# GPU: implement as true matrix operations, e.g. for a Gaussian target:
logp_batch(X)        = -0.5f0 .* vec(sum(abs2, X; dims=1))
gradlogp_batch(X)    = -X
```

Parameter names can also be overridden at sampling time via the `param_names` keyword
argument of `sample(...)`.
"""
struct DensityModel{F,G,FB,GB} <: AbstractMCMC.AbstractModel
    logdensity::F
    grad_logdensity::G
    logdensity_batch::FB          # (D×N AbstractMatrix) -> N-AbstractVector, or Nothing
    grad_logdensity_batch::GB     # (D×N AbstractMatrix) -> D×N AbstractMatrix, or Nothing
    dim::Int
    param_names::Union{Nothing,Vector{Symbol}}
end

"""
Primary constructor — accepts optional batched functions as keyword arguments.
"""
function DensityModel(
    logdensity,
    grad_logdensity,
    dim::Int;
    param_names=nothing,
    logdensity_batch=nothing,
    grad_logdensity_batch=nothing,
)
    return DensityModel(
        logdensity, grad_logdensity,
        logdensity_batch, grad_logdensity_batch,
        dim, param_names,
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

struct MALAState{V<:AbstractVector,L<:Real}
    x::V
    logp::L
end

struct MALATransition{V<:AbstractVector,L<:Real}
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
        model.logdensity, model.grad_logdensity, x, ϵ, ξ, u; cholM=sampler.cholM
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

    vals = Matrix{Float64}(undef, N, D)
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
struct MALATapeElement{FP<:AbstractFloat,V<:AbstractVector{FP}}
    ξ::V
    u::FP
end

"""
    ParallelMALASampler(epsilon; T, maxiter, tol_abs, tol_rel, jacobian, damping, probes, cholM, backend)

DEER-parallelized MALA sampler (arXiv:2508.18413, Section 3.2).

DEER solves for a trajectory of `T` steps in parallel (O(log T) sweep levels),
returning the same samples as sequential MALA up to numerical tolerance.
The AbstractMCMC interface returns samples from that trajectory one-by-one;
when the trajectory is exhausted a new tape is drawn and DEER re-solves from
the last state.

The Jacobian surrogate uses the sigmoid stop-gradient trick from the paper:
`g = σ(logα − log u)` gives a differentiable approximation that captures both
the proposal direction and the acceptance-probability gradient, while the forward
pass remains the exact binary accept/reject.

# Arguments
- `epsilon` — MALA step size.
- `T` — trajectory length per DEER solve (default 64).
- `maxiter` — maximum DEER iterations per solve (default 200).
- `tol_abs`, `tol_rel` — convergence tolerances (default 1e-6, 1e-5).
- `jacobian` — Jacobian mode: `:stoch_diag` (Hutchinson, default) or `:diag` (exact diagonal,
  D JVPs per step — useful for debugging on small CPU problems).
- `damping` — DEER damping factor in (0,1] (default 0.5).
- `probes` — Hutchinson probes for `:stoch_diag` (default 1; GPU is always 1).
- `cholM` — optional Cholesky factor of a mass matrix `M` (default `nothing` = identity).
- `backend` — AD backend (default `AutoEnzyme(mode=Enzyme.Forward)`).
  Pass a GPU-compatible backend for GPU execution.

# GPU use
Pass a GPU vector as `initial_params`.  Use `jacobian=:stoch_diag` (the default) and
`probes=1` (one JVP per Newton step — matches the paper's GPU experiments, Section 5.2).

# Parallel chains
Compatible with `AbstractMCMC.sample(model, sampler, MCMCThreads(), N, nchains)`.
Each chain has its own immutable state with no shared mutable data.
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
    backend=DEER.DEFAULT_BACKEND,
)
    epsilon > 0 || throw(ArgumentError("epsilon must be > 0, got $epsilon"))
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
State for a `ParallelMALASampler` chain.

- `x` — current position (last returned sample).
- `logp` — log-density at `x`.
- `trajectory` — D×T matrix from the most recent DEER solve.
- `tape` — noise tape used for that solve.
- `t` — index within `trajectory` of the last returned sample (1-indexed).
"""
struct ParallelMALAState{V<:AbstractVector,L<:Real,M<:AbstractMatrix}
    x::V
    logp::L
    trajectory::M
    tape::Vector{<:MALATapeElement}
    t::Int
end

"""
One `ParallelMALASampler` sample: parameter vector `x` and its log-density `logp`.
"""
struct ParallelMALATransition{V<:AbstractVector,L<:Real}
    x::V
    logp::L
end

function _build_mala_deer_rec(
    model::DensityModel,
    ε::Real,
    tape::Vector{<:MALATapeElement},
    x0_like::AbstractVector;
    cholM=nothing,
    backend=DEER.DEFAULT_BACKEND,
)
    logp = model.logdensity
    gradlogp = model.grad_logdensity

    hvp_fn = (pt, dir) -> DEER._hvp_nopre(gradlogp, backend, pt, dir)

    # step_fwd: exact binary accept/reject — defines the fixed point DEER converges to.
    step_fwd =
        (x, te) -> MALA.mala_step_taped(logp, gradlogp, x, ε, te.ξ, te.u; cholM=cholM)

    # jvp: explicit analytical JVP of the sigmoid surrogate w.r.t. x (paper Section 3.2).
    # Avoids nested AD when gradlogp is itself AD-computed (e.g. via LogDensityProblemsAD).
    # Inner HVPs H(x)v are computed via a fresh DI.pushforward of gradlogp.
    jvp = (x, te, v) -> MALA.mala_step_surrogate_sigmoid_jvp(
        logp, gradlogp, x, ε, te.ξ, te.u, v, hvp_fn; cholM=cholM
    )

    # fwd_and_jvp: fused — shares the primal gradlogp/logp evaluations between
    # step_fwd and the JVP, halving gradient calls per DEER time step.
    fwd_and_jvp = (x, te, v) -> MALA.mala_step_taped_and_jvp(
        logp, gradlogp, x, ε, te.ξ, te.u, v, hvp_fn; cholM=cholM
    )

    # fwd_and_jvp_batch: GPU-efficient batched path.
    # Pre-stacks the tape (Ξ, U) into device-appropriate matrices so that
    # each deer_update call pays only one small host-to-device transfer for U
    # instead of T transfers.  Requires model.logdensity_batch /
    # model.grad_logdensity_batch to be provided by the user.
    fwd_and_jvp_batch = if model.logdensity_batch !== nothing &&
                           model.grad_logdensity_batch !== nothing
        logp_batch = model.logdensity_batch
        gradlogp_batch = model.grad_logdensity_batch
        T = length(tape)
        D = length(x0_like)

        # Pre-stack noise vectors into a D×T matrix on the same device as x0_like.
        Ξ_stacked = similar(x0_like, D, T)
        for t in 1:T
            copyto!(view(Ξ_stacked, :, t), tape[t].ξ)
        end
        # Uniforms are CPU scalars; stack and transfer once.
        U_stacked = copyto!(similar(x0_like, T), [tape[t].u for t in 1:T])

        # The actual batch function captures the pre-stacked tape.
        (Xbar, Z) -> MALA.mala_fwd_and_jvp_batched(
            logp_batch, gradlogp_batch, Xbar, ε, Ξ_stacked, U_stacked, Z; cholM=cholM
        )
    else
        nothing
    end

    return DEER.TapedRecursion(
        step_fwd, jvp, tape;
        fwd_and_jvp=fwd_and_jvp,
        fwd_and_jvp_batch=fwd_and_jvp_batch,
    )
end

function _deer_solve_new_tape(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::ParallelMALASampler,
    x0::AbstractVector,
)
    D = model.dim
    T = sampler.T
    FP = typeof(sampler.epsilon)
    # Generate noise on the same device as x0.
    # copyto!(CuVector, Vector) performs a host-to-device transfer in CUDA.jl.
    tape = map(1:T) do _
        ξ = copyto!(similar(x0, D), randn(rng, FP, D))
        MALATapeElement(ξ, FP(rand(rng)))
    end
    rec = _build_mala_deer_rec(
        model, sampler.epsilon, tape, x0; cholM=sampler.cholM, backend=sampler.backend
    )
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
    )
    return S, tape
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

    S, tape = _deer_solve_new_tape(rng, model, sampler, x0)
    x1 = S[:, 1]
    logp1 = model.logdensity(x1)
    trans = ParallelMALATransition(x1, logp1)
    state = ParallelMALAState(x1, logp1, S, tape, 1)
    return trans, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::ParallelMALASampler,
    state::ParallelMALAState;
    kwargs...,
)
    T = sampler.T
    t_next = state.t + 1

    if t_next <= T
        x_new = state.trajectory[:, t_next]
        logp_new = model.logdensity(x_new)
        trans = ParallelMALATransition(x_new, logp_new)
        new_state = ParallelMALAState(x_new, logp_new, state.trajectory, state.tape, t_next)
        return trans, new_state
    else
        # Trajectory exhausted — re-solve with a fresh tape.
        x0 = state.trajectory[:, T]
        S_new, tape = _deer_solve_new_tape(rng, model, sampler, x0)
        x_new = S_new[:, 1]
        logp_new = model.logdensity(x_new)
        trans = ParallelMALATransition(x_new, logp_new)
        new_state = ParallelMALAState(x_new, logp_new, S_new, tape, 1)

        return trans, new_state
    end
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:ParallelMALATransition},
    model::DensityModel,
    sampler::ParallelMALASampler,
    state::ParallelMALAState,
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

    vals = Matrix{Float64}(undef, N, D)
    internals = Matrix{Float64}(undef, N, 1)

    for i in 1:N
        vals[i, :] .= samples[i].x
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

struct AdaptiveMALAState{V<:AbstractVector,FP<:AbstractFloat}
    x::V
    logp::FP
    epsilon::FP        # instantaneous step size ε_m
    epsilon_bar::FP    # smoothed step size ε̄_m  (frozen after warmup)
    H_bar::FP          # dual-average statistic H̄_m
    step::Int          # warmup step counter (0 = initialisation)
end

struct AdaptiveMALATransition{V<:AbstractVector,FP<:AbstractFloat}
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
    α = min(one(FP), exp(logα))
    δ = sampler.target_accept
    γ = sampler.gamma
    t0 = sampler.t0
    κ = sampler.kappa
    μ = log(10 * epsilon_init)   # fixed shrinkage target

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
    logp_val = FP(model.logdensity(x))
    trans = AdaptiveMALATransition(x, logp_val, true, sampler.epsilon_init, true)
    state = AdaptiveMALAState(
        x, logp_val, sampler.epsilon_init, sampler.epsilon_init, zero(FP), 0
    )
    return trans, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::AdaptiveMALASampler{FP},
    state::AdaptiveMALAState;
    kwargs...,
) where {FP}
    D = model.dim
    in_warmup = state.step < sampler.n_warmup
    ε = in_warmup ? state.epsilon : state.epsilon_bar

    ξ = randn(rng, eltype(state.x), D)
    u = rand(rng)

    x_next, accepted, logα = MALA.mala_step_with_logα(
        model.logdensity, model.grad_logdensity, state.x, ε, ξ, u; cholM=sampler.cholM
    )

    logp_next = accepted ? FP(model.logdensity(x_next)) : state.logp

    # Dual-average adaptation (only during warmup)
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

    vals = Matrix{Float64}(undef, N, D)
    internals = Matrix{Float64}(undef, N, 4)

    for i in 1:N
        s = filtered[i]
        vals[i, :] .= s.x
        internals[i, 1] = s.logp
        internals[i, 2] = s.accepted ? 1.0 : 0.0
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
