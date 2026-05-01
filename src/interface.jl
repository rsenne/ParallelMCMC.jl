#=
AbstractMCMC interface for ParallelMCMC samplers.

Defines model/sampler/state/transition types and implements
`AbstractMCMC.step` so that `sample(model, sampler, N)` works out of the box.
=#

"""
    DensityModel(logdensity, grad_logdensity, dim; param_names, logdensity_batch, grad_logdensity_batch, hvp, hvp_batch)

Wraps a log-density function, its gradient, and optional Hessian-vector
product helpers for use with ParallelMCMC samplers.

- `logdensity(x::AbstractVector) -> Real`
- `grad_logdensity(x::AbstractVector) -> AbstractVector`
- `hvp(x::AbstractVector, v::AbstractVector) -> AbstractVector` — optional
  model-specific Hessian-vector product used by DEER to avoid AD through
  problematic kernels when available. If omitted, the sampler differentiates
  `logdensity` directly via DifferentiationInterface.
- `logdensity_batch(X::AbstractMatrix) -> AbstractVector` — optional batched
  log-density over columns.
- `grad_logdensity_batch(X::AbstractMatrix) -> AbstractMatrix` — optional batched
  gradient over columns.
- `hvp_batch(X::AbstractMatrix, V::AbstractMatrix) -> AbstractMatrix` — optional
  batched Hessian-vector product over columns. If omitted, the sampler can
  differentiate `grad_logdensity_batch` to keep the batched DEER path enabled.
- `dim::Int` — dimensionality of the parameter space
- `param_names` — optional `Vector{Symbol}` of parameter names used in `MCMCChains` output.
  If `nothing` (the default), names fall back to `x[1], x[2], ...`.

When `logdensity_batch` and `grad_logdensity_batch` are provided,
`ParallelMALASampler` enables the batched DEER update path.
"""
struct DensityModel{F,G,H,FB,GB,HB} <: AbstractMCMC.AbstractModel
    logdensity::F
    grad_logdensity::G
    hvp::H
    logdensity_batch::FB
    grad_logdensity_batch::GB
    hvp_batch::HB
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
    hvp=nothing,
    logdensity_batch=nothing,
    grad_logdensity_batch=nothing,
    hvp_batch=nothing,
)
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
State for a `MALASampler` chain.
"""
struct MALAState{V<:AbstractVector,L<:Real,W,NV<:AbstractVector,H}
    x::V
    logp::L
    workspace::W
    noise::NV
    noise_host::H
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
    logp_val = model.logdensity(x)
    ws = MALA.MALAWorkspace(x)
    noise, noise_host = _make_noise_buffer(x, FP, model.dim)
    t = MALATransition(x, logp_val, true)
    s = MALAState(x, logp_val, ws, noise, noise_host)
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
    s = MALAState(x_next, logp_val, state.workspace, state.noise, state.noise_host)
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

DEER-parallelized MALA sampler.

Supported Jacobian modes are `:stoch_diag` (the default Hutchinson diagonal
estimator) and `:diag` (exact diagonal via `D` JVPs).
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
State for a `ParallelMALASampler` chain.
"""
struct ParallelMALAState{V<:AbstractVector,L<:Real,M<:AbstractMatrix,LV<:AbstractVector,W}
    x::V
    logp::L
    trajectory::M
    logps::LV
    workspace::W
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
    model::DensityModel,
    ε::Real,
    tape::Vector{<:MALATapeElement},
    x0_like::AbstractVector;
    cholM=nothing,
    backend=DEER.DEFAULT_BACKEND,
    tape_noise=nothing,
    tape_uniforms=nothing,
)
    logp = model.logdensity
    gradlogp = model.grad_logdensity

    # Use a model-provided HVP when available. Otherwise differentiate the
    # scalar logdensity directly
    hvp_fn = if model.hvp !== nothing
        model.hvp
    else
        hvp_backend = DEER._hvp_second_order_backend(backend)
        prep_hvp = DEER._prepare_logdensity_hvp(logp, hvp_backend, x0_like)
        (pt, dir) -> DEER._logdensity_hvp_prepared(logp, prep_hvp, hvp_backend, pt, dir)
    end

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
            hvp_batch = if model.hvp_batch !== nothing
                model.hvp_batch
            else
                hvp_batch_backend = DEER._hvp_backend(backend)
                prep_hvp_batch = DEER._prepare_batch_hvp_from_grad(
                    model.grad_logdensity_batch, hvp_batch_backend, X_template
                )
                (X, V) -> DEER._batch_hvp_from_grad_prepared(
                    model.grad_logdensity_batch,
                    prep_hvp_batch,
                    hvp_batch_backend,
                    X,
                    V,
                )
            end

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
    model::DensityModel,
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
        backend=sampler.backend,
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

function _trajectory_logps(model::DensityModel, S::AbstractMatrix)
    if model.logdensity_batch !== nothing
        return Array(model.logdensity_batch(S))
    end

    T = size(S, 2)
    return [model.logdensity(S[:, t]) for t in 1:T]
end

function _parallel_mala_param_names(model::DensityModel, D::Int, param_names)
    if param_names !== nothing
        return param_names
    elseif model.param_names !== nothing
        return model.param_names
    else
        return [Symbol("x[$i]") for i in 1:D]
    end
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
    vals::AbstractMatrix{Float64}, first_row::Int, S::AbstractMatrix, ncols::Int
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
    N::Int;
    initial_params=nothing,
    param_names=nothing,
    progress=AbstractMCMC.PROGRESS[],
    progressname="Sampling",
)
    D = model.dim
    names = _parallel_mala_param_names(model, D, param_names)
    internal_names = [:logp]

    vals = Matrix{Float64}(undef, N, D)
    internals = Matrix{Float64}(undef, N, 1)

    progress = _parallel_mala_progress(progress, progressname)
    x0 = _parallel_mala_initial_x(rng, model, sampler, initial_params)
    ws = nothing
    nsteps = 0
    next_update = N / AbstractMCMC.get_n_updates(progress)
    threshold = next_update

    AbstractMCMC.@maybewithricherlogger begin
        AbstractMCMC.init_progress!(progress)
        try
            while nsteps < N
                S, tape, ws = _deer_solve_new_tape(rng, model, sampler, x0; workspace=ws)
                logps = _trajectory_logps(model, S)
                nkeep = min(sampler.T, N - nsteps)
                first_row = nsteps + 1
                rows = first_row:(first_row + nkeep - 1)

                _copy_trajectory_rows!(vals, first_row, S, nkeep)
                internals[rows, 1] .= view(logps, 1:nkeep)

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

    return MCMCChains.Chains(
        hcat(vals, internals),
        vcat(names, internal_names),
        Dict(:parameters => names, :internals => internal_names),
    )
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
    ws = nothing
    nsteps = 0
    next_update = N / AbstractMCMC.get_n_updates(progress)
    threshold = next_update
    final_state = nothing

    AbstractMCMC.@maybewithricherlogger begin
        AbstractMCMC.init_progress!(progress)
        try
            while nsteps < N
                S, tape, ws = _deer_solve_new_tape(rng, model, sampler, x0; workspace=ws)
                logps = _trajectory_logps(model, S)
                nkeep = min(sampler.T, N - nsteps)
                S_keep = copy(view(S, :, 1:nkeep))
                logps_keep = collect(view(logps, 1:nkeep))
                x0 = copy(view(S_keep, :, nkeep))

                push!(blocks, S_keep)
                push!(logp_blocks, logps_keep)
                final_state = ParallelMALAState(
                    x0, logps_keep[nkeep], S_keep, logps_keep, ws, tape, nkeep
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

    if chain_type === MCMCChains.Chains
        return _sample_parallel_mala_chain(
            rng,
            model,
            sampler,
            N_int;
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

    S, tape, ws = _deer_solve_new_tape(rng, model, sampler, x0)
    logps = _trajectory_logps(model, S)
    x1 = S[:, 1]
    logp1 = logps[1]
    trans = ParallelMALATransition(x1, logp1)
    state = ParallelMALAState(x1, logp1, S, logps, ws, tape, 1)
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
        )
        return trans, new_state
    else
        x0 = state.trajectory[:, T]
        S_new, tape, ws = _deer_solve_new_tape(
            rng, model, sampler, x0; workspace=state.workspace
        )
        logps = _trajectory_logps(model, S_new)
        x_new = S_new[:, 1]
        logp_new = logps[1]
        trans = ParallelMALATransition(x_new, logp_new)
        new_state = ParallelMALAState(x_new, logp_new, S_new, logps, ws, tape, 1)
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
statistics.
"""
struct AdaptiveMALAState{V<:AbstractVector,FP<:AbstractFloat,W,NV<:AbstractVector,H}
    x::V
    logp::FP
    epsilon::FP
    epsilon_bar::FP
    H_bar::FP
    step::Int
    workspace::W
    noise::NV
    noise_host::H
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
    )
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
