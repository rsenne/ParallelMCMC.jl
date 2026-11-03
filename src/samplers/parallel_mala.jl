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
    ParallelMALASampler(epsilon; T, maxiter, tol_abs, tol_rel, jacobian, damping, probes, jacobian_damping, jacobian_clip, cholM, backend)

DEER-parallelized MALA sampler.

Supported Jacobian modes are `:stoch_diag` (the default Hutchinson diagonal
estimator) and `:diag` (exact diagonal via `D` JVPs).

`jacobian_damping` and `jacobian_clip` change the Jacobian estimate to
`clamp(jacobian_damping * J, -jacobian_clip, jacobian_clip)`, which stabilises
DEER for multimodal targets or large step sizes; see [`DEER.solve`](@ref). They
change how the solver reaches the samples, not the samples. The defaults (`1` and
`1`) match the `ParallelMALA` sampler in the reference implementation: no scaling,
and diagonal entries clipped to `[-1, 1]`.

`damping` is deprecated; see [`DEER.solve`](@ref).

`backend` supplies Hessian-vector products when the `DensityModel` brings no
`hvp` / `hvp_batch` of its own. It does not supply gradients. Leave it out when
the model carries its own HVPs. When it is left out and `grad_logdensity` is an
AD backend, the HVP runs the forward-mode counterpart of that backend over the
gradient (`DifferentiationInterface.forward_counterpart`).

With `backend=ADTypes.AutoReactant()`, `grad_logdensity` must be `AutoReactant()`
or a callable.

`callback` and `initial_state` route through the generic step-wise loop instead.
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
    jacobian_damping::FP
    jacobian_clip::FP
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
    damping::Real=1,
    probes::Int=1,
    jacobian_damping::Real=1,
    jacobian_clip::Real=1,
    cholM=nothing,
    backend=nothing,
)
    epsilon > 0 || throw(ArgumentError("epsilon must be > 0, got $epsilon"))
    (jacobian === :stoch_diag || jacobian === :diag) ||
        throw(ArgumentError("jacobian must be :stoch_diag or :diag"))
    DEER._check_jacobian_modifiers(jacobian_damping, jacobian_clip)
    DEER._depwarn_damping(damping, :ParallelMALASampler)
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
        FP(jacobian_damping),
        FP(jacobian_clip),
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

#= Kept samples of a chain, one block of columns per DEER solve. `ends[b]` is the
number of kept samples up to and including block `b`; with thinning the blocks
are not all the same width. =#
struct ParallelMALABlockSamples{B<:AbstractVector,L<:AbstractVector} <:
       AbstractVector{ParallelMALATransition}
    blocks::B
    logps::L
    ends::Vector{Int}
    n::Int
end

Base.size(samples::ParallelMALABlockSamples) = (samples.n,)
Base.length(samples::ParallelMALABlockSamples) = samples.n
Base.IndexStyle(::Type{<:ParallelMALABlockSamples}) = IndexLinear()

function Base.getindex(samples::ParallelMALABlockSamples, i::Int)
    1 <= i <= samples.n || throw(BoundsError(samples, i))
    b = searchsortedfirst(samples.ends, i)
    t = i - (b == 1 ? 0 : samples.ends[b - 1])
    return ParallelMALATransition(samples.blocks[b][:, t], samples.logps[b][t])
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
        jacobian_damping=sampler.jacobian_damping,
        jacobian_clip=sampler.jacobian_clip,
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

"""
Which steps of a `ParallelMALASampler` run are kept. Global step `g` is kept iff
`g > discard_initial` and `(g - discard_initial - 1) % thinning == 0`, i.e., the
steps the `AbstractMCMC.mcmcsample` loop saves, so the batched path and the
step-wise fallback return the same chain for the same RNG.
"""
struct ParallelMALASchedule
    discard_initial::Int
    thinning::Int
    Ntotal::Int
end

function ParallelMALASchedule(
    N::Int; discard_initial::Int=0, thinning::Int=1, num_warmup::Int=0
)
    N > 0 || error("the number of samples must be ≥ 1")
    discard_initial >= 0 ||
        throw(ArgumentError("number of discarded samples must be non-negative"))
    thinning >= 1 || throw(ArgumentError("thinning must be ≥ 1"))
    num_warmup >= 0 ||
        throw(ArgumentError("number of warm-up samples must be non-negative"))
    Ntotal = thinning * (N - 1) + discard_initial + 1
    Ntotal >= num_warmup || throw(
        ArgumentError("number of warm-up samples exceeds the total number of samples")
    )
    return ParallelMALASchedule(discard_initial, thinning, Ntotal)
end

#= Columns of a block covering global steps `offset+1:offset+ncols` that the schedule
keeps, and the chain row the first of them lands in. =#
function _kept_columns(sched::ParallelMALASchedule, offset::Int, ncols::Int)
    k0 = max(0, cld(offset - sched.discard_initial, sched.thinning))
    first_col = sched.discard_initial + 1 + k0 * sched.thinning - offset
    return first_col:(sched.thinning):ncols, k0 + 1
end

#= Runs DEER blocks until `sched.Ntotal` steps are consumed, passing kept columns
to `sink!(S, logps, cols, first_row)`. Uses full-length blocks (`T`) to match the
fallback's RNG stream. Returns the last block; `S` aliases the workspace. =#
function _parallel_mala_run!(
    sink!,
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::ParallelMALASampler,
    sched::ParallelMALASchedule;
    initial_params=nothing,
    progress=AbstractMCMC.PROGRESS[],
    progressname="Sampling",
)
    progress = _parallel_mala_progress(progress, progressname)
    x0 = _parallel_mala_initial_x(rng, model, sampler, initial_params)
    prepped = _prepare_model(model, x0, sampler.T, sampler.backend)
    ws = nothing
    nsteps = 0
    Ntotal = sched.Ntotal
    next_update = Ntotal / AbstractMCMC.get_n_updates(progress)
    threshold = next_update
    last = nothing

    AbstractMCMC.@maybewithricherlogger begin
        AbstractMCMC.init_progress!(progress)
        try
            while nsteps < Ntotal
                S, tape, ws = _deer_solve_new_tape(rng, prepped, sampler, x0; workspace=ws)
                logps = _trajectory_logps(prepped, S)
                nconsume = min(sampler.T, Ntotal - nsteps)
                cols, first_row = _kept_columns(sched, nsteps, nconsume)
                sink!(S, logps, cols, first_row)

                x0 = copy(view(S, :, nconsume))
                nsteps += nconsume
                last = (
                    S=S, logps=logps, workspace=ws, tape=tape, t=nconsume, model=prepped
                )
                next_update = _parallel_mala_update_progress!(
                    progress, nsteps, Ntotal, next_update, threshold
                )
            end
        finally
            AbstractMCMC.finish_progress!(progress)
        end
    end
    return last
end

function _copy_trajectory_rows!(
    vals::AbstractMatrix{<:Real}, first_row::Int, S::AbstractMatrix, cols::AbstractRange
)
    rows = first_row:(first_row + length(cols) - 1)
    S_host = Array(S[:, cols])
    vals[rows, :] .= transpose(S_host)
    return vals
end

function _sample_parallel_mala_chain(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::ParallelMALASampler,
    N::Int,
    ::Type{FlexiChains.FlexiChain{TKey}},
    sched::ParallelMALASchedule;
    initial_params=nothing,
    param_names=nothing,
    progress=AbstractMCMC.PROGRESS[],
    progressname="Sampling",
) where {TKey}
    D = model.dim
    FP = typeof(sampler.epsilon)

    vals = Matrix{FP}(undef, N, D)
    logp = Vector{FP}(undef, N)

    _parallel_mala_run!(
        rng, model, sampler, sched; initial_params, progress, progressname
    ) do S, logps, cols, first_row
        _copy_trajectory_rows!(vals, first_row, S, cols)
        logp[first_row:(first_row + length(cols) - 1)] .= view(logps, cols)
        return nothing
    end

    internals = (logp=logp,)
    return _construct_flexichain(TKey, vals, internals, param_names, model)
end

function _sample_parallel_mala_blocks(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::ParallelMALASampler,
    N::Int,
    sched::ParallelMALASchedule;
    initial_params=nothing,
    progress=AbstractMCMC.PROGRESS[],
    progressname="Sampling",
)
    nblocks = cld(sched.Ntotal, sampler.T)
    blocks = Vector{AbstractMatrix}(undef, 0)
    logp_blocks = Vector{AbstractVector}(undef, 0)
    ends = Int[]
    sizehint!(blocks, nblocks)
    sizehint!(logp_blocks, nblocks)
    sizehint!(ends, nblocks)

    last = _parallel_mala_run!(
        rng, model, sampler, sched; initial_params, progress, progressname
    ) do S, logps, cols, first_row
        isempty(cols) && return nothing
        push!(blocks, S[:, cols])
        push!(logp_blocks, logps[cols])
        push!(ends, first_row + length(cols) - 1)
        return nothing
    end

    S = copy(last.S)
    final_state = ParallelMALAState(
        S[:, last.t],
        last.logps[last.t],
        S,
        last.logps,
        last.workspace,
        last.tape,
        last.t,
        last.model,
    )
    return ParallelMALABlockSamples(blocks, logp_blocks, ends, N), final_state
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
    #= `callback` wants a transition per step and `initial_state` a trajectory to
    replay, so both go through the generic step loop. =#
    if callback !== nothing || initial_state !== nothing
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

    N_int = Int(N)
    sched = ParallelMALASchedule(N_int; discard_initial, thinning, num_warmup)

    if chain_type <: FlexiChains.FlexiChain
        return _sample_parallel_mala_chain(
            rng,
            model,
            sampler,
            N_int,
            chain_type,
            sched;
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
        N_int,
        sched;
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
