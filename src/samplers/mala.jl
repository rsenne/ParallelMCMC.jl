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
happens once per chain, and the log-density and gradient at `x` so each step
evaluates them only at the proposal.
"""
struct MALAState{
    V<:AbstractVector,
    L<:Real,
    G<:AbstractVector,
    W,
    NV<:AbstractVector,
    H,
    DM<:PreppedDensityModel,
}
    x::V
    logp::L
    grad::G
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
    grad = copy(model.grad_logdensity(x))
    ws = MALA.MALAWorkspace(x)
    noise, noise_host = _make_noise_buffer(x, FP, model.dim)
    t = MALATransition(x, logp_val, true)
    s = MALAState(x, logp_val, grad, ws, noise, noise_host, model)
    return t, s
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    user_model::DensityModel,
    sampler::MALASampler,
    state::MALAState;
    kwargs...,
)
    #= Reuse the state's preparation and cached log-density and gradient, unless
    the state came from a different model (via `initial_state`), in which case
    the model we were handed is the one to sample. =#
    x = state.x
    model, logp_x, grad_x = if _prepped_for(state.model, user_model)
        state.model, state.logp, state.grad
    else
        m = _prepare_model(user_model, x)
        m, m.logdensity(x), copy(m.grad_logdensity(x))
    end
    ϵ = sampler.epsilon
    D = model.dim

    ξ = _randn_like!(rng, state.noise, state.noise_host)
    u = rand(rng)

    x_next = similar(x)
    x_next, accepted, _, logp_val, grad_next = MALA.mala_step_cached!(
        x_next,
        state.workspace,
        model.logdensity,
        model.grad_logdensity,
        x,
        logp_x,
        grad_x,
        ϵ,
        ξ,
        u;
        cholM=sampler.cholM,
    )

    t = MALATransition(x_next, logp_val, accepted)
    s = MALAState(
        x_next, logp_val, grad_next, state.workspace, state.noise, state.noise_host, model
    )
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
