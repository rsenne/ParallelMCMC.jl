"""
    AdaptiveMALASampler(epsilon_init; n_warmup=1000,
                        target_accept=0.574, gamma=0.05,
                        t0=10.0, kappa=0.75, cholM=nothing)

Create a sequential MALA sampler with dual-averaging step-size adaptation.

Adaptation begins at the positive step size `epsilon_init`, targets the
acceptance probability `target_accept`, and stops after `n_warmup` warmup
transitions. The averaged step size is then held fixed. Set
`discard_warmup=true` in `sample` to omit warmup-marked states from the returned
chain.

`gamma`, `t0`, and `kappa` control the dual-averaging schedule. Their defaults
are usually a better starting point than manual adjustment. `cholM` has the
same preconditioning meaning as in [`MALASampler`](@ref).

The chain stores `:step_size`, `:accepted`, and `:is_warmup` as extra fields.
This sampler is a convenient way to choose an `epsilon` for
[`ParallelMALASampler`](@ref).
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
    V<:AbstractVector,
    FP<:AbstractFloat,
    G<:AbstractVector,
    W,
    NV<:AbstractVector,
    H,
    DM<:PreppedDensityModel,
}
    x::V
    logp::FP
    grad::G
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
    grad = copy(model.grad_logdensity(x))
    ws = MALA.MALAWorkspace(x)
    noise, noise_host = _make_noise_buffer(x, FP, model.dim)
    trans = AdaptiveMALATransition(x, logp_val, true, sampler.epsilon_init, true)
    state = AdaptiveMALAState(
        x,
        logp_val,
        grad,
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
    model, logp_x, grad_x = if _prepped_for(state.model, user_model)
        state.model, state.logp, state.grad
    else
        m = _prepare_model(user_model, state.x)
        m, FP(m.logdensity(state.x)), copy(m.grad_logdensity(state.x))
    end
    D = model.dim
    in_warmup = state.step < sampler.n_warmup
    ε = in_warmup ? state.epsilon : state.epsilon_bar

    ξ = _randn_like!(rng, state.noise, state.noise_host)
    u = rand(rng)

    x_next = similar(state.x)
    x_next, accepted, logα, logp_next, grad_next = MALA.mala_step_cached!(
        x_next,
        state.workspace,
        model.logdensity,
        model.grad_logdensity,
        state.x,
        logp_x,
        grad_x,
        ε,
        ξ,
        u;
        cholM=sampler.cholM,
    )
    logp_next = FP(logp_next)

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
        grad_next,
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
