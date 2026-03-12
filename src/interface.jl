#=
AbstractMCMC interface for ParallelMCMC samplers.

Defines model/sampler/state/transition types and implements
`AbstractMCMC.step` so that `sample(model, sampler, N)` works out of the box.
=#

"""
    DensityModel(logdensity, grad_logdensity, dim)

Wraps a log-density function and its gradient for use with ParallelMCMC samplers.

- `logdensity(x::AbstractVector) -> Real`
- `grad_logdensity(x::AbstractVector) -> AbstractVector`
- `dim::Int` — dimensionality of the parameter space
"""
struct DensityModel{F,G} <: AbstractMCMC.AbstractModel
    logdensity::F
    grad_logdensity::G
    dim::Int
end

"""
    MALASampler(epsilon)

Metropolis-Adjusted Langevin Algorithm sampler with step size `epsilon`.
"""
struct MALASampler <: AbstractMCMC.AbstractSampler
    epsilon::Float64
end

struct MALAState{V<:AbstractVector}
    x::V
    logp::Float64
end

struct MALATransition{V<:AbstractVector}
    x::V
    logp::Float64
    accepted::Bool
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::MALASampler;
    initial_params=nothing,
    kwargs...,
)
    x = if initial_params !== nothing
        copy(initial_params)
    else
        randn(rng, model.dim)
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

    ξ = randn(rng, D)
    u = rand(rng)

    accepted = MALA.mala_accept_indicator(
        model.logdensity, model.grad_logdensity, x, ϵ, ξ, u
    ) == 1.0

    x_next = MALA.mala_step_taped(
        model.logdensity, model.grad_logdensity, x, ϵ, ξ, u
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
