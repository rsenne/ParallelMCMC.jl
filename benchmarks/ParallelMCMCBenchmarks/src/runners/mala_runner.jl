module MALARunner

using Random
using Statistics
using ParallelMCMC
const MALA = ParallelMCMC.MALA

"""
Generate a deterministic tape for MALA:
  ξ_t ~ N(0, I)
  u_t ~ Uniform(0,1)
"""
function make_tape(rng::AbstractRNG, D::Int, T::Int)
    ξs = [randn(rng, D) for _ in 1:T]
    us = rand(rng, T)
    return ξs, us
end

"""
Very simple step-size adaptation for MALA (Robbins–Monro on log ϵ).

- Runs `Twarm` steps (taped) starting from x0, adjusting ϵ to hit target_accept.
- Returns (x_warm, ϵ_tuned).

This is not a full NUTS-style adaptation; it is a lightweight, effective baseline.
"""
function tune_stepsize_mala(
    logp, gradlogp,
    x0::Vector{Float64},
    ϵ0::Float64;
    Twarm::Int = 2_000,
    target_accept::Float64 = 0.57,
    rng::AbstractRNG = Random.default_rng(),
)
    D = length(x0)

    # Pre-generate tape for warmup
    ξs, us = make_tape(rng, D, Twarm)

    x = copy(x0)
    logϵ = log(ϵ0)

    # Step size schedule; keep conservative to avoid oscillation
    η0 = 0.05

    for t in 1:Twarm
        ϵ = exp(logϵ)
        a = MALA.mala_accept_indicator(logp, gradlogp, x, ϵ, ξs[t], us[t])  # 0.0 or 1.0
        # Robbins–Monro update on log ϵ
        η = η0 / sqrt(t)
        logϵ += η * (a - target_accept)

        x = MALA.mala_step_taped(logp, gradlogp, x, ϵ, ξs[t], us[t])
    end

    return x, exp(logϵ)
end

"""
Run a sequential taped MALA chain and record acceptance indicators.

Returns:
  xs::Vector{Vector{Float64}}  length T+1 (includes x0)
  accepts::Vector{Float64}     length T, entries 0.0 or 1.0
"""
function run_taped_mala_with_accepts(logp, gradlogp, x0::Vector{Float64}, ϵ::Float64,
                                    ξs::Vector{Vector{Float64}}, us::Vector{Float64})
    T = length(us)
    @assert length(ξs) == T

    xs = Vector{Vector{Float64}}(undef, T+1)
    accepts = Vector{Float64}(undef, T)

    xs[1] = copy(x0)
    x = x0
    for t in 1:T
        a = MALA.mala_accept_indicator(logp, gradlogp, x, ϵ, ξs[t], us[t])
        accepts[t] = a
        x = MALA.mala_step_taped(logp, gradlogp, x, ϵ, ξs[t], us[t])
        xs[t+1] = x
    end

    return xs, accepts
end

accept_rate(accepts::AbstractVector) = mean(accepts)

end # module
