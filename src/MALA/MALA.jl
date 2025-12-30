module MALA

using Random, LinearAlgebra

"""
Compute the log of the MALA proposal density q(y | x).
We use the Gaussian:
  y ~ Normal(x + ϵ∇logp(x), 2ϵ I)
"""
function logq_mala(
    y::AbstractVector, x::AbstractVector, gradlogp_x::AbstractVector, ϵ::Real
)
    μ = x .+ ϵ .* gradlogp_x
    # log N(y; μ, 2ϵ I) up to constant:
    # -0.5 * ||y-μ||^2 / (2ϵ) - (d/2) log(4πϵ)
    d = length(x)
    r = y .- μ
    return -0.5 * dot(r, r) / (2ϵ) - (d / 2) * log(4π * ϵ)
end

"""
One taped MALA step.

Inputs:
- logp(x): log density (up to constant)
- gradlogp(x): gradient of log density
- x: current state
- ϵ: step size
- ξ: N(0, I) noise vector (tape)
- u: Uniform(0,1) scalar (tape)

Returns:
- x_next
"""
function mala_step_taped(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real
)
    @assert length(x) == length(ξ)
    @assert 0.0 < u < 1.0
    g_x = gradlogp(x)

    # Proposal
    y = x .+ ϵ .* g_x .+ sqrt(2ϵ) .* ξ

    # Compute log acceptance ratio:
    # log α = logp(y) + log q(x|y) - logp(x) - log q(y|x)
    logp_x = logp(x)
    logp_y = logp(y)

    g_y = gradlogp(y)
    logq_y_given_x = logq_mala(y, x, g_x, ϵ)
    logq_x_given_y = logq_mala(x, y, g_y, ϵ)

    logα = (logp_y + logq_x_given_y) - (logp_x + logq_y_given_x)

    # Accept/reject using tape u
    return (log(u) < logα) ? y : x
end

"""
Run sequential taped MALA for T steps.

Inputs:
- x0: initial state
- ξs: vector of ξ_t noise vectors, length T
- us: vector of u_t uniforms, length T

Returns:
- xs: Vector of states length T+1, xs[1]=x0, xs[t+1]=x_t
"""
function run_mala_sequential_taped(
    logp,
    gradlogp,
    x0::AbstractVector,
    ϵ::Real,
    ξs::Vector{<:AbstractVector},
    us::AbstractVector,
)
    T = length(us)
    @assert length(ξs) == T
    xs = Vector{typeof(x0)}(undef, T + 1)
    xs[1] = copy(x0)
    x = copy(x0)
    for t in 1:T
        x = mala_step_taped(logp, gradlogp, x, ϵ, ξs[t], us[t])
        xs[t + 1] = copy(x)
    end
    return xs
end

end # module
