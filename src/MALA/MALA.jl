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

"Compute the MALA proposal map y = x + ϵ∇logp(x) + sqrt(2ϵ) ξ."
function mala_proposal(logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector)
    @assert length(x) == length(ξ)
    g_x = gradlogp(x)
    return x .+ ϵ .* g_x .+ sqrt(2ϵ) .* ξ
end

"Compute log acceptance ratio logα(x→y) for MALA."
function mala_logα(logp, gradlogp, x::AbstractVector, y::AbstractVector, ϵ::Real)
    g_x = gradlogp(x)
    g_y = gradlogp(y)
    logp_x = logp(x)
    logp_y = logp(y)
    logq_y_given_x = logq_mala(y, x, g_x, ϵ)
    logq_x_given_y = logq_mala(x, y, g_y, ϵ)
    return (logp_y + logq_x_given_y) - (logp_x + logq_y_given_x)
end

"""
Primal accept indicator for a taped MALA step.
Returns Float64 in {0.0, 1.0} so it can be used as a constant gate.
"""
function mala_accept_indicator(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real
)
    y = mala_proposal(logp, gradlogp, x, ϵ, ξ)
    logα = mala_logα(logp, gradlogp, x, y, ϵ)
    return (log(u) < logα) ? 1.0 : 0.0
end

"""
Stop-gradient surrogate step used for Jacobians.
`a` (0.0 or 1.0) must be provided as a constant by the DEER machinery.
"""
function mala_step_surrogate(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, a::Real
)
    y = mala_proposal(logp, gradlogp, x, ϵ, ξ)
    return (a .* y) .+ ((1 - a) .* x)
end

end # module
