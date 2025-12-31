module BayesLinReg

using LinearAlgebra

"""
Bayesian linear regression with:
  y | β ~ N(Xβ, σ² I)
  β ~ N(0, τ² I)

Returns closures (logpost, gradlogpost) for β::Vector.
Also returns the analytic posterior mean/cov for verification.
"""
function make_problem(X::AbstractMatrix, y::AbstractVector; σ::Real=1.0, τ::Real=10.0)
    n, p = size(X)
    @assert length(y) == n

    σ2 = float(σ)^2
    τ2 = float(τ)^2

    # log posterior up to constant
    function logpost(β::AbstractVector)
        r = y .- X*β
        ll = -0.5/σ2 * dot(r, r)
        lp = -0.5/τ2 * dot(β, β)
        return ll + lp
    end

    # gradient of log posterior
    function gradlogpost(β::AbstractVector)
        r = y .- X*β
        # ∇β ll = (1/σ²) Xᵀ (y - Xβ)
        g = (X' * r) ./ σ2
        # ∇β lp = -(1/τ²) β
        g .-= β ./ τ2
        return g
    end

    # Analytic Gaussian posterior:
    # Σ_post = (XᵀX/σ² + I/τ²)^{-1}
    # μ_post = Σ_post * (Xᵀy/σ²)
    A = (X' * X) ./ σ2
    A += Diagonal(fill(1.0 / τ2, p))
    Σ_post = inv(Symmetric(A))
    μ_post = Σ_post * (X' * y) ./ σ2

    return logpost, gradlogpost, μ_post, Σ_post
end

end # module
