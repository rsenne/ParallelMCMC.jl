module BayesLogReg

using Random
using LinearAlgebra

"""
    make_data(rng, N, D; β_scale=1.0) -> (X, y, β_true)

Generate synthetic Bayesian logistic regression data:
  β_true ~ N(0, β_scale² I_D)
  X_i    ~ N(0, I_D)
  y_i    ~ Bernoulli(sigmoid(X_i β_true))

Returns Float64 arrays.
"""
function make_data(rng::AbstractRNG, N::Int, D::Int; β_scale::Real=1.0)
    β_true = randn(rng, D) .* float(β_scale)
    X = randn(rng, N, D)
    logits = X * β_true
    y = Float64[rand(rng) < 1 / (1 + exp(-l)) ? 1.0 : 0.0 for l in logits]
    return X, y, β_true
end

"""
    make_problem(X, y) -> (logp, gradlogp)

Return the log-posterior and its gradient for:
  β ~ N(0, I_D)
  y_i | β ~ Bernoulli(sigmoid(Xᵢβ))

Both closures accept an AbstractVector and work with any eltype, so they
are compatible with ForwardDiff dual numbers and GPU arrays alike.
"""
function make_problem(X::AbstractMatrix, y::AbstractVector)
    function logp(β::AbstractVector)
        logits = X * β
        ll = sum(@. y * (-log1p(exp(-logits))) + (1 - y) * (-log1p(exp(logits))))
        return ll - oftype(ll, 0.5) * sum(abs2, β)
    end

    function gradlogp(β::AbstractVector)
        logits = X * β
        p = @. 1 / (1 + exp(-logits))
        return X' * (y .- p) .- β
    end

    return logp, gradlogp
end

"""
    make_problem_with_hvp(X, y) -> (logp, gradlogp, hvp)

Return the log-posterior, its gradient, and an analytic Hessian-vector product
for Bayesian logistic regression.
"""
function make_problem_with_hvp(X::AbstractMatrix, y::AbstractVector)
    logp, gradlogp = make_problem(X, y)

    function hvp(β::AbstractVector, v::AbstractVector)
        logits = X * β
        T = eltype(logits)
        oneT = one(T)
        p = oneT ./ (oneT .+ exp.(-logits))
        w = p .* (oneT .- p)
        Xv = X * v
        tmp = w .* Xv
        Xt_tmp = X' * tmp
        return Xt_tmp .* (-oneT) .- v
    end

    return logp, gradlogp, hvp
end

"""
    make_problem_batched(X, y) -> (logp_batch, gradlogp_batch)

Batched counterparts of `make_problem` that accept D×M matrices and return M-vectors
or D×M matrices.
"""
function make_problem_batched(X::AbstractMatrix, y::AbstractVector)
    function logp_batch(B::AbstractMatrix)
        logits = X * B
        T = eltype(logits)
        oneT = one(T)
        halfT = T(0.5)

        neg_logits = logits .* (-oneT)
        ll_yes = y .* (-log1p.(exp.(neg_logits)))
        ll_no = (oneT .- y) .* (-log1p.(exp.(logits)))
        ll_mat = ll_yes .+ ll_no
        ll = vec(sum(ll_mat; dims=1))
        quad = vec(sum(abs2, B; dims=1))
        return ll .- halfT .* quad
    end

    function gradlogp_batch(B::AbstractMatrix)
        logits = X * B
        T = eltype(logits)
        oneT = one(T)
        p = oneT ./ (oneT .+ exp.(logits .* (-oneT)))
        resid = y .- p
        return X' * resid .- B
    end

    return logp_batch, gradlogp_batch
end

"""
    make_problem_batched_with_hvp(X, y) -> (logp_batch, gradlogp_batch, hvp_batch)

Batched log-posterior, gradient, and analytic Hessian-vector product for
Bayesian logistic regression. Inputs `B` and `V` are both D×M matrices.
"""
function make_problem_batched_with_hvp(X::AbstractMatrix, y::AbstractVector)
    logp_batch, gradlogp_batch = make_problem_batched(X, y)

    function hvp_batch(B::AbstractMatrix, V::AbstractMatrix)
        size(B) == size(V) || throw(DimensionMismatch("B and V must have the same size"))
        logits = X * B
        T = eltype(logits)
        oneT = one(T)
        p = oneT ./ (oneT .+ exp.(logits .* (-oneT)))
        w = p .* (oneT .- p)
        Xv = X * V
        tmp = w .* Xv
        return -(X' * tmp) .- V
    end

    return logp_batch, gradlogp_batch, hvp_batch
end

end # module
