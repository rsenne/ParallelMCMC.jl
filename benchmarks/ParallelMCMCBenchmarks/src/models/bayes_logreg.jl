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
    make_problem_batched(X, y) -> (logp_batch, gradlogp_batch)

Batched counterparts of `make_problem` that accept D×M matrices and return M-vectors
or D×M matrices.  Used by `ParallelMALASampler` GPU path for efficient matrix-level
parallelism across trajectory steps.

- `logp_batch(B::AbstractMatrix)` — `B` is D×M; returns M-vector of log-posteriors.
- `gradlogp_batch(B::AbstractMatrix)` — returns D×M gradient matrix.

Both closures are GPU-compatible: if X and y are CuArrays, all operations
(`X*B`, broadcasts, sums) dispatch to CUDA kernels.
"""
function make_problem_batched(X::AbstractMatrix, y::AbstractVector)
    function logp_batch(B::AbstractMatrix)
        logits = X * B                                              # N×M
        ll_mat = @. y * (-log1p(exp(-logits))) +
                    (one(eltype(logits)) - y) * (-log1p(exp(logits)))  # N×M
        ll = vec(sum(ll_mat; dims=1))                              # M
        return ll .- oftype(eltype(ll), 0.5) .* vec(sum(abs2, B; dims=1))
    end

    function gradlogp_batch(B::AbstractMatrix)
        logits = X * B                                              # N×M
        p = @. one(eltype(logits)) / (one(eltype(logits)) + exp(-logits))  # N×M
        return X' * (y .- p) .- B                                  # D×M
    end

    return logp_batch, gradlogp_batch
end

end # module
