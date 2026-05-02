module BayesLogReg

using Random
using LinearAlgebra

function _sum_columns!(out::AbstractVector, row::AbstractMatrix, A::AbstractMatrix)
    sum!(row, A)
    copyto!(out, row)
    return out
end

"""
    make_data(rng, N, D; β_scale=1.0) -> (X, y, β_true)

Generate synthetic Bayesian logistic regression data.
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

Optimized to reuse memory buffers for GPU performance.
"""
function make_problem(X::AbstractMatrix, y::AbstractVector)
    # Pre-allocate workspaces in the closure scope
    N = size(X, 1)
    logits = similar(X, N)
    p = similar(X, N)
    resid = similar(X, N)

    function logp(β::AbstractVector)
        mul!(logits, X, β)
        # ll = sum(@. y * (-log1p(exp(-logits))) + (1 - y) * (-log1p(exp(logits))))
        # We perform this in-place to avoid allocations
        # Note: log1p(exp(x)) is the softplus function
        @. p = y * (-log1p(exp(-logits))) + (1 - y) * (-log1p(exp(logits)))
        return sum(p) - 0.5 * sum(abs2, β)
    end

    function gradlogp(β::AbstractVector)
        mul!(logits, X, β)
        @. p = 1 / (1 + exp(-logits))
        @. resid = y - p
        # grad = X' * (y - p) - β
        # We reuse the output allocation here by using mul!
        grad = (X' * resid) .- β
        return grad
    end

    return logp, gradlogp
end

"""
    make_problem_with_hvp(X, y) -> (logp, gradlogp, hvp)
"""
function make_problem_with_hvp(X::AbstractMatrix, y::AbstractVector)
    logp, gradlogp = make_problem(X, y)

    # Workspace buffers
    N = size(X, 1)
    logits = similar(X, N)
    p = similar(X, N)
    w = similar(X, N)
    Xv = similar(X, N)

    function hvp(β::AbstractVector, v::AbstractVector)
        mul!(logits, X, β)
        @. p = 1 / (1 + exp(-logits))
        @. w = p * (1 - p)
        mul!(Xv, X, v)
        @. Xv = w * Xv
        return -(X' * Xv) .- v
    end

    return logp, gradlogp, hvp
end

"""
    make_problem_batched(X, y) -> (logp_batch, gradlogp_batch)
"""
function make_problem_batched(X::AbstractMatrix, y::AbstractVector)
    N, D = size(X)
    y_col = reshape(y, N, 1)

    logits = Ref(similar(X, N, 1))
    res = Ref(similar(X, N, 1))
    sq = Ref(similar(X, D, 1))
    grad = Ref(similar(X, D, 1))
    out = Ref(similar(X, eltype(X), 1))
    quad = Ref(similar(X, eltype(X), 1))
    out_row = Ref(similar(X, eltype(X), 1, 1))
    quad_row = Ref(similar(X, eltype(X), 1, 1))

    function ensure_workspace!(M::Int)
        if size(logits[], 2) != M
            logits[] = similar(X, N, M)
            res[] = similar(X, N, M)
            sq[] = similar(X, D, M)
            grad[] = similar(X, D, M)
            out[] = similar(X, eltype(X), M)
            quad[] = similar(X, eltype(X), M)
            out_row[] = similar(X, eltype(X), 1, M)
            quad_row[] = similar(X, eltype(X), 1, M)
        end
        return nothing
    end

    function logp_batch(B::AbstractMatrix)
        M = size(B, 2)
        size(B, 1) == D || throw(DimensionMismatch("B must have size (D, M)"))
        ensure_workspace!(M)

        logits_buf = logits[]
        res_buf = res[]
        sq_buf = sq[]
        out_buf = out[]
        quad_buf = quad[]

        mul!(logits_buf, X, B)
        @. res_buf =
            y_col * (-log1p(exp(-logits_buf))) + (1 - y_col) * (-log1p(exp(logits_buf)))
        @. sq_buf = abs2(B)
        _sum_columns!(out_buf, out_row[], res_buf)
        _sum_columns!(quad_buf, quad_row[], sq_buf)
        @. out_buf = out_buf - 0.5 * quad_buf
        return out_buf
    end

    function gradlogp_batch(B::AbstractMatrix)
        M = size(B, 2)
        size(B, 1) == D || throw(DimensionMismatch("B must have size (D, M)"))
        ensure_workspace!(M)

        logits_buf = logits[]
        res_buf = res[]
        grad_buf = grad[]

        mul!(logits_buf, X, B)
        @. logits_buf = 1 / (1 + exp(-logits_buf))
        @. res_buf = y_col - logits_buf
        mul!(grad_buf, adjoint(X), res_buf)
        @. grad_buf = grad_buf - B
        return grad_buf
    end

    return logp_batch, gradlogp_batch
end

"""
    make_problem_batched_with_hvp(X, y) -> (logp_batch, gradlogp_batch, hvp_batch)
"""
function make_problem_batched_with_hvp(X::AbstractMatrix, y::AbstractVector)
    logp_batch, gradlogp_batch = make_problem_batched(X, y)
    N, D = size(X)

    logits = Ref(similar(X, N, 1))
    w = Ref(similar(X, N, 1))
    Xv = Ref(similar(X, N, 1))
    out = Ref(similar(X, D, 1))

    function ensure_hvp_workspace!(M::Int)
        if size(logits[], 2) != M
            logits[] = similar(X, N, M)
            w[] = similar(X, N, M)
            Xv[] = similar(X, N, M)
            out[] = similar(X, D, M)
        end
        return nothing
    end

    function hvp_batch(B::AbstractMatrix, V::AbstractMatrix)
        size(B) == size(V) || throw(DimensionMismatch("B and V must have the same size"))
        M = size(B, 2)
        size(B, 1) == D || throw(DimensionMismatch("B and V must have size (D, M)"))
        ensure_hvp_workspace!(M)

        logits_buf = logits[]
        w_buf = w[]
        Xv_buf = Xv[]
        out_buf = out[]

        mul!(logits_buf, X, B)

        @. logits_buf = 1 / (1 + exp(-logits_buf))
        @. w_buf = logits_buf * (1 - logits_buf)

        mul!(Xv_buf, X, V)
        @. Xv_buf = w_buf * Xv_buf

        mul!(out_buf, adjoint(X), Xv_buf)
        @. out_buf = -out_buf - V
        return out_buf
    end

    return logp_batch, gradlogp_batch, hvp_batch
end

end # module
