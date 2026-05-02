module MALA

using Random, LinearAlgebra

export MALAWorkspace,
    MALABatchedWorkspace,
    logq_mala,
    logq_mala!,
    mala_step_taped,
    mala_step_taped!,
    run_mala_sequential_taped,
    mala_proposal,
    mala_proposal!,
    mala_logα,
    mala_logα!,
    mala_accept_indicator,
    mala_step_full,
    mala_step_with_logα,
    mala_step_with_logα!,
    mala_step_surrogate_sigmoid,
    mala_step_surrogate_sigmoid_jvp,
    mala_step_taped_and_jvp,
    mala_step_taped_and_jvp!,
    mala_step_batched,
    mala_step_batched!,
    mala_step_batched_fwd_and_jvp,
    mala_step_batched_fwd_and_jvp!

# Preconditioner dispatch helpers. cholM is either nothing (identity) or a Cholesky factor.
_apply_M!(out, g, ::Nothing, tmp=out) = copyto!(out, g)
function _apply_M!(out, g, cholM::Cholesky, tmp)
    mul!(tmp, adjoint(cholM.L), g)
    mul!(out, cholM.L, tmp)
    return out
end

_apply_L!(out, ξ, ::Nothing) = (out .= ξ)
_apply_L!(out, ξ, cholM::Cholesky) = mul!(out, cholM.L, ξ)

_quad_Minv!(tmp, r, ::Nothing) = dot(r, r)
function _quad_Minv!(tmp, r, cholM::Cholesky)
    ldiv!(tmp, cholM.L, r)
    return dot(tmp, tmp)
end

_logdet_M(::Nothing) = false  # Bool promotes to any numeric type without widening
_logdet_M(cholM::Cholesky) = logdet(cholM)

"""
Reusable scratch buffers for allocation-light MALA primitives.

All vectors are created with the same array type / device placement as `x_template`,
so this workspace is GPU-compatible when constructed from a `CuVector`.
"""
struct MALAWorkspace{V}
    g_x::V
    g_y::V
    y::V
    μ::V
    r::V
    Hv_x::V
    Hv_y::V
    w::V
    dr::V
    jvp_out::V
    solve_buf::V
end

function MALAWorkspace(x::AbstractVector)
    return MALAWorkspace(
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
    )
end

"""
Reusable scratch buffers for batched MALA primitives.

Matrix buffers have the same size, eltype, and device placement as `X_template`.
Vector buffers have length `size(X_template, 2)`.
"""
struct MALABatchedWorkspace{M,V,RM,BV,BM}
    G_X::M
    G_Y::M
    Y::M
    MG_X::M
    MG_Y::M
    LΞ::M
    R::M
    dR::M
    Hv_X::M
    Hv_Y::M
    W::M
    M_Hv_X::M
    M_Hv_Y::M
    Minv_R::M
    solve_tmp::M
    prod::M
    lp_X::V
    lp_Y::V
    lq_YX::V
    lq_XY::V
    logα::V
    logu::V
    g::V
    dlogα::V
    dg::V
    dot1::V
    dot2::V
    dot3::V
    row::RM
    accepted::BV
    accepted_row::BM
end

function MALABatchedWorkspace(X::AbstractMatrix)
    N = size(X, 2)
    mat() = similar(X)
    vec() = similar(X, eltype(X), N)
    row() = similar(X, eltype(X), 1, N)
    accepted = similar(X, Bool, N)
    accepted_row = similar(X, Bool, 1, N)
    return MALABatchedWorkspace(
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        mat(),
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        vec(),
        row(),
        accepted,
        accepted_row,
    )
end

"""
    logq_mala!(ws, y, x, gradlogp_x, ε; cholM=nothing)

Allocation-light log proposal density using `ws.μ` and `ws.r` scratch.
"""
function logq_mala!(
    ws::MALAWorkspace,
    y::AbstractVector,
    x::AbstractVector,
    gradlogp_x::AbstractVector,
    ϵ::Real;
    cholM=nothing,
)
    T = typeof(ϵ)
    d = length(x)

    if cholM === nothing
        @. ws.μ = x + ϵ * gradlogp_x
    else
        _apply_M!(ws.w, gradlogp_x, cholM, ws.solve_buf)
        @. ws.μ = x + ϵ * ws.w
    end
    @. ws.r = y - ws.μ

    return -T(0.5) * _quad_Minv!(ws.solve_buf, ws.r, cholM) / (2ϵ) -
           (T(d) / 2) * log(T(4π) * ϵ) - T(0.5) * _logdet_M(cholM)
end

function logq_mala(
    y::AbstractVector, x::AbstractVector, gradlogp_x::AbstractVector, ϵ::Real; cholM=nothing
)
    ws = MALAWorkspace(x)
    return logq_mala!(ws, y, x, gradlogp_x, ϵ; cholM=cholM)
end

"""
    mala_proposal!(y, ws, logp, gradlogp, x, ε, ξ; cholM=nothing)

Write proposal into `y`, and gradient at `x` into `ws.g_x`.
"""
function mala_proposal!(
    y::AbstractVector,
    ws::MALAWorkspace,
    logp,
    gradlogp,
    x::AbstractVector,
    ϵ::Real,
    ξ::AbstractVector;
    cholM=nothing,
)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    copyto!(ws.g_x, gradlogp(x))
    sqrt2ϵ = sqrt(2 * ϵ)

    if cholM === nothing
        @. y = x + ϵ * ws.g_x + sqrt2ϵ * ξ
    else
        _apply_M!(ws.μ, ws.g_x, cholM, ws.solve_buf)
        _apply_L!(ws.r, ξ, cholM)
        @. y = x + ϵ * ws.μ + sqrt2ϵ * ws.r
    end
    return y
end

function mala_proposal(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector; cholM=nothing
)
    ws = MALAWorkspace(x)
    y = similar(x)
    mala_proposal!(y, ws, logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    return y
end

"""
    mala_logα!(ws, logp, gradlogp, x, y, ε; cholM=nothing)

Allocation-light log acceptance ratio. Uses `ws.g_x`, `ws.g_y`, `ws.μ`, `ws.r`.
"""
function mala_logα!(
    ws::MALAWorkspace,
    logp,
    gradlogp,
    x::AbstractVector,
    y::AbstractVector,
    ϵ::Real;
    cholM=nothing,
)
    copyto!(ws.g_x, gradlogp(x))
    copyto!(ws.g_y, gradlogp(y))
    logp_x = logp(x)
    logp_y = logp(y)
    logq_y_given_x = logq_mala!(ws, y, x, ws.g_x, ϵ; cholM=cholM)
    logq_x_given_y = logq_mala!(ws, x, y, ws.g_y, ϵ; cholM=cholM)
    return (logp_y + logq_x_given_y) - (logp_x + logq_y_given_x)
end

function mala_logα(
    logp, gradlogp, x::AbstractVector, y::AbstractVector, ϵ::Real; cholM=nothing
)
    ws = MALAWorkspace(x)
    return mala_logα!(ws, logp, gradlogp, x, y, ϵ; cholM=cholM)
end

"""
One taped MALA step.

Returns:
- x_next
"""
function mala_step_taped!(
    x_next::AbstractVector,
    ws::MALAWorkspace,
    logp,
    gradlogp,
    x::AbstractVector,
    ϵ::Real,
    ξ::AbstractVector,
    u::Real;
    cholM=nothing,
)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    0.0 < u < 1.0 || throw(ArgumentError("u must be in (0, 1)"))

    mala_proposal!(ws.y, ws, logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    logp_x = logp(x)
    logp_y = logp(ws.y)
    copyto!(ws.g_y, gradlogp(ws.y))

    logq_y_given_x = logq_mala!(ws, ws.y, x, ws.g_x, ϵ; cholM=cholM)
    logq_x_given_y = logq_mala!(ws, x, ws.y, ws.g_y, ϵ; cholM=cholM)
    logα = (logp_y + logq_x_given_y) - (logp_x + logq_y_given_x)

    if log(u) < logα
        copyto!(x_next, ws.y)
    else
        copyto!(x_next, x)
    end
    return x_next
end

function mala_step_taped(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real; cholM=nothing
)
    ws = MALAWorkspace(x)
    x_next = similar(x)
    mala_step_taped!(x_next, ws, logp, gradlogp, x, ϵ, ξ, u; cholM=cholM)
    return x_next
end

"""
Run sequential taped MALA for T steps.

Returns:
- xs: Vector of states length T+1, xs[1]=x0, xs[t+1]=x_t
"""
function run_mala_sequential_taped(
    logp,
    gradlogp,
    x0::AbstractVector,
    ϵ::Real,
    ξs::Vector{<:AbstractVector},
    us::AbstractVector;
    cholM=nothing,
)
    T = length(us)
    length(ξs) == T || throw(DimensionMismatch("ξs and us must have the same length"))
    xs = Vector{typeof(x0)}(undef, T + 1)
    xs[1] = copy(x0)

    ws = MALAWorkspace(x0)
    x = copy(x0)
    x_next = similar(x0)

    for t in 1:T
        mala_step_taped!(x_next, ws, logp, gradlogp, x, ϵ, ξs[t], us[t]; cholM=cholM)
        xs[t + 1] = copy(x_next)
        x, x_next = x_next, x
    end
    return xs
end

"""
Primal accept indicator for a taped MALA step.
Returns a float in {0, 1} matching the precision of `u`.
"""
function mala_accept_indicator(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real; cholM=nothing
)
    ws = MALAWorkspace(x)
    mala_proposal!(ws.y, ws, logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    logα = mala_logα!(ws, logp, gradlogp, x, ws.y, ϵ; cholM=cholM)
    FP = typeof(float(u))
    return (log(u) < logα) ? one(FP) : zero(FP)
end

function mala_step_full(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real; cholM=nothing
)
    x_next, accepted, _ = mala_step_with_logα(logp, gradlogp, x, ϵ, ξ, u; cholM=cholM)
    return x_next, accepted
end

function mala_step_with_logα(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real; cholM=nothing
)
    ws = MALAWorkspace(x)
    x_next = similar(x)
    return mala_step_with_logα!(x_next, ws, logp, gradlogp, x, ϵ, ξ, u; cholM=cholM)
end

function mala_step_with_logα!(
    x_next::AbstractVector,
    ws::MALAWorkspace,
    logp,
    gradlogp,
    x::AbstractVector,
    ϵ::Real,
    ξ::AbstractVector,
    u::Real;
    cholM=nothing,
)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    length(x_next) == length(x) ||
        throw(DimensionMismatch("x_next must have the same length as x"))
    length(ws.y) == length(x) ||
        throw(DimensionMismatch("workspace must match length of x"))
    0.0 < u < 1.0 || throw(ArgumentError("u must be in (0, 1)"))

    mala_proposal!(ws.y, ws, logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    logp_x = logp(x)
    logp_y = logp(ws.y)
    copyto!(ws.g_y, gradlogp(ws.y))

    logq_yx = logq_mala!(ws, ws.y, x, ws.g_x, ϵ; cholM=cholM)
    logq_xy = logq_mala!(ws, x, ws.y, ws.g_y, ϵ; cholM=cholM)
    logα = (logp_y + logq_xy) - (logp_x + logq_yx)

    accepted = log(u) < logα
    if accepted
        copyto!(x_next, ws.y)
    else
        copyto!(x_next, x)
    end
    return x_next, accepted, logα
end

"""
Stop-gradient surrogate step used for Jacobians.
"""
function mala_step_surrogate_sigmoid(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real; cholM=nothing
)
    ws = MALAWorkspace(x)
    mala_proposal!(ws.y, ws, logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    logα = mala_logα!(ws, logp, gradlogp, x, ws.y, ϵ; cholM=cholM)
    g̃ = logα - log(u)
    g = one(g̃) / (one(g̃) + exp(-g̃))
    out = similar(x)
    @. out = g * ws.y + (one(g) - g) * x
    return out
end

# Apply mass matrix to a D×N matrix of gradient columns.
_apply_M_batched!(out, G, ::Nothing, tmp=out) = copyto!(out, G)
function _apply_M_batched!(out, G, cholM::Cholesky, tmp)
    mul!(tmp, adjoint(cholM.L), G)
    mul!(out, cholM.L, tmp)
    return out
end

_apply_L_batched!(out, Ξ, ::Nothing) = copyto!(out, Ξ)
_apply_L_batched!(out, Ξ, cholM::Cholesky) = mul!(out, cholM.L, Ξ)

function _sum_columns!(out::AbstractVector, row::AbstractMatrix, A::AbstractMatrix)
    sum!(row, A)
    copyto!(out, row)
    return out
end

function logq_mala_batched!(
    out::AbstractVector,
    ws::MALABatchedWorkspace,
    Y::AbstractMatrix,
    X::AbstractMatrix,
    gradlogp_X::AbstractMatrix,
    ε::Real;
    cholM=nothing,
)
    D, N = size(X)
    size(Y) == (D, N) || throw(DimensionMismatch("X and Y must have the same size"))
    size(gradlogp_X) == (D, N) ||
        throw(DimensionMismatch("gradlogp_X must have the same size as X"))
    length(out) == N || throw(DimensionMismatch("out must have length size(X,2)"))

    εT = convert(eltype(X), ε)
    if cholM === nothing
        @. ws.R = Y - X - εT * gradlogp_X
        @. ws.R = abs2(ws.R)
        _sum_columns!(out, ws.row, ws.R)
    else
        _apply_M_batched!(ws.MG_X, gradlogp_X, cholM, ws.solve_tmp)
        @. ws.R = Y - X - εT * ws.MG_X
        ldiv!(ws.Minv_R, cholM.L, ws.R)
        @. ws.Minv_R = abs2(ws.Minv_R)
        _sum_columns!(out, ws.row, ws.Minv_R)
    end

    FT = typeof(εT)
    ldet = cholM === nothing ? zero(FT) : FT(_logdet_M(cholM))
    c = (FT(D) / 2) * log(FT(4π) * εT) + FT(0.5) * ldet
    @. out = -FT(0.5) * out / (2εT) - c
    return out
end

function logq_mala_batched(Y, X, gradlogp_X, ε; cholM=nothing)
    ws = MALABatchedWorkspace(X)
    out = similar(X, eltype(X), size(X, 2))
    return logq_mala_batched!(out, ws, Y, X, gradlogp_X, ε; cholM=cholM)
end

function mala_step_batched!(
    X_next::AbstractMatrix,
    accepted::AbstractVector,
    ws::MALABatchedWorkspace,
    logp_batch,
    gradlogp_batch,
    X::AbstractMatrix,
    ε::Real,
    Ξ::AbstractMatrix,
    u::AbstractVector;
    cholM=nothing,
)
    D, N = size(X)
    size(Ξ) == (D, N) || throw(DimensionMismatch("X and Ξ must have the same size"))
    size(X_next) == (D, N) ||
        throw(DimensionMismatch("X_next must have the same size as X"))
    length(u) == N || throw(DimensionMismatch("u must have length N = size(X,2)"))
    length(accepted) == N ||
        throw(DimensionMismatch("accepted must have length N = size(X,2)"))

    εT = convert(eltype(X), ε)
    sqrt2εT = sqrt(2 * εT)

    copyto!(ws.G_X, gradlogp_batch(X))
    _apply_M_batched!(ws.MG_X, ws.G_X, cholM, ws.solve_tmp)
    _apply_L_batched!(ws.LΞ, Ξ, cholM)
    @. ws.Y = X + εT * ws.MG_X + sqrt2εT * ws.LΞ

    copyto!(ws.lp_X, logp_batch(X))
    copyto!(ws.lp_Y, logp_batch(ws.Y))
    copyto!(ws.G_Y, gradlogp_batch(ws.Y))

    logq_mala_batched!(ws.lq_YX, ws, ws.Y, X, ws.G_X, εT; cholM=cholM)
    logq_mala_batched!(ws.lq_XY, ws, X, ws.Y, ws.G_Y, εT; cholM=cholM)

    @. ws.logα = (ws.lp_Y + ws.lq_XY) - (ws.lp_X + ws.lq_YX)
    @. ws.logu = log(u)
    @. accepted = ws.logu < ws.logα
    copyto!(ws.accepted_row, accepted)
    @. X_next = ifelse(ws.accepted_row, ws.Y, X)
    return X_next, accepted
end

function mala_step_batched(
    logp_batch,
    gradlogp_batch,
    X::AbstractMatrix,
    ε::Real,
    Ξ::AbstractMatrix,
    u::AbstractVector;
    cholM=nothing,
)
    D, N = size(X)
    size(Ξ) == (D, N) || throw(DimensionMismatch("X and Ξ must have the same size"))
    length(u) == N || throw(DimensionMismatch("u must have length N = size(X,2)"))
    ws = MALABatchedWorkspace(X)
    X_next = similar(X)
    accepted = similar(X, Bool, N)
    return mala_step_batched!(
        X_next, accepted, ws, logp_batch, gradlogp_batch, X, ε, Ξ, u; cholM=cholM
    )
end

"""
Batched fused surrogate forward step and JVP over columns.

`X`, `Ξ`, and `Z` are all D×T, and `u` has length T.
Returns `(FT, Jt)` where both outputs are D×T.
"""
function mala_step_batched_fwd_and_jvp!(
    FT::AbstractMatrix,
    Jt::AbstractMatrix,
    ws::MALABatchedWorkspace,
    logp_batch,
    gradlogp_batch,
    hvp_batch,
    X::AbstractMatrix,
    ε::Real,
    Ξ::AbstractMatrix,
    u::AbstractVector,
    Z::AbstractMatrix;
    cholM=nothing,
)
    D, T = size(X)
    size(Ξ) == (D, T) || throw(DimensionMismatch("X and Ξ must have the same size"))
    size(Z) == (D, T) || throw(DimensionMismatch("X and Z must have the same size"))
    size(FT) == (D, T) || throw(DimensionMismatch("FT must have the same size as X"))
    size(Jt) == (D, T) || throw(DimensionMismatch("Jt must have the same size as X"))
    length(u) == T || throw(DimensionMismatch("u must have length size(X,2)"))

    εT = convert(eltype(X), ε)
    oneT = one(eltype(X))

    copyto!(ws.G_X, gradlogp_batch(X))
    _apply_M_batched!(ws.MG_X, ws.G_X, cholM, ws.solve_tmp)
    _apply_L_batched!(ws.LΞ, Ξ, cholM)
    @. ws.Y = X + εT * ws.MG_X + sqrt(2 * εT) * ws.LΞ

    copyto!(ws.lp_X, logp_batch(X))
    copyto!(ws.lp_Y, logp_batch(ws.Y))
    copyto!(ws.G_Y, gradlogp_batch(ws.Y))

    logq_mala_batched!(ws.lq_YX, ws, ws.Y, X, ws.G_X, εT; cholM=cholM)
    logq_mala_batched!(ws.lq_XY, ws, X, ws.Y, ws.G_Y, εT; cholM=cholM)

    @. ws.logα = (ws.lp_Y + ws.lq_XY) - (ws.lp_X + ws.lq_YX)
    @. ws.logu = log(u)
    @. ws.g = oneT / (oneT + exp(-(ws.logα - ws.logu)))
    @. ws.accepted = ws.logu < ws.logα
    copyto!(ws.accepted_row, ws.accepted)
    @. FT = ifelse(ws.accepted_row, ws.Y, X)

    copyto!(ws.Hv_X, hvp_batch(X, Z))
    _apply_M_batched!(ws.M_Hv_X, ws.Hv_X, cholM, ws.solve_tmp)
    @. ws.W = Z + εT * ws.M_Hv_X

    copyto!(ws.Hv_Y, hvp_batch(ws.Y, ws.W))
    _apply_M_batched!(ws.MG_Y, ws.G_Y, cholM, ws.solve_tmp)
    _apply_M_batched!(ws.M_Hv_Y, ws.Hv_Y, cholM, ws.solve_tmp)
    @. ws.R = X - ws.Y - εT * ws.MG_Y
    @. ws.dR = Z - ws.W - εT * ws.M_Hv_Y

    if cholM === nothing
        @. ws.prod = ws.R * ws.dR
    else
        ldiv!(ws.Minv_R, cholM, ws.R)
        @. ws.prod = ws.Minv_R * ws.dR
    end
    _sum_columns!(ws.dot3, ws.row, ws.prod)
    @. ws.prod = ws.G_Y * ws.W
    _sum_columns!(ws.dot1, ws.row, ws.prod)
    @. ws.prod = ws.G_X * Z
    _sum_columns!(ws.dot2, ws.row, ws.prod)

    @. ws.dlogα = ws.dot1 - ws.dot2 - inv(2 * εT) * ws.dot3
    @. ws.dg = ws.g * (oneT - ws.g) * ws.dlogα

    copyto!(ws.row, ws.dg)
    @. Jt = ifelse(ws.accepted_row, ws.W, Z) + (ws.Y - X) * ws.row

    return FT, Jt
end

function mala_step_batched_fwd_and_jvp!(
    FT::AbstractMatrix,
    Jt::AbstractMatrix,
    logp_batch,
    gradlogp_batch,
    hvp_batch,
    X::AbstractMatrix,
    ε::Real,
    Ξ::AbstractMatrix,
    u::AbstractVector,
    Z::AbstractMatrix;
    cholM=nothing,
)
    ws = MALABatchedWorkspace(X)
    return mala_step_batched_fwd_and_jvp!(
        FT, Jt, ws, logp_batch, gradlogp_batch, hvp_batch, X, ε, Ξ, u, Z; cholM=cholM
    )
end

function mala_step_batched_fwd_and_jvp(
    logp_batch,
    gradlogp_batch,
    hvp_batch,
    X::AbstractMatrix,
    ε::Real,
    Ξ::AbstractMatrix,
    u::AbstractVector,
    Z::AbstractMatrix;
    cholM=nothing,
)
    FT = similar(X)
    Jt = similar(X)
    ws = MALABatchedWorkspace(X)
    return mala_step_batched_fwd_and_jvp!(
        FT, Jt, ws, logp_batch, gradlogp_batch, hvp_batch, X, ε, Ξ, u, Z; cholM=cholM
    )
end

"""
Explicit JVP (pushforward) of `mala_step_surrogate_sigmoid` w.r.t. `x`.
"""
function mala_step_surrogate_sigmoid_jvp(
    logp,
    gradlogp,
    x::AbstractVector,
    ε::Real,
    ξ::AbstractVector,
    u::Real,
    v::AbstractVector,
    hvp_fn;
    cholM=nothing,
)
    ws = MALAWorkspace(x)

    # forward pass with reused scratch
    mala_proposal!(ws.y, ws, logp, gradlogp, x, ε, ξ; cholM=cholM)
    logp_x = logp(x)
    logp_y = logp(ws.y)
    copyto!(ws.g_y, gradlogp(ws.y))
    logq_yx = logq_mala!(ws, ws.y, x, ws.g_x, ε; cholM=cholM)
    logq_xy = logq_mala!(ws, x, ws.y, ws.g_y, ε; cholM=cholM)
    logα = (logp_y + logq_xy) - (logp_x + logq_yx)
    g̃ = logα - log(u)
    g = one(g̃) / (one(g̃) + exp(-g̃))
    a = log(u) < logα ? one(g) : zero(g)

    copyto!(ws.Hv_x, hvp_fn(x, v))
    if cholM === nothing
        @. ws.w = v + ε * ws.Hv_x
    else
        _apply_M!(ws.μ, ws.Hv_x, cholM, ws.solve_buf)
        @. ws.w = v + ε * ws.μ
    end

    copyto!(ws.Hv_y, hvp_fn(ws.y, ws.w))
    if cholM === nothing
        @. ws.r = x - ws.y - ε * ws.g_y
        @. ws.dr = v - ws.w - ε * ws.Hv_y
        Minv_r = ws.r
    else
        _apply_M!(ws.μ, ws.g_y, cholM, ws.solve_buf)
        _apply_M!(ws.jvp_out, ws.Hv_y, cholM, ws.solve_buf)
        @. ws.r = x - ws.y - ε * ws.μ
        @. ws.dr = v - ws.w - ε * ws.jvp_out
        ldiv!(ws.solve_buf, cholM, ws.r)
        Minv_r = ws.solve_buf
    end

    dlogα = dot(ws.g_y, ws.w) - dot(ws.g_x, v) - inv(2 * ε) * dot(Minv_r, ws.dr)
    dg = g * (one(g) - g) * dlogα

    @. ws.jvp_out = a * ws.w + (ws.y - x) * dg + (one(a) - a) * v
    return copy(ws.jvp_out)
end

"""
Fused MALA forward step and JVP of the sigmoid surrogate, sharing the primal computation.

Returns `(x_next, J·v)`.
"""
function mala_step_taped_and_jvp!(
    x_next::AbstractVector,
    jvp_out::AbstractVector,
    ws::MALAWorkspace,
    logp,
    gradlogp,
    x::AbstractVector,
    ε::Real,
    ξ::AbstractVector,
    u::Real,
    v::AbstractVector,
    hvp_fn;
    cholM=nothing,
)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    0.0 < u < 1.0 || throw(ArgumentError("u must be in (0, 1)"))

    mala_proposal!(ws.y, ws, logp, gradlogp, x, ε, ξ; cholM=cholM)
    logp_x = logp(x)
    logp_y = logp(ws.y)
    copyto!(ws.g_y, gradlogp(ws.y))
    logq_yx = logq_mala!(ws, ws.y, x, ws.g_x, ε; cholM=cholM)
    logq_xy = logq_mala!(ws, x, ws.y, ws.g_y, ε; cholM=cholM)
    logα = (logp_y + logq_xy) - (logp_x + logq_yx)

    accepted = log(u) < logα
    if accepted
        copyto!(x_next, ws.y)
    else
        copyto!(x_next, x)
    end

    g̃ = logα - log(u)
    g = one(g̃) / (one(g̃) + exp(-g̃))
    a = accepted ? one(g) : zero(g)

    copyto!(ws.Hv_x, hvp_fn(x, v))
    if cholM === nothing
        @. ws.w = v + ε * ws.Hv_x
    else
        _apply_M!(ws.μ, ws.Hv_x, cholM, ws.solve_buf)
        @. ws.w = v + ε * ws.μ
    end

    copyto!(ws.Hv_y, hvp_fn(ws.y, ws.w))
    if cholM === nothing
        @. ws.r = x - ws.y - ε * ws.g_y
        @. ws.dr = v - ws.w - ε * ws.Hv_y
        Minv_r = ws.r
    else
        _apply_M!(ws.μ, ws.g_y, cholM, ws.solve_buf)
        _apply_M!(ws.jvp_out, ws.Hv_y, cholM, ws.solve_buf)
        @. ws.r = x - ws.y - ε * ws.μ
        @. ws.dr = v - ws.w - ε * ws.jvp_out
        ldiv!(ws.solve_buf, cholM, ws.r)
        Minv_r = ws.solve_buf
    end

    dlogα = dot(ws.g_y, ws.w) - dot(ws.g_x, v) - inv(2 * ε) * dot(Minv_r, ws.dr)
    dg = g * (one(g) - g) * dlogα

    @. jvp_out = a * ws.w + (ws.y - x) * dg + (one(a) - a) * v
    return x_next, jvp_out
end

function mala_step_taped_and_jvp(
    logp,
    gradlogp,
    x::AbstractVector,
    ε::Real,
    ξ::AbstractVector,
    u::Real,
    v::AbstractVector,
    hvp_fn;
    cholM=nothing,
)
    ws = MALAWorkspace(x)
    x_next = similar(x)
    jvp_out = similar(x)
    mala_step_taped_and_jvp!(
        x_next, jvp_out, ws, logp, gradlogp, x, ε, ξ, u, v, hvp_fn; cholM=cholM
    )
    return x_next, jvp_out
end

end # module MALA
