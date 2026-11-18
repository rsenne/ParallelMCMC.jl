module MALA

using Random, LinearAlgebra

import ..ParallelMCMC: needs_host_staging

export MALAWorkspace,
    MALABatchedWorkspace,
    logq_mala,
    logq_mala!,
    mala_step_taped,
    mala_step_taped!,
    mala_proposal,
    mala_proposal!,
    mala_logα,
    mala_logα!,
    mala_step_with_logα,
    mala_step_with_logα!,
    mala_step_surrogate_sigmoid,
    mala_step_surrogate_sigmoid_jvp,
    mala_step_taped_and_jvp,
    mala_step_batched,
    mala_step_batched!,
    mala_step_batched_fwd_and_jvp,
    mala_step_batched_fwd_and_jvp!

#= The MALA step has two implementations:

- The vector step (`_mala_step!`, on a `MALAWorkspace`) runs the sequential samplers,
  one state at a time.
- The batched step (`MALABatchedWorkspace`) advances independent columns at once. DEER
  runs it over all T states of a trajectory, and it holds the only implementation of
  the surrogate JVP: the vector JVP functions run it on a D×1 matrix.

The sequential samplers keep a vector step because the batched one on a D×1 matrix
costs 100 to 250 ns more per step, which doubles a sequential step on cheap targets. =#

# Preconditioner helpers for vectors and D×N matrices. `cholM` is `nothing` (M = I) or
# a Cholesky factor of M.
_apply_M!(out, g, ::Nothing, tmp=out) = copyto!(out, g)
function _apply_M!(out, g, cholM::Cholesky, tmp)
    mul!(tmp, adjoint(cholM.L), g)
    mul!(out, cholM.L, tmp)
    return out
end

_apply_L!(out, ξ, ::Nothing) = (out .= ξ)
_apply_L!(out, ξ, cholM::Cholesky) = mul!(out, cholM.L, ξ)

_quad_Minv!(tmp, r, ::Nothing) = sum(abs2, r)
function _quad_Minv!(tmp, r, cholM::Cholesky)
    ldiv!(tmp, cholM.L, r)
    return sum(abs2, tmp)
end

_logdet_M(::Nothing) = false  # Bool promotes to any numeric type without widening
_logdet_M(cholM::Cholesky) = logdet(cholM)

"""
Reusable scratch buffers for the vector MALA step.

All vectors are created with the same array type / device placement as `x_template`,
so this workspace is GPU-compatible when constructed from a `CuVector`.
"""
struct MALAWorkspace{V}
    g_x::V
    g_y::V
    y::V
    μ::V
    r::V
    w::V
    solve_buf::V
end

function MALAWorkspace(x::AbstractVector)
    return MALAWorkspace(
        similar(x), similar(x), similar(x), similar(x), similar(x), similar(x), similar(x)
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

_sigmoid(x) = one(x) / (one(x) + exp(-x))

# ---------------------------------------------------------------------------
# Vector step
# ---------------------------------------------------------------------------

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
    return _mala_proposal_from_grad!(y, ws, x, ws.g_x, ϵ, ξ; cholM=cholM)
end

# The proposal for a known gradient `g` at `x`.
function _mala_proposal_from_grad!(
    y::AbstractVector,
    ws::MALAWorkspace,
    x::AbstractVector,
    g::AbstractVector,
    ϵ::Real,
    ξ::AbstractVector;
    cholM=nothing,
)
    sqrt2ϵ = sqrt(2 * ϵ)
    if cholM === nothing
        @. y = x + ϵ * g + sqrt2ϵ * ξ
    else
        _apply_M!(ws.μ, g, cholM, ws.solve_buf)
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

# log α of the move x → y, given the log-density and gradient at both.
function _mala_logα_from!(ws, x, y, logp_x, logp_y, g_x, g_y, ϵ; cholM)
    logq_yx = logq_mala!(ws, y, x, g_x, ϵ; cholM=cholM)
    logq_xy = logq_mala!(ws, x, y, g_y, ϵ; cholM=cholM)
    return (logp_y + logq_xy) - (logp_x + logq_yx)
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
    return _mala_logα_from!(ws, x, y, logp(x), logp(y), ws.g_x, ws.g_y, ϵ; cholM=cholM)
end

function mala_logα(
    logp, gradlogp, x::AbstractVector, y::AbstractVector, ϵ::Real; cholM=nothing
)
    ws = MALAWorkspace(x)
    return mala_logα!(ws, logp, gradlogp, x, y, ϵ; cholM=cholM)
end

#= One MALA step from `x`, whose log-density `logp_x` and gradient `g_x` are known.
Every vector step runs through here. Leaves the proposal in `ws.y` and its gradient
in `ws.g_y`, and returns `(accepted, logα, logp_y)`. =#
function _mala_step!(ws::MALAWorkspace, logp, gradlogp, x, logp_x, g_x, ϵ, ξ, u; cholM)
    _mala_proposal_from_grad!(ws.y, ws, x, g_x, ϵ, ξ; cholM=cholM)
    logp_y = logp(ws.y)
    copyto!(ws.g_y, gradlogp(ws.y))
    logα = _mala_logα_from!(ws, x, ws.y, logp_x, logp_y, g_x, ws.g_y, ϵ; cholM=cholM)
    return log(u) < logα, logα, logp_y
end

function _check_step_args(x_next, ws::MALAWorkspace, x, ξ, u)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    length(x_next) == length(x) ||
        throw(DimensionMismatch("x_next must have the same length as x"))
    length(ws.y) == length(x) ||
        throw(DimensionMismatch("workspace must match length of x"))
    0.0 < u < 1.0 || throw(ArgumentError("u must be in (0, 1)"))
    return nothing
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
    mala_step_with_logα!(x_next, ws, logp, gradlogp, x, ϵ, ξ, u; cholM=cholM)
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
    _check_step_args(x_next, ws, x, ξ, u)
    copyto!(ws.g_x, gradlogp(x))
    accepted, logα, _ = _mala_step!(
        ws, logp, gradlogp, x, logp(x), ws.g_x, ϵ, ξ, u; cholM=cholM
    )
    copyto!(x_next, accepted ? ws.y : x)
    return x_next, accepted, logα
end

"""
    mala_step_cached!(x_next, ws, logp, gradlogp, x, logp_x, grad_x, ε, ξ, u; cholM=nothing)
        -> (x_next, accepted, logα, logp_next, grad_next)

Same step as [`mala_step_with_logα!`](@ref), but `logp(x)` and `gradlogp(x)` are passed in
as `logp_x` and `grad_x`, so the step evaluates `logp` and `gradlogp` once each, at the
proposal. It returns the log-density and gradient at `x_next` for the next step.

`grad_next` is a new array when the proposal is accepted and `grad_x` otherwise. Neither
aliases `ws`, so a later step cannot overwrite it.
"""
function mala_step_cached!(
    x_next::AbstractVector,
    ws::MALAWorkspace,
    logp,
    gradlogp,
    x::AbstractVector,
    logp_x::Real,
    grad_x::AbstractVector,
    ϵ::Real,
    ξ::AbstractVector,
    u::Real;
    cholM=nothing,
)
    _check_step_args(x_next, ws, x, ξ, u)
    length(grad_x) == length(x) ||
        throw(DimensionMismatch("grad_x must have the same length as x"))

    accepted, logα, logp_y = _mala_step!(
        ws, logp, gradlogp, x, logp_x, grad_x, ϵ, ξ, u; cholM=cholM
    )
    if accepted
        copyto!(x_next, ws.y)
        return x_next, true, logα, logp_y, copy(ws.g_y)
    else
        copyto!(x_next, x)
        return x_next, false, logα, logp_x, grad_x
    end
end

"""
Stop-gradient surrogate step used for Jacobians.
"""
function mala_step_surrogate_sigmoid(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real; cholM=nothing
)
    ws = MALAWorkspace(x)
    copyto!(ws.g_x, gradlogp(x))
    _, logα, _ = _mala_step!(ws, logp, gradlogp, x, logp(x), ws.g_x, ϵ, ξ, u; cholM=cholM)
    g = _sigmoid(logα - log(u))
    out = similar(x)
    @. out = g * ws.y + (one(g) - g) * x
    return out
end

# ---------------------------------------------------------------------------
# Batched step
# ---------------------------------------------------------------------------

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
        _apply_M!(ws.MG_X, gradlogp_X, cholM, ws.solve_tmp)
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

# Proposals Y = X + ε M ∇logp(X) + √(2ε) L Ξ, from the gradients in `ws.G_X`.
function _mala_proposal_batched!(ws::MALABatchedWorkspace, X, ε, Ξ, cholM)
    εT = convert(eltype(X), ε)
    _apply_M!(ws.MG_X, ws.G_X, cholM, ws.solve_tmp)
    _apply_L!(ws.LΞ, Ξ, cholM)
    @. ws.Y = X + εT * ws.MG_X + sqrt(2 * εT) * ws.LΞ
    return ws.Y
end

#= Accept or reject each column's proposal: log α into `ws.logα`, flags into
`accepted`, the next states into `X_next`. Needs the log-densities and gradients at X
and Y in `ws`. `ε` stays as given: under Reactant tracing its converted form is a
traced number, not a `Real`. =#
function _mala_accept_batched!(X_next, accepted, ws::MALABatchedWorkspace, X, ε, u, cholM)
    logq_mala_batched!(ws.lq_YX, ws, ws.Y, X, ws.G_X, ε; cholM=cholM)
    logq_mala_batched!(ws.lq_XY, ws, X, ws.Y, ws.G_Y, ε; cholM=cholM)
    @. ws.logα = (ws.lp_Y + ws.lq_XY) - (ws.lp_X + ws.lq_YX)
    @. ws.logu = log(u)
    @. accepted = ws.logu < ws.logα
    copyto!(ws.accepted_row, accepted)
    @. X_next = ifelse(ws.accepted_row, ws.Y, X)
    return X_next
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

    copyto!(ws.G_X, gradlogp_batch(X))
    _mala_proposal_batched!(ws, X, ε, Ξ, cholM)
    copyto!(ws.lp_X, logp_batch(X))
    copyto!(ws.lp_Y, logp_batch(ws.Y))
    copyto!(ws.G_Y, gradlogp_batch(ws.Y))
    _mala_accept_batched!(X_next, accepted, ws, X, ε, u, cholM)
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

#= Evaluators for the batched step and JVP. `_value_grad_hvp!(lp, G, Hv, e, X, V)`
writes the log-densities, gradients and HVPs of the columns of `X` along `V`.

- A callable `(X, V) -> (lp, G, Hv)`: a model's `value_grad_hvp_batch`.
- `SeparateBatchEval`: a model's three batched functions.
- `ColumnwiseEval`: a model's per-state functions, one column at a time. =#

function _value_grad_hvp!(lp, G, Hv, value_grad_hvp_batch, X, V)
    lp_X, G_X, Hv_X = value_grad_hvp_batch(X, V)
    copyto!(lp, lp_X)
    copyto!(G, G_X)
    copyto!(Hv, Hv_X)
    return nothing
end

struct SeparateBatchEval{F,G,H}
    logp_batch::F
    gradlogp_batch::G
    hvp_batch::H
end

#= Copy each result before the next call: hand-written batched functions often
return a buffer that the next call overwrites. =#
function _value_grad_hvp!(lp, G, Hv, e::SeparateBatchEval, X, V)
    copyto!(G, e.gradlogp_batch(X))
    copyto!(lp, e.logp_batch(X))
    copyto!(Hv, e.hvp_batch(X, V))
    return nothing
end

"""
    ColumnwiseEval(logp, gradlogp, hvp, x_template)

Batched evaluator over per-state functions, so models without batched functions run
the same batched MALA step. Each column is copied into a buffer like `x_template`
before the call, so AD preparations made for that vector type still apply.
"""
struct ColumnwiseEval{F,G,H,V<:AbstractVector}
    logp::F
    gradlogp::G
    hvp::H
    x::V
    v::V
end

function ColumnwiseEval(logp, gradlogp, hvp, x_template::AbstractVector)
    return ColumnwiseEval(logp, gradlogp, hvp, similar(x_template), similar(x_template))
end

function _value_grad_hvp!(lp, G, Hv, e::ColumnwiseEval, X, V)
    D, N = size(X)
    length(e.x) == D || throw(DimensionMismatch("evaluator buffers must have length D"))
    # Log-densities come back as host scalars; device vectors take them in one copy.
    lp_host = needs_host_staging(lp) ? Vector{eltype(lp)}(undef, N) : lp
    for t in 1:N
        off = (t - 1) * D + 1
        copyto!(e.x, 1, X, off, D)
        copyto!(e.v, 1, V, off, D)
        copyto!(G, off, e.gradlogp(e.x), 1, D)
        lp_host[t] = e.logp(e.x)
        copyto!(Hv, off, e.hvp(e.x, e.v), 1, D)
    end
    lp_host === lp || copyto!(lp, lp_host)
    return nothing
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
    return _mala_step_batched_fwd_and_jvp!(
        FT,
        Jt,
        ws,
        SeparateBatchEval(logp_batch, gradlogp_batch, hvp_batch),
        X,
        ε,
        Ξ,
        u,
        Z;
        cholM=cholM,
    )
end

#= The MALA step and the JVP of its sigmoid surrogate, for every column. The only
implementation of the surrogate JVP. =#
function _mala_step_batched_fwd_and_jvp!(
    FT::AbstractMatrix,
    Jt::AbstractMatrix,
    ws::MALABatchedWorkspace,
    value_grad_hvp_batch,
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

    _value_grad_hvp!(ws.lp_X, ws.G_X, ws.Hv_X, value_grad_hvp_batch, X, Z)
    _mala_proposal_batched!(ws, X, ε, Ξ, cholM)

    _apply_M!(ws.M_Hv_X, ws.Hv_X, cholM, ws.solve_tmp)
    @. ws.W = Z + εT * ws.M_Hv_X

    _value_grad_hvp!(ws.lp_Y, ws.G_Y, ws.Hv_Y, value_grad_hvp_batch, ws.Y, ws.W)
    _mala_accept_batched!(FT, ws.accepted, ws, X, ε, u, cholM)
    @. ws.g = oneT / (oneT + exp(-(ws.logα - ws.logu)))

    _apply_M!(ws.MG_Y, ws.G_Y, cholM, ws.solve_tmp)
    _apply_M!(ws.M_Hv_Y, ws.Hv_Y, cholM, ws.solve_tmp)
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

# ---------------------------------------------------------------------------
# Vector JVP: the batched step on one column
# ---------------------------------------------------------------------------

# `x` as a D×1 matrix. Wrapped arrays are copied so the batched broadcasts see a
# plain array, which device array types need.
_as_column(x::DenseVector) = reshape(x, length(x), 1)
_as_column(x::AbstractVector) = reshape(copy(x), length(x), 1)

function _mala_step_and_jvp_vector(logp, gradlogp, hvp_fn, x, ε, ξ, u, v; cholM)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    length(v) == length(x) || throw(DimensionMismatch("x and v must have the same length"))
    0.0 < u < 1.0 || throw(ArgumentError("u must be in (0, 1)"))

    X = _as_column(x)
    U = fill!(similar(X, eltype(X), 1), u)
    FT, Jt = similar(X), similar(X)
    _mala_step_batched_fwd_and_jvp!(
        FT,
        Jt,
        MALABatchedWorkspace(X),
        ColumnwiseEval(logp, gradlogp, hvp_fn, x),
        X,
        ε,
        _as_column(ξ),
        U,
        _as_column(v);
        cholM=cholM,
    )
    return copyto!(similar(x), FT), copyto!(similar(x), Jt)
end

"""
    mala_step_surrogate_sigmoid_jvp(logp, gradlogp, x, ε, ξ, u, v, hvp_fn; cholM=nothing)

JVP (pushforward) along `v` of [`mala_step_surrogate_sigmoid`](@ref) with respect to `x`.
Runs the batched step on `x` as a one-column matrix.
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
    _, jvp = _mala_step_and_jvp_vector(logp, gradlogp, hvp_fn, x, ε, ξ, u, v; cholM=cholM)
    return jvp
end

"""
    mala_step_taped_and_jvp(logp, gradlogp, x, ε, ξ, u, v, hvp_fn; cholM=nothing)

The MALA step from `x` and the JVP of its sigmoid surrogate along `v`, sharing one
evaluation. Returns `(x_next, J·v)`.
"""
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
    return _mala_step_and_jvp_vector(logp, gradlogp, hvp_fn, x, ε, ξ, u, v; cholM=cholM)
end

end # module MALA
