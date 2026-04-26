module MALA

using Random, LinearAlgebra

export MALAWorkspace,
       logq_mala, logq_mala!,
       mala_step_taped, mala_step_taped!,
       run_mala_sequential_taped,
       mala_proposal, mala_proposal!,
       mala_logα, mala_logα!,
       mala_accept_indicator,
       mala_step_full,
       mala_step_with_logα,
       mala_step_surrogate_sigmoid,
       mala_step_surrogate_sigmoid_jvp,
       mala_step_taped_and_jvp,
       mala_step_taped_and_jvp!,
       mala_step_batched_fwd_and_jvp

# Preconditioner dispatch helpers. cholM is either nothing (identity) or a Cholesky factor.
_apply_M(g, ::Nothing) = g
_apply_M(g, cholM::Cholesky) = cholM.L * (cholM.L' * g)

_apply_L(ξ, ::Nothing) = ξ
_apply_L(ξ, cholM::Cholesky) = cholM.L * ξ

_quad_Minv(r, ::Nothing) = dot(r, r)
function _quad_Minv(r, cholM::Cholesky)
    w = cholM.L \ r
    return dot(w, w)
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
end

function MALAWorkspace(x_template::AbstractVector)
    g_x = similar(x_template)
    g_y = similar(x_template)
    y = similar(x_template)
    μ = similar(x_template)
    r = similar(x_template)
    Hv_x = similar(x_template)
    Hv_y = similar(x_template)
    w = similar(x_template)
    dr = similar(x_template)
    jvp_out = similar(x_template)
    return MALAWorkspace(g_x, g_y, y, μ, r, Hv_x, Hv_y, w, dr, jvp_out)
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
        tmp = _apply_M(gradlogp_x, cholM)
        @. ws.μ = x + ϵ * tmp
    end
    @. ws.r = y - ws.μ

    return -T(0.5) * _quad_Minv(ws.r, cholM) / (2ϵ) - (T(d) / 2) * log(T(4π) * ϵ) -
           T(0.5) * _logdet_M(cholM)
end

function logq_mala(
    y::AbstractVector,
    x::AbstractVector,
    gradlogp_x::AbstractVector,
    ϵ::Real;
    cholM=nothing,
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
        Mgx = _apply_M(ws.g_x, cholM)
        Lξ = _apply_L(ξ, cholM)
        @. y = x + ϵ * Mgx + sqrt2ϵ * Lξ
    end
    return y
end

function mala_proposal(
    logp,
    gradlogp,
    x::AbstractVector,
    ϵ::Real,
    ξ::AbstractVector;
    cholM=nothing,
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
    logp,
    gradlogp,
    x::AbstractVector,
    y::AbstractVector,
    ϵ::Real;
    cholM=nothing,
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
    logp,
    gradlogp,
    x::AbstractVector,
    ϵ::Real,
    ξ::AbstractVector,
    u::Real;
    cholM=nothing,
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
    logp,
    gradlogp,
    x::AbstractVector,
    ϵ::Real,
    ξ::AbstractVector,
    u::Real;
    cholM=nothing,
)
    ws = MALAWorkspace(x)
    mala_proposal!(ws.y, ws, logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    logα = mala_logα!(ws, logp, gradlogp, x, ws.y, ϵ; cholM=cholM)
    FP = typeof(float(u))
    return (log(u) < logα) ? one(FP) : zero(FP)
end

function mala_step_full(
    logp,
    gradlogp,
    x::AbstractVector,
    ϵ::Real,
    ξ::AbstractVector,
    u::Real;
    cholM=nothing,
)
    x_next, accepted, _ = mala_step_with_logα(logp, gradlogp, x, ϵ, ξ, u; cholM=cholM)
    return x_next, accepted
end

function mala_step_with_logα(
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

    ws = MALAWorkspace(x)
    mala_proposal!(ws.y, ws, logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    logp_x = logp(x)
    logp_y = logp(ws.y)
    copyto!(ws.g_y, gradlogp(ws.y))

    logq_yx = logq_mala!(ws, ws.y, x, ws.g_x, ϵ; cholM=cholM)
    logq_xy = logq_mala!(ws, x, ws.y, ws.g_y, ϵ; cholM=cholM)
    logα = (logp_y + logq_xy) - (logp_x + logq_yx)

    accepted = log(u) < logα
    x_next = accepted ? copy(ws.y) : copy(x)
    return x_next, accepted, logα
end

"""
Stop-gradient surrogate step used for Jacobians.
"""
function mala_step_surrogate_sigmoid(
    logp,
    gradlogp,
    x::AbstractVector,
    ϵ::Real,
    ξ::AbstractVector,
    u::Real;
    cholM=nothing,
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
_apply_M_batched(G::AbstractMatrix, ::Nothing) = G
_apply_M_batched(G::AbstractMatrix, cholM::Cholesky) = cholM.L * (cholM.L' * G)

_apply_L_batched(Ξ::AbstractMatrix, ::Nothing) = Ξ
_apply_L_batched(Ξ::AbstractMatrix, cholM::Cholesky) = cholM.L * Ξ

function _quad_Minv_batched(R::AbstractMatrix, ::Nothing)
    return vec(sum(abs2, R; dims=1))
end

function _quad_Minv_batched(R::AbstractMatrix, cholM::Cholesky)
    W = cholM.L \ R
    return vec(sum(abs2, W; dims=1))
end

function logq_mala_batched(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    gradlogp_X::AbstractMatrix,
    ε::Real;
    cholM=nothing,
)
    T = typeof(ε)
    D = size(X, 1)
    μ = X .+ ε .* _apply_M_batched(gradlogp_X, cholM)
    R = Y .- μ
    q = _quad_Minv_batched(R, cholM)
    ldet = _logdet_M(cholM)
    return @. -T(0.5) * q / (2ε) - (T(D) / 2) * log(T(4π) * ε) - T(0.5) * ldet
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

    ε_T = eltype(X)(ε)

    G_X = gradlogp_batch(X)
    Y = X .+ ε_T .* _apply_M_batched(G_X, cholM) .+ sqrt(2 * ε_T) .* _apply_L_batched(Ξ, cholM)

    lp_X = logp_batch(X)
    lp_Y = logp_batch(Y)
    G_Y = gradlogp_batch(Y)

    lq_YX = logq_mala_batched(Y, X, G_X, ε_T; cholM=cholM)
    lq_XY = logq_mala_batched(X, Y, G_Y, ε_T; cholM=cholM)

    logα = @. (lp_Y + lq_XY) - (lp_X + lq_YX)
    accepted = @. log(u) < logα

    mask = reshape(accepted, 1, N)
    X_next = @. ifelse(mask, Y, X)
    return X_next, vec(accepted)
end

"""
Batched fused surrogate forward step and JVP over columns.

`X`, `Ξ`, and `Z` are all D×T, and `u` has length T.
Returns `(FT, Jt)` where both outputs are D×T.
"""
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
    D, T = size(X)
    size(Ξ) == (D, T) || throw(DimensionMismatch("X and Ξ must have the same size"))
    size(Z) == (D, T) || throw(DimensionMismatch("X and Z must have the same size"))
    length(u) == T || throw(DimensionMismatch("u must have length T"))

    εT = convert(eltype(X), ε)
    oneT = one(eltype(X))
    sqrt2εT = sqrt(2 * εT)

    G_X = gradlogp_batch(X)
    Y = X .+ εT .* _apply_M_batched(G_X, cholM) .+ sqrt2εT .* _apply_L_batched(Ξ, cholM)

    lp_X = logp_batch(X)
    lp_Y = logp_batch(Y)
    G_Y = gradlogp_batch(Y)

    lq_YX = logq_mala_batched(Y, X, G_X, εT; cholM=cholM)
    lq_XY = logq_mala_batched(X, Y, G_Y, εT; cholM=cholM)
    logα = @. (lp_Y + lq_XY) - (lp_X + lq_YX)

    u_like = similar(logα, eltype(logα), T)
    copyto!(u_like, u)
    logu = log.(u_like)

    g̃ = logα .- logu
    g = @. oneT / (oneT + exp(-g̃))
    g_row = reshape(g, 1, T)

    accepted = @. logu < logα
    accepted_row = reshape(accepted, 1, T)
    FT = @. ifelse(accepted_row, Y, X)

    Hv_X = hvp_batch(X, Z)
    W = Z .+ εT .* _apply_M_batched(Hv_X, cholM)

    Hv_Y = hvp_batch(Y, W)
    R = X .- Y .- εT .* _apply_M_batched(G_Y, cholM)
    dR = Z .- W .- εT .* _apply_M_batched(Hv_Y, cholM)

    Minv_R = cholM === nothing ? R : (cholM \ R)

    dlogα = vec(sum(G_Y .* W; dims=1)) .-
            vec(sum(G_X .* Z; dims=1)) .-
            inv(2 * εT) .* vec(sum(Minv_R .* dR; dims=1))

    dg = g .* (oneT .- g) .* dlogα
    dg_row = reshape(dg, 1, T)

    Jt = @. ifelse(accepted_row, W, Z) + (Y - X) * dg_row
    return FT, Jt
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
        MHv_x = _apply_M(ws.Hv_x, cholM)
        @. ws.w = v + ε * MHv_x
    end

    copyto!(ws.Hv_y, hvp_fn(ws.y, ws.w))
    if cholM === nothing
        @. ws.r = x - ws.y - ε * ws.g_y
        @. ws.dr = v - ws.w - ε * ws.Hv_y
        Minv_r = ws.r
    else
        Mg_y = _apply_M(ws.g_y, cholM)
        MHv_y = _apply_M(ws.Hv_y, cholM)
        @. ws.r = x - ws.y - ε * Mg_y
        @. ws.dr = v - ws.w - ε * MHv_y
        Minv_r = cholM \ ws.r
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
        MHv_x = _apply_M(ws.Hv_x, cholM)
        @. ws.w = v + ε * MHv_x
    end

    copyto!(ws.Hv_y, hvp_fn(ws.y, ws.w))
    if cholM === nothing
        @. ws.r = x - ws.y - ε * ws.g_y
        @. ws.dr = v - ws.w - ε * ws.Hv_y
        Minv_r = ws.r
    else
        Mg_y = _apply_M(ws.g_y, cholM)
        MHv_y = _apply_M(ws.Hv_y, cholM)
        @. ws.r = x - ws.y - ε * Mg_y
        @. ws.dr = v - ws.w - ε * MHv_y
        Minv_r = cholM \ ws.r
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
        x_next,
        jvp_out,
        ws,
        logp,
        gradlogp,
        x,
        ε,
        ξ,
        u,
        v,
        hvp_fn;
        cholM=cholM,
    )
    return x_next, jvp_out
end

end # module MALA
