module MALA

using Random, LinearAlgebra

# Preconditioner dispatch helpers.  cholM is either nothing (identity) or a Cholesky factor.
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
Compute the log of the MALA proposal density q(y | x).

With mass matrix M (passed as `cholM = cholesky(M)`):
  y ~ Normal(x + ϵ M ∇logp(x), 2ϵ M)

With `cholM=nothing` (default), uses identity M = I.
"""
function logq_mala(
    y::AbstractVector, x::AbstractVector, gradlogp_x::AbstractVector, ϵ::Real;
    cholM=nothing,
)
    T = typeof(ϵ)
    μ = x .+ ϵ .* _apply_M(gradlogp_x, cholM)
    d = length(x)
    r = y .- μ
    return -T(0.5) * _quad_Minv(r, cholM) / (2ϵ) - (T(d) / 2) * log(T(4π) * ϵ) - T(0.5) * _logdet_M(cholM)
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
- cholM: optional Cholesky factor of mass matrix M (default: identity)

Returns:
- x_next
"""
function mala_step_taped(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real;
    cholM=nothing,
)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    0.0 < u < 1.0 || throw(ArgumentError("u must be in (0, 1)"))
    g_x = gradlogp(x)

    y = x .+ ϵ .* _apply_M(g_x, cholM) .+ sqrt(2ϵ) .* _apply_L(ξ, cholM)

    logp_x = logp(x)
    logp_y = logp(y)

    g_y = gradlogp(y)
    logq_y_given_x = logq_mala(y, x, g_x, ϵ; cholM=cholM)
    logq_x_given_y = logq_mala(x, y, g_y, ϵ; cholM=cholM)

    logα = (logp_y + logq_x_given_y) - (logp_x + logq_y_given_x)

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
    us::AbstractVector;
    cholM=nothing,
)
    T = length(us)
    length(ξs) == T || throw(DimensionMismatch("ξs and us must have the same length"))
    xs = Vector{typeof(x0)}(undef, T + 1)
    xs[1] = copy(x0)
    x = copy(x0)
    for t in 1:T
        x = mala_step_taped(logp, gradlogp, x, ϵ, ξs[t], us[t]; cholM=cholM)
        xs[t + 1] = copy(x)
    end
    return xs
end

"Compute the MALA proposal map y = x + ϵ M ∇logp(x) + √(2ϵ) L ξ."
function mala_proposal(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector; cholM=nothing,
)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    g_x = gradlogp(x)
    return x .+ ϵ .* _apply_M(g_x, cholM) .+ sqrt(2ϵ) .* _apply_L(ξ, cholM)
end

"Compute log acceptance ratio logα(x→y) for MALA."
function mala_logα(
    logp, gradlogp, x::AbstractVector, y::AbstractVector, ϵ::Real; cholM=nothing,
)
    g_x = gradlogp(x)
    g_y = gradlogp(y)
    logp_x = logp(x)
    logp_y = logp(y)
    logq_y_given_x = logq_mala(y, x, g_x, ϵ; cholM=cholM)
    logq_x_given_y = logq_mala(x, y, g_y, ϵ; cholM=cholM)
    return (logp_y + logq_x_given_y) - (logp_x + logq_y_given_x)
end

"""
Primal accept indicator for a taped MALA step.
Returns a float in {0, 1} matching the precision of `u`, for use as a constant
gate in the DEER surrogate step.
"""
function mala_accept_indicator(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real;
    cholM=nothing,
)
    y    = mala_proposal(logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    logα = mala_logα(logp, gradlogp, x, y, ϵ; cholM=cholM)
    FP   = typeof(float(u))
    return (log(u) < logα) ? one(FP) : zero(FP)
end

"""
One taped MALA step, returning both the next state and the accept flag.

This is the efficient entry point: it evaluates `gradlogp` exactly twice (once at `x`,
once at the proposal `y`), compared to calling `mala_accept_indicator` + `mala_step_taped`
separately which evaluates it five times.

Returns `(x_next, accepted::Bool)`.
"""
function mala_step_full(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real;
    cholM=nothing,
)
    x_next, accepted, _ = mala_step_with_logα(logp, gradlogp, x, ϵ, ξ, u; cholM=cholM)
    return x_next, accepted
end

"""
One taped MALA step, returning the next state, the accept flag, **and** the raw
log acceptance ratio `logα = log p(y) + log q(x|y) - log p(x) - log q(y|x)`.

The returned `logα` is the un-clamped value; the actual acceptance probability is
`min(1, exp(logα))`.  This is needed by adaptive step-size schemes (dual averaging).

Returns `(x_next, accepted::Bool, logα)`.
"""
function mala_step_with_logα(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real;
    cholM=nothing,
)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    0.0 < u < 1.0 || throw(ArgumentError("u must be in (0, 1)"))

    g_x = gradlogp(x)
    y   = x .+ ϵ .* _apply_M(g_x, cholM) .+ sqrt(2ϵ) .* _apply_L(ξ, cholM)

    logp_x = logp(x)
    logp_y = logp(y)
    g_y    = gradlogp(y)

    logq_y_given_x = logq_mala(y, x, g_x, ϵ; cholM=cholM)
    logq_x_given_y = logq_mala(x, y, g_y, ϵ; cholM=cholM)
    logα           = (logp_y + logq_x_given_y) - (logp_x + logq_y_given_x)

    accepted = log(u) < logα
    x_next = accepted ? y : x
    return x_next, accepted, logα
end

"""
Stop-gradient surrogate step used for Jacobians.
`a` (0 or 1) must be provided as a constant by the DEER machinery.
"""
function mala_step_surrogate(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, a::Real;
    cholM=nothing,
)
    y = mala_proposal(logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    return (a .* y) .+ ((1 - a) .* x)
end

# Apply mass matrix to a D×N matrix of gradient columns (same math as scalar,
# matrix multiply broadcasts naturally).
_apply_M_batched(G::AbstractMatrix, ::Nothing) = G
_apply_M_batched(G::AbstractMatrix, cholM::Cholesky) = cholM.L * (cholM.L' * G)

_apply_L_batched(Ξ::AbstractMatrix, ::Nothing) = Ξ
_apply_L_batched(Ξ::AbstractMatrix, cholM::Cholesky) = cholM.L * Ξ

"""
Compute column-wise M⁻¹-norm squared: `[||R[:,n]||²_{M⁻¹}]_n`.
`R` is D×N; returns a length-N vector.
GPU-compatible (uses `sum(abs2, …; dims=1)` which works on CuArrays).
Note: `cholM` must be `nothing` when using GPU arrays; the triangular solve
`cholM.L \\ R` pulls device arrays to CPU.
"""
function _quad_Minv_batched(R::AbstractMatrix, ::Nothing)
    return vec(sum(abs2, R; dims=1))
end

function _quad_Minv_batched(R::AbstractMatrix, cholM::Cholesky)
    W = cholM.L \ R
    return vec(sum(abs2, W; dims=1))
end

"""
    logq_mala_batched(Y, X, gradlogp_X, ε; cholM=nothing)

Compute `log q(Y[:,n] | X[:,n])` for all N chains simultaneously.
`Y`, `X`, `gradlogp_X` are D×N; returns a length-N vector.
"""
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

"""
    mala_step_batched(logp_batch, gradlogp_batch, X, ε, Ξ, u; cholM=nothing)

Run one MALA step for N chains simultaneously.

- `X` :: D×N — current states (one chain per column).
- `Ξ` :: D×N — N(0,I) noise.
- `u` :: length-N — Uniform(0,1) draws.
- `logp_batch(X)` → length-N log-densities.
- `gradlogp_batch(X)` → D×N gradient matrix.

Returns `(X_next::AbstractMatrix, accepted::AbstractVector)`.

**GPU use:** pass `CuArray` inputs and GPU-compatible `logp_batch`/`gradlogp_batch`.
Requires `cholM=nothing` for full on-device execution (Cholesky preconditioner involves
a CPU-side triangular solve).  Use `eltype(X)` for `ε` to avoid float-type promotions
that would pull data off GPU.
"""
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
    length(u) == N    || throw(DimensionMismatch("u must have length N = size(X,2)"))

    # Cast ε to element type of X to avoid float-promotion off GPU.
    ε_T = eltype(X)(ε)

    G_X = gradlogp_batch(X)                                                 # D×N
    Y   = X .+ ε_T .* _apply_M_batched(G_X, cholM) .+
          sqrt(2 * ε_T) .* _apply_L_batched(Ξ, cholM)                      # D×N

    lp_X = logp_batch(X)                                                    # N
    lp_Y = logp_batch(Y)                                                    # N
    G_Y  = gradlogp_batch(Y)                                                # D×N

    lq_YX = logq_mala_batched(Y, X, G_X, ε_T; cholM=cholM)                # N
    lq_XY = logq_mala_batched(X, Y, G_Y, ε_T; cholM=cholM)                # N

    logα = @. (lp_Y + lq_XY) - (lp_X + lq_YX)                            # N
    accepted = @. log(u) < logα                                             # N Bool

    # Select: proposal if accepted, current if rejected.
    # reshape to 1×N so it broadcasts against D×N.
    mask   = reshape(accepted, 1, N)
    X_next = @. ifelse(mask, Y, X)                                          # D×N
    return X_next, vec(accepted)
end

end # module
