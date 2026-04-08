module DEERScan

export AffineScanWorkspace,
       solve_affine_seq!, solve_affine_seq,
       solve_affine_scan_diag!, solve_affine_scan_diag,
       check_affine_scan, affine_scan_residual

"""
Reusable workspace for the diagonal affine scan.

Buffers are allocated with the same array type / device placement as the template
matrix used to construct the workspace, so this is GPU-compatible for `CuArray`
inputs and CPU-compatible for standard arrays.
"""
struct AffineScanWorkspace{M}
    alpha::M
    beta::M
    alpha_new::M
    beta_new::M
end

function AffineScanWorkspace(A_template::AbstractMatrix)
    alpha = similar(A_template)
    beta = similar(A_template)
    alpha_new = similar(A_template)
    beta_new = similar(A_template)
    return AffineScanWorkspace(alpha, beta, alpha_new, beta_new)
end

"""
    solve_affine_seq!(S, A, B, s0)

Reference sequential solver for the diagonal affine recurrence

    s_t = A[:, t] .* s_{t-1} + B[:, t],   t = 1, ..., T

with initial condition `s_0 = s0`.

Inputs:
- `A :: D×T`
- `B :: D×T`
- `s0 :: length-D`
- `S :: D×T` output buffer

This is the ground-truth implementation to compare against the parallel scan path.
It is intentionally simple and should be used in tests.
"""
function solve_affine_seq!(S::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector)
    D, T = size(A)
    size(B) == (D, T) || throw(DimensionMismatch("B must have the same size as A"))
    size(S) == (D, T) || throw(DimensionMismatch("S must have size (D, T)"))
    length(s0) == D || throw(DimensionMismatch("length(s0) must equal size(A,1)"))

    T == 0 && return S

    @views begin
        S[:, 1] .= A[:, 1] .* s0 .+ B[:, 1]
        for t in 2:T
            S[:, t] .= A[:, t] .* S[:, t - 1] .+ B[:, t]
        end
    end
    return S
end

function solve_affine_seq(A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector)
    S = similar(A, promote_type(eltype(A), eltype(B), eltype(s0)), size(A))
    return solve_affine_seq!(S, A, B, s0)
end

"""
    solve_affine_scan_diag!(S, A, B, s0, ws; check_shapes=true)

Parallel-doubling solver for the diagonal affine recurrence

    s_t = A[:, t] .* s_{t-1} + B[:, t],   t = 1, ..., T

This implements an associative scan over affine maps

    x ↦ a .* x + b

using the composition rule

    (a2, b2) ∘ (a1, b1) = (a2 .* a1, a2 .* b1 .+ b2)

The implementation is generic over `AbstractMatrix` and is intended to work on
CPU arrays and GPU arrays (`CuArray`) as long as broadcasted assignment and
slicing/views are supported by the array type.

Notes:
- This is a clean, backend-agnostic implementation, not a backend-specialized
  scan kernel.
- It performs O(log T) sweep levels, each level consisting of whole-array
  broadcast operations.
- `S` may alias neither `A` nor `B`.
- `ws` allows the scan to run without warm-path allocations.
"""
function solve_affine_scan_diag!(
    S::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    s0::AbstractVector,
    ws::AffineScanWorkspace;
    check_shapes::Bool=true,
)
    D, T = size(A)

    if check_shapes
        size(B) == (D, T) || throw(DimensionMismatch("B must have the same size as A"))
        size(S) == (D, T) || throw(DimensionMismatch("S must have size (D, T)"))
        length(s0) == D || throw(DimensionMismatch("length(s0) must equal size(A,1)"))
        size(ws.alpha) == (D, T) || throw(DimensionMismatch("workspace has wrong matrix size"))
        size(ws.beta) == (D, T) || throw(DimensionMismatch("workspace has wrong matrix size"))
        size(ws.alpha_new) == (D, T) || throw(DimensionMismatch("workspace has wrong matrix size"))
        size(ws.beta_new) == (D, T) || throw(DimensionMismatch("workspace has wrong matrix size"))
        Base.mightalias(S, A) && throw(ArgumentError("S must not alias A"))
        Base.mightalias(S, B) && throw(ArgumentError("S must not alias B"))
    end

    T == 0 && return S

    alpha = ws.alpha
    beta = ws.beta
    alpha_new = ws.alpha_new
    beta_new = ws.beta_new

    copyto!(alpha, A)
    copyto!(beta, B)

    offset = 1
    while offset < T
        @views begin
            alpha_new[:, 1:offset] .= alpha[:, 1:offset]
            beta_new[:, 1:offset] .= beta[:, 1:offset]

            if offset < T
                alpha_new[:, (offset + 1):T] .= alpha[:, (offset + 1):T] .* alpha[:, 1:(T - offset)]
                beta_new[:, (offset + 1):T] .=
                    alpha[:, (offset + 1):T] .* beta[:, 1:(T - offset)] .+ beta[:, (offset + 1):T]
            end
        end

        alpha, alpha_new = alpha_new, alpha
        beta, beta_new = beta_new, beta
        offset <<= 1
    end

    S .= alpha .* reshape(s0, D, 1) .+ beta
    return S
end

function solve_affine_scan_diag!(
    S::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    s0::AbstractVector;
    check_shapes::Bool=true,
)
    ws = AffineScanWorkspace(A)
    return solve_affine_scan_diag!(S, A, B, s0, ws; check_shapes=check_shapes)
end

function solve_affine_scan_diag(A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector)
    T = promote_type(eltype(A), eltype(B), eltype(s0))
    S = similar(A, T, size(A))
    ws = AffineScanWorkspace(A)
    return solve_affine_scan_diag!(S, A, B, s0, ws)
end

"""
    affine_scan_residual(S, A, B, s0)

Return the maximum absolute residual of the recurrence

    S[:,1]      ?= A[:,1] .* s0       .+ B[:,1]
    S[:,t]      ?= A[:,t] .* S[:,t-1] .+ B[:,t],   t ≥ 2

Useful for debugging scan correctness independent of DEER.
"""
function affine_scan_residual(S::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector)
    D, T = size(A)
    size(B) == (D, T) || throw(DimensionMismatch("B must have the same size as A"))
    size(S) == (D, T) || throw(DimensionMismatch("S must have size (D, T)"))
    length(s0) == D || throw(DimensionMismatch("length(s0) must equal size(A,1)"))

    T == 0 && return zero(promote_type(eltype(S), eltype(A), eltype(B), eltype(s0)))

    r = maximum(abs.(S[:, 1] .- (A[:, 1] .* s0 .+ B[:, 1])))
    @views for t in 2:T
        rt = maximum(abs.(S[:, t] .- (A[:, t] .* S[:, t - 1] .+ B[:, t])))
        r = max(r, rt)
    end
    return r
end

"""
    check_affine_scan(A, B, s0; atol=1e-6, rtol=1e-6)

Compare the sequential reference solver against the parallel-doubling solver.
Returns a named tuple with:
- `ok`
- `max_abs_err`
- `max_rel_err`
- `residual_seq`
- `residual_scan`
- `S_seq`
- `S_scan`

This is intended as a lightweight validation helper when plugging the scan into DEER.
"""
function check_affine_scan(A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector; atol=1e-6, rtol=1e-6)
    S_seq = solve_affine_seq(A, B, s0)
    S_scan = solve_affine_scan_diag(A, B, s0)

    diff = abs.(S_seq .- S_scan)
    scale = atol .+ rtol .* abs.(S_seq)
    max_abs_err = maximum(diff)
    max_rel_err = maximum(diff ./ scale)
    residual_seq = affine_scan_residual(S_seq, A, B, s0)
    residual_scan = affine_scan_residual(S_scan, A, B, s0)

    return (
        ok = all(diff .<= scale),
        max_abs_err = max_abs_err,
        max_rel_err = max_rel_err,
        residual_seq = residual_seq,
        residual_scan = residual_scan,
        S_seq = S_seq,
        S_scan = S_scan,
    )
end

end # module
