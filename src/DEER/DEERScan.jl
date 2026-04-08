module DEERScan

export solve_affine_seq!, solve_affine_seq, solve_affine_scan_diag!, solve_affine_scan_diag,
       check_affine_scan, affine_scan_residual
       
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
    solve_affine_scan_diag!(S, A, B, s0; check_shapes=true)

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
"""
function solve_affine_scan_diag!(
    S::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    s0::AbstractVector;
    check_shapes::Bool=true,
)
    D, T = size(A)

    if check_shapes
        size(B) == (D, T) || throw(DimensionMismatch("B must have the same size as A"))
        size(S) == (D, T) || throw(DimensionMismatch("S must have size (D, T)"))
        length(s0) == D || throw(DimensionMismatch("length(s0) must equal size(A,1)"))
        Base.mightalias(S, A) && throw(ArgumentError("S must not alias A"))
        Base.mightalias(S, B) && throw(ArgumentError("S must not alias B"))
    end

    T == 0 && return S

    # Work buffers for the composed affine maps up to each time index.
    α = copy(A)
    β = copy(B)
    αnew = similar(α)
    βnew = similar(β)

    offset = 1
    while offset < T
        @views begin
            # Unchanged prefix.
            αnew[:, 1:offset] .= α[:, 1:offset]
            βnew[:, 1:offset] .= β[:, 1:offset]

            # Compose map at t with map at t-offset:
            #   αnew[:, t] = α[:, t] .* α[:, t-offset]
            #   βnew[:, t] = α[:, t] .* β[:, t-offset] .+ β[:, t]
            if offset < T
                αnew[:, (offset + 1):T] .= α[:, (offset + 1):T] .* α[:, 1:(T - offset)]
                βnew[:, (offset + 1):T] .=
                    α[:, (offset + 1):T] .* β[:, 1:(T - offset)] .+ β[:, (offset + 1):T]
            end
        end

        α, αnew = αnew, α
        β, βnew = βnew, β
        offset <<= 1
    end

    # Apply the initial condition:
    #   s_t = α[:, t] .* s0 + β[:, t]
    S .= α .* reshape(s0, D, 1) .+ β
    return S
end

function solve_affine_scan_diag(A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector)
    T = promote_type(eltype(A), eltype(B), eltype(s0))
    S = similar(A, T, size(A))
    return solve_affine_scan_diag!(S, A, B, s0)
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
