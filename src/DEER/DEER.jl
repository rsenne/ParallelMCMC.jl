module DEER

using Base.Threads: @threads, nthreads, threadid
using LinearAlgebra
using DifferentiationInterface
import Mooncake: Mooncake

const DI = DifferentiationInterface
backend = DI.AutoMooncake(; config=nothing)

struct TapedRecursion{F,Tt}
    step::F
    tape::Vector{Tt}
end

# Stable callable for DI
struct StepWithTape{R}
    rec::R
end
(s::StepWithTape)(x, tape) = s.rec.step(x, tape)

"Prepare Jacobian for this recursion (reusable across t as long as tape element type is stable)."
function prepare(rec::TapedRecursion, x0::AbstractVector)
    f = StepWithTape(rec)
    return DI.prepare_jacobian(f, backend, x0, DI.Constant(rec.tape[1]))
end

"Full Jacobian of step(x, tape_t) w.r.t. x."
function jac_full(rec::TapedRecursion, prep, x::AbstractVector, t::Int)
    f = StepWithTape(rec)
    return DI.jacobian(f, prep, backend, x, DI.Constant(rec.tape[t]))
end

"Diagonal Jacobian: currently computed via full Jacobian then diag."
function jac_diag(rec::TapedRecursion, prep, x::AbstractVector, t::Int)
    return diag(jac_full(rec, prep, x, t))
end

"""
Solve s_t = a_t .* s_{t-1} + b_t for t=1..T using an associative inclusive scan.

Inputs:
- A :: D×T matrix, A[:,t] = a_t
- B :: D×T matrix, B[:,t] = b_t
- s0 :: length-D vector

Returns:
- S :: D×T matrix, S[:,t] = s_t
"""
function solve_affine_scan_diag(A::AbstractMatrix, B::AbstractMatrix, s0::AbstractVector)
    D, T = size(A)
    size(B) == (D, T) || throw(ArgumentError("B must have the same size as A"))
    length(s0) == D || throw(ArgumentError("s0 length must match size(A,1)"))

    α = Matrix{Float64}(A)         # working prefixes
    β = Matrix{Float64}(B)
    αnew = similar(α)
    βnew = similar(β)

    offset = 1
    while offset < T
        @threads for t in 1:T
            if t > offset
                @inbounds for i in 1:D
                    ai = α[i, t]
                    αnew[i, t] = ai * α[i, t - offset]
                    βnew[i, t] = ai * β[i, t - offset] + β[i, t]
                end
            else
                @inbounds for i in 1:D
                    αnew[i, t] = α[i, t]
                    βnew[i, t] = β[i, t]
                end
            end
        end
        α, αnew = αnew, α
        β, βnew = βnew, β
        offset <<= 1
    end

    S = Matrix{Float64}(undef, D, T)
    @threads for t in 1:T
        @inbounds for i in 1:D
            S[i, t] = α[i, t] * s0[i] + β[i, t]
        end
    end
    return S
end

"""
Compute one DEER update given a trajectory guess S (D×T).
Returns S_new (D×T).

Keywords:
- jacobian: :diag or :full
- damping: in (0,1]
"""
function deer_update(
    rec::TapedRecursion,
    s0_in::AbstractVector,
    S::AbstractMatrix,
    prep;
    jacobian::Symbol = :diag,
    damping::Real = 1.0,
)
    s0 = vec(s0_in)
    D, T = size(S)
    length(s0) == D || throw(ArgumentError("S must be D×T with D=length(s0)"))
    T == length(rec.tape) || throw(ArgumentError("size(S,2) must match length(rec.tape)"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))

    if jacobian === :diag
        A = Matrix{Float64}(undef, D, T)
        B = Matrix{Float64}(undef, D, T)

        # scratch vector per thread
        xbufs = [zeros(Float64, D) for _ in 1:Base.Threads.maxthreadid()]

        # Build A,B in parallel across time using x̄_{t-1} = (t==1 ? s0 : S[:,t-1])
        @threads for t in 1:T
            xbar = if t == 1
                s0
            else
                xb = xbufs[threadid()]
                @inbounds for i in 1:D
                    xb[i] = S[i, t - 1]
                end
                xb
            end

            jt = jac_diag(rec, prep, xbar, t)
            ft = rec.step(xbar, rec.tape[t])

            @inbounds for i in 1:D
                A[i, t] = jt[i]
                B[i, t] = ft[i] - jt[i] * xbar[i]
            end
        end

        S_new = solve_affine_scan_diag(A, B, s0)

        if damping != 1.0
            # S_new = (1-d)*S + d*S_new
            @threads for t in 1:T
                for i in 1:D
                    S_new[i, t] = (1 - damping) * S[i, t] + damping * S_new[i, t]
                end
            end
        end

        return S_new

    elseif jacobian === :full
        A = Vector{Matrix{Float64}}(undef, T)
        b = Matrix{Float64}(undef, D, T)

        xbuf = zeros(Float64, D)  # single scratch vector (sequential, will add parallel scan alter)

        for t in 1:T
            if t == 1
                xbar = s0
            else
                @inbounds for i in 1:D
                    xbuf[i] = S[i, t - 1]
                end
                xbar = xbuf
            end

            Jt = jac_full(rec, prep, xbar, t)
            ft = rec.step(xbar, rec.tape[t])

            A[t] = Jt
            @inbounds for i in 1:D
                b[i, t] = ft[i] - (Jt * xbar)[i]   # see note below
            end
        end

        S_new = Matrix{Float64}(undef, D, T)
        s_prev = copy(s0)
        for t in 1:T
            s_prev = A[t] * s_prev .+ view(b, :, t)
            @inbounds S_new[:, t] .= s_prev
        end

        if damping != 1.0
            @inbounds S_new .= (1 - damping) .* S .+ damping .* S_new
        end

        return S_new
    else
        throw(ArgumentError("jacobian must be :diag or :full"))
    end
end

"""
Run DEER iterations until convergence.

Returns:
- S :: D×T matrix trajectory (S[:,t] is state at time t)

Keywords:
- init: initial trajectory guess, D×T (default: repeat s0)
- tol_abs, tol_rel: stopping tolerances (∞-norm per time step)
- maxiter
- jacobian: :diag or :full
- damping
- return_info
"""
function solve(
    rec::TapedRecursion,
    s0_in::AbstractVector;
    init = nothing,
    tol_abs::Real = 1e-4,
    tol_rel::Real = 1e-3,
    maxiter::Int = 200,
    jacobian::Symbol = :diag,
    damping::Real = 1.0,
    return_info::Bool = false,
)
    s0 = vec(s0_in)
    D = length(s0)
    T = length(rec.tape)

    maxiter ≥ 1 || throw(ArgumentError("maxiter must be ≥ 1"))
    tol_abs ≥ 0 || throw(ArgumentError("tol_abs must be ≥ 0"))
    tol_rel ≥ 0 || throw(ArgumentError("tol_rel must be ≥ 0"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))

    S = if init === nothing
        # repeat s0 across time
        M = Matrix{Float64}(undef, D, T)
        @threads for t in 1:T
            @inbounds M[:, t] .= s0
        end
        M
    else
        size(init) == (D, T) || throw(ArgumentError("init must be size (D,T)"))
        Matrix{Float64}(init)
    end

    prep = prepare(rec, s0)

    converged = false
    last_metric = Inf
    iters = 0

    for iter in 1:maxiter
        iters = iter
        S_new = deer_update(rec, s0, S, prep; jacobian=jacobian, damping=damping)

        metric = 0.0
        for t in 1:T
            Δ = maximum(abs.(view(S_new, :, t) .- view(S, :, t)))
            scale = tol_abs + tol_rel * maximum(abs.(view(S_new, :, t)))
            metric = max(metric, Δ / scale)
        end

        S = S_new
        last_metric = metric

        if metric ≤ 1.0
            converged = true
            break
        end
    end

    if return_info
        return S, (converged=converged, iters=iters, metric=last_metric, jacobian=jacobian, damping=damping)
    else
        return S
    end
end

end # module DEER
