module DEER

using Base.Threads: @threads
using LinearAlgebra
using DifferentiationInterface
using Mooncake: Mooncake

const DI = DifferentiationInterface
backend = DI.AutoMooncake(; config=nothing)

struct TapedRecursion{F,Tt}
    step::F
    tape::Vector{Tt}
end

# Single non-closure function for DI
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
    f = StepWithTape(rec)  # same type as in prepare (StepWithTape{TapedRecursion{...}})
    return DI.jacobian(f, prep, backend, x, DI.Constant(rec.tape[t]))
end

"Diagonal of Jacobian: for now compute full Jacobian then take diag."
function jac_diag(rec::TapedRecursion, prep, x::AbstractVector, t::Int)
    J = jac_full(rec, prep, x, t)
    return diag(J)
end

"""
Solve s_t = a_t .* s_{t-1} + b_t using an associative inclusive scan
over the affine operators (a_t, b_t).

Inputs:
- a :: Vector{<:AbstractVector} length T, each a[t] length D
- b :: Vector{<:AbstractVector} length T, each b[t] length D
- s0 :: AbstractVector length D

Returns:
- s :: Vector{Vector} length T, s[t] is state at time t
"""
function solve_affine_scan_diag(
    a::Vector{<:AbstractVector}, b::Vector{<:AbstractVector}, s0::AbstractVector
)::Vector{Vector}
    T = length(a)
    T == length(b) || throw(ArgumentError("length(a) must match length(b)"))
    D = length(s0)
    D > 0 || throw(ArgumentError("state dimension must be positive"))
    for t in 1:T
        length(a[t]) == D || throw(ArgumentError("a[$t] has wrong length"))
        length(b[t]) == D || throw(ArgumentError("b[$t] has wrong length"))
    end

    # Working copies of prefix operators (α, β)
    α = [Vector{Float64}(a[t]) for t in 1:T]
    β = [Vector{Float64}(b[t]) for t in 1:T]

    # Temporary buffers for the scan stages
    αnew = [similar(α[t]) for t in 1:T]
    βnew = [similar(β[t]) for t in 1:T]

    # Hillis–Steele inclusive scan:
    # after each stage with offset, (α[t],β[t]) becomes composition with (α[t-offset],β[t-offset])
    offset = 1
    while offset < T
        @threads for t in 1:T
            if t > offset
                at = α[t]
                bt = β[t]
                ap = α[t - offset]
                bp = β[t - offset]
                αt = αnew[t]
                βt = βnew[t]
                @inbounds for i in 1:D
                    ai = at[i]
                    αt[i] = ai * ap[i]
                    βt[i] = ai * bp[i] + bt[i]
                end
            else
                copyto!(αnew[t], α[t])
                copyto!(βnew[t], β[t])
            end
        end
        α, αnew = αnew, α
        β, βnew = βnew, β
        offset <<= 1
    end

    # Apply prefix operators to s0: s[t] = α[t].*s0 + β[t]
    s = [similar(s0, Float64) for _ in 1:T]
    @threads for t in 1:T
        st = s[t]
        at = α[t]
        bt = β[t]
        for i in 1:D
            st[i] = at[i] * s0[i] + bt[i]
        end
    end

    return s
end

function deer_update(
    rec::TapedRecursion,
    s0::AbstractVector,
    s_guess::Vector,
    prep;
    jacobian::Symbol=:diag,
    damping::Real=1.0,
)
    T = length(s_guess)
    T == length(rec.tape) ||
        throw(ArgumentError("length(s_guess) must match length(rec.tape)"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0, 1]"))

    ref_prev = s0

    if jacobian === :full
        A = Vector{Matrix}(undef, T)
        b = Vector{Vector}(undef, T)

        for t in 1:T
            Jt = jac_full(rec, prep, ref_prev, t)
            ft = rec.step(ref_prev, rec.tape[t])
            ut = ft .- Jt * ref_prev
            A[t] = Jt
            b[t] = ut
            ref_prev = s_guess[t]
        end

        s_new = Vector{Vector}(undef, T)
        s_prev = copy(s0)
        for t in 1:T
            s_prev = A[t] * s_prev .+ b[t]
            s_new[t] = copy(s_prev)
        end

        if damping != 1.0
            for t in 1:T
                s_new[t] = (1 - damping) .* s_guess[t] .+ damping .* s_new[t]
            end
        end

        return s_new

    elseif jacobian === :diag
        a = Vector{Vector}(undef, T)
        b = Vector{Vector}(undef, T)

        for t in 1:T
            jt = jac_diag(rec, prep, ref_prev, t)
            ft = rec.step(ref_prev, rec.tape[t])
            ut = ft .- jt .* ref_prev
            a[t] = jt
            b[t] = ut
            ref_prev = s_guess[t]
        end

        s_new = solve_affine_scan_diag(a, b, s0)

        if damping != 1.0
            for t in 1:T
                s_new[t] = (1 - damping) .* s_guess[t] .+ damping .* s_new[t]
            end
        end

        return s_new
    else
        throw(ArgumentError("jacobian must be :diag or :full"))
    end
end

function solve(
    rec::TapedRecursion,
    s0_in;
    init=nothing,
    tol_abs::Real=1e-4,
    tol_rel::Real=1e-3,
    maxiter::Int=200,
    jacobian::Symbol=:diag,
    damping::Real=1.0,
    return_info::Bool=false,
)
    # Force 1D vector active argument (avoids Matrix corner cases)
    s0 = vec(s0_in)

    T = length(rec.tape)
    s = if init === nothing
        [copy(s0) for _ in 1:T]
    else
        length(init) == T || throw(ArgumentError("init must have length T"))
        init
    end

    prep = prepare(rec, s0)

    converged = false
    last_metric = Inf
    iters = 0

    for iter in 1:maxiter
        iters = iter
        s_new = deer_update(rec, s0, s, prep; jacobian=jacobian, damping=damping)

        metric = 0.0
        for t in 1:T
            Δ = maximum(abs.(s_new[t] .- s[t]))
            scale = tol_abs + tol_rel * maximum(abs.(s_new[t]))
            metric = max(metric, Δ / scale)
        end

        s = s_new
        last_metric = metric

        if metric ≤ 1.0
            converged = true
            break
        end
    end

    if return_info
        return s,
        (
            converged=converged,
            iters=iters,
            metric=last_metric,
            jacobian=jacobian,
            damping=damping,
        )
    else
        return s
    end
end

end # module DEER
