module DEER

using LinearAlgebra
using DifferentiationInterface
import Mooncake

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

function deer_update(
    rec::TapedRecursion,
    s0::AbstractVector,
    s_guess::Vector,
    prep;
    jacobian::Symbol = :diag,
    damping::Real = 1.0,
)
    T = length(s_guess)
    T == length(rec.tape) || throw(ArgumentError("length(s_guess) must match length(rec.tape)"))
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

        s_new = Vector{Vector}(undef, T)
        s_prev = copy(s0)
        for t in 1:T
            s_prev = a[t] .* s_prev .+ b[t]
            s_new[t] = copy(s_prev)
        end

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
    init = nothing,
    tol_abs::Real = 1e-4,
    tol_rel::Real = 1e-3,
    maxiter::Int = 200,
    jacobian::Symbol = :diag,
    damping::Real = 1.0,
    return_info::Bool = false,
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
        return s, (converged=converged, iters=iters, metric=last_metric, jacobian=jacobian, damping=damping)
    else
        return s
    end
end

end # module
