module DEER

using LinearAlgebra
using DifferentiationInterface
using ADTypes: ADTypes, AbstractADType
using Random

import ..ParallelMCMC: needs_host_staging

include("DEERScan.jl")
using .DEERScan

const DI = DifferentiationInterface

export TapedRecursion, DEERWorkspace, deer_update!, deer_update, solve

"""
Deterministic recursion driven by a pre-generated tape.

- `step_fwd(x, tape_t)`: exact forward transition (may include MH accept/reject).
- `jvp(x, tape_t, v)`: explicit JVP `(x, tape_t, v) -> J·v` of the differentiable
  surrogate step. Must not call AD through `step_fwd`'s control flow.
- `fwd_and_jvp(x, tape_t, v)`: optional fused function `-> (x_next, J·v)`.
- `fwd_and_jvp_batch(Xbar, Z)`: optional batched function `-> (FT, Jt)` over all
  steps at once, column `t` of each matrix belonging to step `t`.

DEER always works on whole trajectories. Without `fwd_and_jvp_batch` it calls the
per-step functions one column at a time, so `step_fwd` and `jvp` are required then
and may be `nothing` otherwise.
"""
struct TapedRecursion{Ff,Fj,Ffj,Ffb,V<:AbstractVector}
    step_fwd::Ff
    jvp::Fj
    fwd_and_jvp::Ffj
    fwd_and_jvp_batch::Ffb
    tape::V
end

function TapedRecursion(
    step_fwd, jvp, tape::AbstractVector; fwd_and_jvp=nothing, fwd_and_jvp_batch=nothing
)
    fwd_and_jvp_batch === nothing &&
        (jvp === nothing || (step_fwd === nothing && fwd_and_jvp === nothing)) &&
        throw(
            ArgumentError(
                "TapedRecursion needs `fwd_and_jvp_batch`, or `jvp` with `step_fwd` " *
                "or `fwd_and_jvp`",
            ),
        )
    return TapedRecursion(step_fwd, jvp, fwd_and_jvp, fwd_and_jvp_batch, tape)
end

"""
Reusable buffers for allocation-light DEER updates and solves.

All buffers are created with the same array type / device placement as `S_template`
(or `s0_template` for vector buffers), so the workspace is GPU-compatible when
constructed from device-array templates.

`probe_rng` is a random number generator on the device of `S_template`, or
`nothing`. If it is set, the batched Hutchinson probes are drawn on the device,
seeded from the caller's `rng`; otherwise they are drawn on the host.
"""
struct DEERWorkspace{M,V,SW,HZ,R}
    A::M
    B::M
    Xbar::M
    Z::M
    Zhost::HZ
    S_work::M
    S_tmp::M
    zbuf::V
    xbar_buf::V
    scan::SW
    probe_rng::R
end

"""
    _device_probe_rng(Z) -> rng or nothing

A random number generator on the device of `Z`, so Hutchinson probes need no
copy from the host. Package extensions define it for their array types; the
default `nothing` keeps the probes on the host.
"""
_device_probe_rng(::AbstractArray) = nothing

# Fill `Z` with Rademacher probes, on the device when the workspace has a device RNG.
function _probe_matrix!(Z::AbstractMatrix, rng::AbstractRNG, ws)
    return _probe_matrix!(Z, rng, ws.Zhost, ws.probe_rng)
end
function _probe_matrix!(Z::AbstractMatrix, rng::AbstractRNG, host, ::Nothing)
    return _rademacher_matrix!(Z, rng, host)
end

"""
    _max_abs_change(S_new, S) -> (maximum(abs.(S_new .- S)), maximum(abs.(S_new)))

Both maxima of the convergence test in one pass, so a device method needs only one
reduction and one copy to the host.
"""
function _max_abs_change(S_new::AbstractArray{T}, S::AbstractArray) where {T}
    size(S_new) == size(S) ||
        throw(DimensionMismatch("S_new and S must have the same size"))
    return mapreduce(
        (a, b) -> (abs(a - b), abs(a)),
        (x, y) -> (max(x[1], y[1]), max(x[2], y[2])),
        S_new,
        S;
        init=(zero(real(T)), zero(real(T))),
    )
end

function DEERWorkspace(
    S_template::AbstractMatrix,
    s0_template::AbstractVector;
    probe_rng=_device_probe_rng(S_template),
)
    A = similar(S_template)
    B = similar(S_template)
    Xbar = similar(S_template)
    Z = similar(S_template)
    Zhost = if probe_rng === nothing && needs_host_staging(Z)
        Matrix{eltype(S_template)}(undef, size(Z)...)
    else
        nothing
    end
    S_work = similar(S_template)
    S_tmp = similar(S_template)
    # Column buffers for recursions given per step; see `_add_probe!`.
    zbuf = similar(s0_template)
    xbar_buf = similar(s0_template)
    scan = DEERScan.AffineScanWorkspace(S_template)
    return DEERWorkspace(
        A, B, Xbar, Z, Zhost, S_work, S_tmp, zbuf, xbar_buf, scan, probe_rng
    )
end

function DEERWorkspace(s0_template::AbstractVector, T::Integer)
    D = length(s0_template)
    S_template = similar(s0_template, D, T)
    return DEERWorkspace(S_template, s0_template)
end

function _prepare_hvp(f, backend::AbstractADType, x_template::AbstractVector)
    v_template = similar(x_template)
    fill!(v_template, zero(eltype(x_template)))
    return DI.prepare_pushforward(
        f, _normalized_backend(backend), x_template, (v_template,)
    )
end

function _materialize_ad_array(x::AbstractArray)
    return x isa SubArray ? copy(x) : x
end

_materialize_ad_vector(x::AbstractVector) = _materialize_ad_array(x)
_materialize_ad_matrix(x::AbstractMatrix) = _materialize_ad_array(x)

function _tangent_like(x::AbstractArray, v::AbstractArray)
    typeof(v) === typeof(x) && return v
    v_copy = similar(x)
    copyto!(v_copy, v)
    return v_copy
end

function _hvp_prepared(
    f, prep, backend::AbstractADType, x::AbstractVector, v::AbstractVector
)
    x_exec = _materialize_ad_vector(x)
    v_exec = _tangent_like(x_exec, v)
    res = DI.pushforward(f, prep, _normalized_backend(backend), x_exec, (v_exec,))
    return res isa Tuple ? first(res) : res
end

function _hvp_nopre(f, backend::AbstractADType, x::AbstractVector, v::AbstractVector)
    x_exec = _materialize_ad_vector(x)
    v_exec = _tangent_like(x_exec, v)
    res = DI.pushforward(f, _normalized_backend(backend), x_exec, (v_exec,))
    return res isa Tuple ? first(res) : res
end

function _prepare_batch_hvp_from_grad(
    grad_batch, backend::AbstractADType, X_template::AbstractMatrix
)
    V_template = similar(X_template)
    fill!(V_template, zero(eltype(X_template)))
    return DI.prepare_pushforward(
        grad_batch, _normalized_backend(backend), X_template, (V_template,)
    )
end

function _batch_hvp_from_grad_prepared(
    grad_batch, prep, backend::AbstractADType, X::AbstractMatrix, V::AbstractMatrix
)
    X_exec = _materialize_ad_matrix(X)
    V_exec = _tangent_like(X_exec, V)
    res = DI.pushforward(grad_batch, prep, _normalized_backend(backend), X_exec, (V_exec,))
    return res isa Tuple ? first(res) : res
end

#=
---------------------------------------------------------------------------
Reverse-on-grad HVP. Computes Hv as the gradient of
`x -> pmcmc_dot(gradlogp(x), v)`, with `v` carried as a DI Constant context
so a single `prepare_gradient` covers all subsequent Newton steps. The
batched variant uses `B -> pmcmc_dotsum(grad_batch(B), V)` and exploits the
column-independence of the user's batched gradient — its gradient w.r.t. B
is the columnwise HVP.

The reductions go through `pmcmc_dot`/`pmcmc_dotsum` (rather than `dot`/`sum`)
so that on GPU the EnzymeExt reverse rules intercept them — otherwise Enzyme
reverse-mode hits the `cuMemcpyDtoHAsync_v2` gc-transition abort (see the
Enzyme rules in `ext/EnzymeExt.jl`).

We bundle the closure with the prep so `prepare_gradient` and `gradient`
see the same function instance (DI keys preparations on function identity).
---------------------------------------------------------------------------
=#
import ..ParallelMCMC: pmcmc_dot, pmcmc_dotsum, _REACTANT_LOAD_HINT

struct _HvpReverseClosure{F}
    grad::F
end
(c::_HvpReverseClosure)(x, v) = pmcmc_dot(c.grad(x), v)

struct _BatchHvpReverseClosure{F}
    grad_batch::F
end
(c::_BatchHvpReverseClosure)(X, V) = pmcmc_dotsum(c.grad_batch(X), V)

#=
Pick the AD-HVP fallback strategy from the user's backend. Both are one AD pass
over a hand-written `gradlogp`; an AD-derived gradient goes to
`_make_hvp_fn_second_order` instead.

  ForwardOnGrad()   — `pushforward(gradlogp, x, v)`. Routes through the
                      `pmcmc_matmul` frule.
  ReverseOnGrad()   — `gradient(x -> pmcmc_dot(gradlogp(x), v))`. Routes
                      through the matmul and dot/sum rrules.

These are singleton types rather than symbols so the choice dispatches
statically — `_make_hvp_fn(_hvp_strategy(backend), ...)` resolves to one
concrete method (and one concrete return type) at compile time, without
relying on constant propagation through `===`.

The routing follows DI's `hvp_mode`: a forward outer pass
(`DI.ForwardOverAnything`) takes `ForwardOnGrad`, anything else
`ReverseOnGrad`. Only the outer direction matters since we differentiate
the already-built `gradlogp`. The mode is whatever the user's backend carries;
nothing here substitutes one (see `_normalized_backend`).
=#
abstract type HVPStrategy end
struct ForwardOnGrad <: HVPStrategy end
struct ReverseOnGrad <: HVPStrategy end

struct ReactantHVP <: HVPStrategy end

_strategy_from(::DI.ForwardOverAnything) = ForwardOnGrad()
_strategy_from(::DI.HVPMode) = ReverseOnGrad()

_hvp_strategy(backend::AbstractADType) = _strategy_from(DI.hvp_mode(backend))
_hvp_strategy(::ADTypes.AutoReactant) = ReactantHVP()

# Extensions may add annotations required by DEER's wrappers, but must not
# choose a differentiation mode.
_normalized_backend(backend::AbstractADType) = backend

function _prepare_hvp_via_grad_reverse(
    gradlogp, backend::AbstractADType, x_template::AbstractVector
)
    v_template = similar(x_template)
    fill!(v_template, zero(eltype(x_template)))
    f = _HvpReverseClosure(gradlogp)
    eff_backend = _normalized_backend(backend)
    prep = DI.prepare_gradient(f, eff_backend, x_template, DI.Constant(v_template))
    return (f, prep, eff_backend)
end

function _hvp_via_grad_reverse_prepared(prep_pair, x::AbstractVector, v::AbstractVector)
    f, prep, eff_backend = prep_pair
    return DI.gradient(f, prep, eff_backend, x, DI.Constant(v))
end

function _prepare_batch_hvp_via_grad_reverse(
    grad_batch, backend::AbstractADType, X_template::AbstractMatrix
)
    V_template = similar(X_template)
    fill!(V_template, zero(eltype(X_template)))
    f = _BatchHvpReverseClosure(grad_batch)
    eff_backend = _normalized_backend(backend)
    prep = DI.prepare_gradient(f, eff_backend, X_template, DI.Constant(V_template))
    return (f, prep, eff_backend)
end

function _batch_hvp_via_grad_reverse_prepared(
    prep_pair, X::AbstractMatrix, V::AbstractMatrix
)
    f, prep, eff_backend = prep_pair
    return DI.gradient(f, prep, eff_backend, X, DI.Constant(V))
end

#=
Strategy-dispatched factories. Each method returns a closure with one
concrete type, so the call site `_make_hvp_fn(_hvp_strategy(backend), ...)`
is type-stable: dispatch on the singleton `HVPStrategy` picks the method
at compile time, and the returned closure type is statically known.
=#
function _make_hvp_fn(
    ::ForwardOnGrad, gradlogp, backend::AbstractADType, x_template::AbstractVector
)
    prep = _prepare_hvp(gradlogp, backend, x_template)
    return (pt, dir) -> _hvp_prepared(gradlogp, prep, backend, pt, dir)
end

function _make_hvp_fn(
    ::ReverseOnGrad, gradlogp, backend::AbstractADType, x_template::AbstractVector
)
    prep = _prepare_hvp_via_grad_reverse(gradlogp, backend, x_template)
    return (pt, dir) -> _hvp_via_grad_reverse_prepared(prep, pt, dir)
end

function _make_hvp_batch_fn(
    ::ForwardOnGrad, grad_batch, backend::AbstractADType, X_template::AbstractMatrix
)
    prep = _prepare_batch_hvp_from_grad(grad_batch, backend, X_template)
    return (X, V) -> _batch_hvp_from_grad_prepared(grad_batch, prep, backend, X, V)
end

function _make_hvp_batch_fn(
    ::ReverseOnGrad, grad_batch, backend::AbstractADType, X_template::AbstractMatrix
)
    prep = _prepare_batch_hvp_via_grad_reverse(grad_batch, backend, X_template)
    return (X, V) -> _batch_hvp_via_grad_reverse_prepared(prep, X, V)
end

#= Fallbacks that ReactantExt overrides with `backend::AutoReactant` methods.
They must stay strictly wider than the extension's signatures: do not type as `AutoReactant` =#
function _make_hvp_fn(
    ::ReactantHVP, gradlogp, backend::AbstractADType, x_template::AbstractVector
)
    return error(_REACTANT_LOAD_HINT)
end

function _make_hvp_batch_fn(
    ::ReactantHVP, grad_batch, backend::AbstractADType, X_template::AbstractMatrix
)
    return error(_REACTANT_LOAD_HINT)
end

function _make_hvp_fn_second_order(
    logdensity, backend::AbstractADType, x_template::AbstractVector
)
    return error(_REACTANT_LOAD_HINT)
end

function _make_hvp_batch_fn_second_order(
    logdensity_batch_sum, backend::AbstractADType, X_template::AbstractMatrix
)
    return error(_REACTANT_LOAD_HINT)
end

# DI.hvp differentiates the log-density twice; it does not differentiate the
# prepared gradient. Only the outer backend receives DEER-specific annotations.
function _normalized_second_order(backend::DI.SecondOrder)
    return DI.SecondOrder(_normalized_backend(DI.outer(backend)), DI.inner(backend))
end

function _make_hvp_fn_second_order(
    logdensity, backend::DI.SecondOrder, x_template::AbstractVector
)
    so = _normalized_second_order(backend)
    v_template = similar(x_template)
    fill!(v_template, zero(eltype(x_template)))
    prep = DI.prepare_hvp(logdensity, so, x_template, (v_template,))
    return function (pt, dir)
        x_exec = _materialize_ad_vector(pt)
        v_exec = _tangent_like(x_exec, dir)
        return first(DI.hvp(logdensity, prep, so, x_exec, (v_exec,)))
    end
end

function _make_hvp_batch_fn_second_order(
    logdensity_batch_sum, backend::DI.SecondOrder, X_template::AbstractMatrix
)
    so = _normalized_second_order(backend)
    V_template = similar(X_template)
    fill!(V_template, zero(eltype(X_template)))
    prep = DI.prepare_hvp(logdensity_batch_sum, so, X_template, (V_template,))
    return function (X, V)
        X_exec = _materialize_ad_matrix(X)
        V_exec = _tangent_like(X_exec, V)
        return first(DI.hvp(logdensity_batch_sum, prep, so, X_exec, (V_exec,)))
    end
end

@inline function _rademacher!(z::AbstractArray{T}, rng::AbstractRNG) where {T}
    # No host buffer supplied: allocate one for this call. The hot paths pass a
    # workspace buffer instead.
    needs_host_staging(z) && return _rademacher!(z, rng, Vector{T}(undef, length(z)))
    @inbounds for i in eachindex(z)
        z[i] = rand(rng, Bool) ? one(T) : -one(T)
    end
    return z
end

@inline function _rademacher!(
    z::AbstractArray{T}, rng::AbstractRNG, host::AbstractArray{T}
) where {T}
    length(host) == length(z) || throw(DimensionMismatch("host buffer must match z"))
    _rademacher!(host, rng)
    copyto!(z, host)
    return z
end

@inline _rademacher!(z::AbstractArray, rng::AbstractRNG, ::Nothing) = _rademacher!(z, rng)

@inline _rademacher_matrix!(Z::AbstractMatrix, rng::AbstractRNG) = _rademacher!(Z, rng)
@inline _rademacher_matrix!(Z::AbstractMatrix, rng::AbstractRNG, host) =
    _rademacher!(Z, rng, host)

#= Probe `k` of a DEER update. `:stoch_diag` draws Rademacher probes. `:diag` probes
with the k-th unit vector in every column, so after all D probes the sum of
`Z .* (J Z)` is the exact diagonal. =#
function _probe!(Z::AbstractMatrix, jacobian::Symbol, k::Int, rng::AbstractRNG, ws)
    jacobian === :stoch_diag && return _probe_matrix!(Z, rng, ws)
    fill!(Z, zero(eltype(Z)))
    view(Z, k, :) .= one(eltype(Z))
    return Z
end

#= Adds probe `Z`'s diagonal estimate `Z .* (J Z)` to `A`, overwriting `A` on the
first probe, and returns the forward steps `f(Xbar)`. Every DEER update runs through
here; a recursion without a batched function is called one column at a time. =#
function _add_probe!(A, rec::TapedRecursion, ws, Xbar, Z, first::Bool)
    if rec.fwd_and_jvp_batch !== nothing
        FT, Jt = rec.fwd_and_jvp_batch(Xbar, Z)
        first ? (@. A = Z * Jt) : (@. A += Z * Jt)
        return FT
    end

    # The forward steps go to `ws.B`, which `deer_update!` then turns into offsets.
    D, T = size(Xbar)
    x, z = ws.xbar_buf, ws.zbuf
    for t in 1:T
        off = (t - 1) * D + 1
        copyto!(x, 1, Xbar, off, D)
        copyto!(z, 1, Z, off, D)
        if first
            ft, jv = _step_and_jvp(rec, x, rec.tape[t], z)
            copyto!(ws.B, off, ft, 1, D)
        else
            jv = rec.jvp(x, rec.tape[t], z)
        end
        a = view(A, :, t)
        first ? (@. a = z * jv) : (@. a += z * jv)
    end
    return ws.B
end

function _step_and_jvp(rec::TapedRecursion, x, tape_t, z)
    rec.fwd_and_jvp === nothing || return rec.fwd_and_jvp(x, tape_t, z)
    return rec.step_fwd(x, tape_t), rec.jvp(x, tape_t, z)
end

#=
Damping and clipping of the diagonal Jacobian, as in the parallel-mcmc reference
code: J <- clamp(k J, -c, c). The offset is formed from the modified diagonal,
u_t = f_t(s_{t-1}) - J s_{t-1}, so the fixed point does not move.
=#
_modifies_jacobian(k::Real, c::Real) = k != 1 || isfinite(c)

function _modify_jacobian!(J::AbstractArray, k::Real, c::Real)
    FT = eltype(J)
    kT = convert(FT, k)
    cT = convert(FT, c)
    @. J = clamp(kT * J, -cT, cT)
    return J
end

function _check_jacobian_modifiers(k::Real, c::Real)
    0 < k ≤ 1 || throw(ArgumentError("jacobian_damping must be in (0,1]"))
    c > 0 || throw(ArgumentError("jacobian_clip must be > 0"))
    return nothing
end

function _depwarn_damping(damping::Real, caller::Symbol)
    damping == 1 && return nothing
    Base.depwarn(
        "`damping` is deprecated and will be removed; use `jacobian_damping` and " *
        "`jacobian_clip` instead. Relaxing the iterate can stop DEER from converging " *
        "in at most `T` iterations.",
        caller,
    )
    return nothing
end

"""
In-place DEER update.

Writes the updated trajectory into `S_out` using `S_in` as the current iterate and
workspace buffers for all internal temporaries that can be controlled here.

`jacobian_damping` and `jacobian_clip` modify the linearization; see [`solve`](@ref).
"""
function deer_update!(
    ws::DEERWorkspace,
    S_out::AbstractMatrix,
    rec::TapedRecursion,
    s0_in::AbstractVector,
    S_in::AbstractMatrix;
    jacobian::Symbol=:stoch_diag,
    damping::Real=1.0,
    probes::Int=1,
    rng::AbstractRNG=Random.default_rng(),
    jacobian_damping::Real=1,
    jacobian_clip::Real=1,
)
    s0 = vec(s0_in)
    D, T = size(S_in)
    size(S_out) == (D, T) || throw(ArgumentError("S_out must have same size as S_in"))
    length(s0) == D || throw(ArgumentError("S must be D×T with D=length(s0)"))
    T == length(rec.tape) || throw(ArgumentError("size(S,2) must match length(rec.tape)"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))
    probes ≥ 1 || throw(ArgumentError("probes must be ≥ 1"))
    (jacobian === :stoch_diag || jacobian === :diag) ||
        throw(ArgumentError("jacobian must be :diag or :stoch_diag"))
    size(ws.A) == (D, T) || throw(ArgumentError("workspace matrix size mismatch"))
    _check_jacobian_modifiers(jacobian_damping, jacobian_clip)
    modify = _modifies_jacobian(jacobian_damping, jacobian_clip)

    A, B, Xbar, Z = ws.A, ws.B, ws.Xbar, ws.Z

    # Column t of Xbar is the state step t starts from.
    copyto!(Xbar, 1, s0, 1, D)
    T > 1 && copyto!(Xbar, D + 1, S_in, 1, D * (T - 1))

    nprobes = jacobian === :diag ? D : probes
    FT = _add_probe!(A, rec, ws, Xbar, _probe!(Z, jacobian, 1, rng, ws), true)
    for k in 2:nprobes
        _add_probe!(A, rec, ws, Xbar, _probe!(Z, jacobian, k, rng, ws), false)
    end
    if jacobian === :stoch_diag && probes > 1
        A .*= one(eltype(A)) / probes
    end
    modify && _modify_jacobian!(A, jacobian_damping, jacobian_clip)
    @. B = FT - A * Xbar

    DEERScan.solve_affine_scan_diag!(S_out, A, B, s0, ws.scan)
    if damping != 1.0
        @. S_out = (1 - damping) * S_in + damping * S_out
    end
    return S_out
end

function deer_update(
    rec::TapedRecursion,
    s0_in::AbstractVector,
    S::AbstractMatrix;
    jacobian::Symbol=:stoch_diag,
    damping::Real=1.0,
    probes::Int=1,
    rng::AbstractRNG=Random.default_rng(),
    jacobian_damping::Real=1,
    jacobian_clip::Real=1,
)
    _depwarn_damping(damping, :deer_update)
    ws = DEERWorkspace(S, vec(s0_in))
    S_out = similar(S)
    deer_update!(
        ws,
        S_out,
        rec,
        s0_in,
        S;
        jacobian=jacobian,
        damping=damping,
        probes=probes,
        rng=rng,
        jacobian_damping=jacobian_damping,
        jacobian_clip=jacobian_clip,
    )
    return S_out
end

function _check_solve_args(maxiter, tol_abs, tol_rel, damping)
    maxiter ≥ 1 || throw(ArgumentError("maxiter must be ≥ 1"))
    tol_abs ≥ 0 || throw(ArgumentError("tol_abs must be ≥ 0"))
    tol_rel ≥ 0 || throw(ArgumentError("tol_rel must be ≥ 0"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))
    return nothing
end

# `step(S_spare, S) -> (S_next, Δ_max, S_max)` may write into or free `S_spare`.
function _deer_loop(step, S, S_spare, tol_abs, tol_rel, maxiter::Int)
    converged = false
    metric = Inf
    iters = 0
    for iter in 1:maxiter
        iters = iter
        S_next, Δ_max, S_max = step(S_spare, S)
        metric = Δ_max / (tol_abs + tol_rel * S_max)
        S_spare, S = S, S_next
        if metric ≤ 1
            converged = true
            break
        end
    end
    return S, S_spare, converged, iters, metric
end

function _warn_not_converged(maxiter, metric)
    @warn "DEER did not converge in `maxiter = $maxiter` iterations " *
        "(final metric $metric, needs ≤ 1). The returned trajectory is not " *
        "an exact sequential trajectory. Raise `maxiter`, or set " *
        "`jacobian_damping` or `jacobian_clip` to stabilise the solve."
    return nothing
end

"""
Run DEER iterations until convergence.

When no workspace is supplied, one is created automatically. Supplying a
pre-allocated `DEERWorkspace` is the intended path for repeated GPU solves.

If `workspace` is supplied, `copy_result` defaults to `true` so the returned
trajectory is owned by the caller and will not be overwritten by a later solve
using the same workspace. Set `copy_result=false` for allocation-sensitive
internal loops; in that mode the returned trajectory may alias workspace-owned
buffers such as `workspace.S_tmp`.

Two keywords stabilise the Newton step when the linearized recursion amplifies
errors, which happens for multimodal targets and large step sizes (Zoltowski et
al. 2025, App. B). Each diagonal Jacobian estimate `J` is replaced with
`clamp(jacobian_damping * J, -jacobian_clip, jacobian_clip)` before the scan.
Damping by `k < 1` is the scale-ELK of Gonzalez et al. (2024). Both only change
the linearization, so the fixed point is the same and the solve still finishes
in at most `T` iterations. The defaults (`1` and `1`) match
[`ParallelMALASampler`](@ref): no scaling, and entries clipped to `[-1, 1]`.
Without clipping, one-probe `:stoch_diag` estimates on correlated targets often
exceed 1 in magnitude, and rounding error can then grow along the scan until the
solve runs far past `T` iterations. Pass `jacobian_clip=Inf` to use the
estimate as is.

`damping` is deprecated. It relaxes the iterate, `S ← (1 - damping) S_old +
damping S_new`, which loses the `T`-iteration bound; use the Jacobian options
instead.

If the solve uses all `maxiter` iterations without meeting the tolerance, it logs a
warning and returns the last iterate, which is not an exact sequential trajectory.
Pass `warn=false` to silence the warning, and `return_info=true` to check
`info.converged` yourself.
"""
function solve(
    rec::TapedRecursion,
    s0_in::AbstractVector;
    init=nothing,
    tol_abs::Real=1e-4,
    tol_rel::Real=1e-3,
    maxiter::Int=200,
    jacobian::Symbol=:stoch_diag,
    damping::Real=1.0,
    probes::Int=1,
    rng::AbstractRNG=Random.default_rng(),
    return_info::Bool=false,
    workspace::Union{Nothing,DEERWorkspace}=nothing,
    copy_result::Union{Nothing,Bool}=nothing,
    jacobian_damping::Real=1,
    jacobian_clip::Real=1,
    warn::Bool=true,
)
    s0 = vec(s0_in)
    D = length(s0)
    T = length(rec.tape)

    _check_solve_args(maxiter, tol_abs, tol_rel, damping)
    _depwarn_damping(damping, :solve)
    _check_jacobian_modifiers(jacobian_damping, jacobian_clip)

    copy_result === nothing && (copy_result = workspace !== nothing)

    ws = workspace === nothing ? DEERWorkspace(s0, T) : workspace
    size(ws.S_work) == (D, T) ||
        throw(ArgumentError("workspace state buffer has wrong size"))
    size(ws.S_tmp) == (D, T) ||
        throw(ArgumentError("workspace state buffer has wrong size"))

    S = if init === nothing
        S0 = ws.S_work
        S0 .= s0
        S0
    else
        size(init) == (D, T) || throw(ArgumentError("init must be size (D,T)"))
        copyto!(ws.S_work, init)
        ws.S_work
    end
    step = function (S_out, S_in)
        deer_update!(
            ws,
            S_out,
            rec,
            s0,
            S_in;
            jacobian=jacobian,
            damping=damping,
            probes=probes,
            rng=rng,
            jacobian_damping=jacobian_damping,
            jacobian_clip=jacobian_clip,
        )
        return (S_out, _max_abs_change(S_out, S_in)...)
    end
    S, _, converged, iters, last_metric = _deer_loop(
        step, S, ws.S_tmp, tol_abs, tol_rel, maxiter
    )

    warn && !converged && _warn_not_converged(maxiter, last_metric)

    S_result = copy_result ? copy(S) : S

    return_info || return S_result
    return S_result,
    (
        converged=converged,
        iters=iters,
        metric=last_metric,
        jacobian=jacobian,
        damping=damping,
        probes=probes,
        jacobian_damping=jacobian_damping,
        jacobian_clip=jacobian_clip,
    )
end

end # module DEER
