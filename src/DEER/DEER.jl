module DEER

using LinearAlgebra
using DifferentiationInterface
using ADTypes: ADTypes, AbstractADType
import Enzyme: Enzyme
using Random

include("DEERScan.jl")
using .DEERScan

const DI = DifferentiationInterface
const DEFAULT_BACKEND = ADTypes.AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Duplicated)
const DEFAULT_HVP_BACKEND = DI.SecondOrder(
    DEFAULT_BACKEND,
    ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)),
)

export TapedRecursion,
       DEERWorkspace,
       deer_update!, deer_update,
       solve,
       _prepare_hvp, _hvp_prepared, _hvp_nopre,
       DEFAULT_BACKEND, DEFAULT_HVP_BACKEND

"""
Deterministic recursion driven by a pre-generated tape.

- `step_fwd(x, tape_t)`: exact forward transition (may include MH accept/reject).
- `jvp(x, tape_t, v)`: explicit JVP `(x, tape_t, v) -> J·v` of the differentiable
  surrogate step. Required; must not call AD through `step_fwd`'s control flow.
- `fwd_and_jvp(x, tape_t, v)`: optional fused function `-> (x_next, J·v)`.
- `fwd_and_jvp_batch(Xbar, Z)`: optional batched function `-> (FT, Jt)`.
"""
struct TapedRecursion{Ff,Fj,Ffj,Ffb,Tt}
    step_fwd::Ff
    jvp::Fj
    fwd_and_jvp::Ffj
    fwd_and_jvp_batch::Ffb
    tape::Vector{Tt}
end

function TapedRecursion(step_fwd, jvp, tape::Vector; fwd_and_jvp=nothing, fwd_and_jvp_batch=nothing)
    return TapedRecursion(step_fwd, jvp, fwd_and_jvp, fwd_and_jvp_batch, tape)
end

"""
Reusable buffers for allocation-light DEER updates and solves.

All buffers are created with the same array type / device placement as `S_template`
(or `s0_template` for vector buffers), so the workspace is GPU-compatible when
constructed from `CuArray` templates.
"""
struct DEERWorkspace{M,V,SW}
    A::M
    B::M
    Xbar::M
    Z::M
    S_tmp::M
    diff_buf::M
    zbuf::V
    jt_buf::V
    xbar_buf::V
    scan::SW
end

function DEERWorkspace(S_template::AbstractMatrix, s0_template::AbstractVector)
    A = similar(S_template)
    B = similar(S_template)
    Xbar = similar(S_template)
    Z = similar(S_template)
    S_tmp = similar(S_template)
    diff_buf = similar(S_template)
    zbuf = similar(s0_template)
    jt_buf = similar(s0_template)
    xbar_buf = similar(s0_template)
    scan = DEERScan.AffineScanWorkspace(S_template)
    return DEERWorkspace(A, B, Xbar, Z, S_tmp, diff_buf, zbuf, jt_buf, xbar_buf, scan)
end

function DEERWorkspace(s0_template::AbstractVector, T::Integer)
    D = length(s0_template)
    S_template = similar(s0_template, D, T)
    return DEERWorkspace(S_template, s0_template)
end

function _prepare_hvp(f, backend::AbstractADType, x_template::AbstractVector)
    v_template = similar(x_template)
    fill!(v_template, zero(eltype(x_template)))
    return DI.prepare_pushforward(f, backend, x_template, (v_template,); strict=Val(false))
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

function _hvp_prepared(f, prep, backend::AbstractADType, x::AbstractVector, v::AbstractVector)
    x_exec = _materialize_ad_vector(x)
    v_exec = _tangent_like(x_exec, v)
    res = DI.pushforward(f, prep, backend, x_exec, (v_exec,))
    return res isa Tuple ? first(res) : res
end

function _hvp_nopre(f, backend::AbstractADType, x::AbstractVector, v::AbstractVector)
    x_exec = _materialize_ad_vector(x)
    v_exec = _tangent_like(x_exec, v)
    prep = DI.prepare_pushforward(f, backend, x_exec, (v_exec,); strict=Val(false))
    res = DI.pushforward(f, prep, backend, x_exec, (v_exec,))
    return res isa Tuple ? first(res) : res
end

_hvp_backend(backend::AbstractADType) = backend

_hvp_second_order_backend(backend::AbstractADType) = DI.SecondOrder(backend, backend)
_hvp_second_order_backend(backend::DI.SecondOrder) = backend

_hvp_backend(::ADTypes.AutoEnzyme) = DEFAULT_BACKEND

_hvp_second_order_backend(::ADTypes.AutoEnzyme) = DEFAULT_HVP_BACKEND

function _prepare_logdensity_hvp(f, backend::AbstractADType, x_template::AbstractVector)
    v_template = similar(x_template)
    fill!(v_template, zero(eltype(x_template)))
    return DI.prepare_hvp(f, backend, x_template, (v_template,); strict=Val(false))
end

function _logdensity_hvp_prepared(
    f,
    prep,
    backend::AbstractADType,
    x::AbstractVector,
    v::AbstractVector,
)
    x_exec = _materialize_ad_vector(x)
    v_exec = _tangent_like(x_exec, v)
    res = DI.hvp(f, prep, backend, x_exec, (v_exec,))
    return res isa Tuple ? first(res) : res
end

function _logdensity_hvp_nopre(
    f,
    backend::AbstractADType,
    x::AbstractVector,
    v::AbstractVector,
)
    x_exec = _materialize_ad_vector(x)
    v_exec = _tangent_like(x_exec, v)
    prep = DI.prepare_hvp(f, backend, x_exec, (v_exec,); strict=Val(false))
    res = DI.hvp(f, prep, backend, x_exec, (v_exec,))
    return res isa Tuple ? first(res) : res
end

function _prepare_batch_hvp_from_grad(
    grad_batch,
    backend::AbstractADType,
    X_template::AbstractMatrix,
)
    V_template = similar(X_template)
    fill!(V_template, zero(eltype(X_template)))
    return DI.prepare_pushforward(
        grad_batch, backend, X_template, (V_template,); strict=Val(false)
    )
end

function _batch_hvp_from_grad_prepared(
    grad_batch,
    prep,
    backend::AbstractADType,
    X::AbstractMatrix,
    V::AbstractMatrix,
)
    X_exec = _materialize_ad_matrix(X)
    V_exec = _tangent_like(X_exec, V)
    res = DI.pushforward(grad_batch, prep, backend, X_exec, (V_exec,))
    return res isa Tuple ? first(res) : res
end

@inline function _rademacher!(z::AbstractVector{T}, rng::AbstractRNG) where {T}
    D = length(z)
    bits = rand(rng, Bool, D)
    vals = Vector{T}(undef, D)
    @inbounds for i in 1:D
        vals[i] = bits[i] ? one(T) : -one(T)
    end
    copyto!(z, vals)
    return z
end

@inline function _rademacher_matrix!(Z::AbstractMatrix{T}, rng::AbstractRNG) where {T}
    D, n = size(Z)
    bits = rand(rng, Bool, D, n)
    vals = Matrix{T}(undef, D, n)
    @. vals = ifelse(bits, one(T), -one(T))
    copyto!(Z, vals)
    return Z
end

function jac_diag_via_jvps(rec::TapedRecursion, x::AbstractVector, t::Int)
    D = length(x)
    FT = eltype(x)
    ei_host = Vector{FT}(undef, D)
    ei = similar(x)
    d = similar(x)
    fill!(d, zero(FT))
    for i in 1:D
        fill!(ei_host, zero(FT))
        ei_host[i] = one(FT)
        copyto!(ei, ei_host)
        jv = rec.jvp(x, rec.tape[t], ei)
        d .+= ei .* jv
    end
    return d
end

function jac_diag_stoch(
    rec::TapedRecursion,
    x::AbstractVector,
    t::Int;
    probes::Int=1,
    rng::AbstractRNG=Random.default_rng(),
    zbuf::Union{Nothing,AbstractVector}=nothing,
)
    D = length(x)
    FT = float(eltype(x))
    probes ≥ 1 || throw(ArgumentError("probes must be ≥ 1"))

    z = zbuf === nothing ? Vector{FT}(undef, D) : zbuf
    length(z) == D || throw(ArgumentError("zbuf must have length D"))

    d = fill!(similar(z, D), zero(FT))
    for _ in 1:probes
        _rademacher!(z, rng)
        jv = rec.jvp(x, rec.tape[t], z)
        @. d += z * jv
    end
    d .*= one(FT) / probes
    return d
end

"""
In-place DEER update.

Writes the updated trajectory into `S_out` using `S_in` as the current iterate and
workspace buffers for all internal temporaries that can be controlled here.
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
)
    s0 = vec(s0_in)
    D, T = size(S_in)
    size(S_out) == (D, T) || throw(ArgumentError("S_out must have same size as S_in"))
    length(s0) == D || throw(ArgumentError("S must be D×T with D=length(s0)"))
    T == length(rec.tape) || throw(ArgumentError("size(S,2) must match length(rec.tape)"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))
    probes ≥ 1 || throw(ArgumentError("probes must be ≥ 1"))
    size(ws.A) == (D, T) || throw(ArgumentError("workspace matrix size mismatch"))

    A = ws.A
    B = ws.B

    if rec.fwd_and_jvp_batch !== nothing && jacobian === :stoch_diag
        Xbar = ws.Xbar
        Z = ws.Z

        copyto!(view(Xbar, :, 1), s0)
        if T > 1
            @views Xbar[:, 2:end] .= S_in[:, 1:(T - 1)]
        end

        _rademacher_matrix!(Z, rng)

        FT, Jt = rec.fwd_and_jvp_batch(Xbar, Z)
        @. A = Z * Jt
        for _ in 2:probes
            _rademacher_matrix!(Z, rng)
            _, Jt = rec.fwd_and_jvp_batch(Xbar, Z)
            @. A += Z * Jt
        end
        if probes > 1
            A .*= one(eltype(A)) / probes
        end
        @. B = FT - A * Xbar
    else
        zbuf = ws.zbuf
        jt_buf = ws.jt_buf
        xbar_buf = ws.xbar_buf

        for t in 1:T
            xbar = if t == 1
                s0
            else
                copyto!(xbar_buf, view(S_in, :, t - 1))
                xbar_buf
            end

            if jacobian === :stoch_diag
                _rademacher!(zbuf, rng)
                ft, jvp1 = if rec.fwd_and_jvp !== nothing
                    rec.fwd_and_jvp(xbar, rec.tape[t], zbuf)
                else
                    (rec.step_fwd(xbar, rec.tape[t]), rec.jvp(xbar, rec.tape[t], zbuf))
                end

                @. jt_buf = zbuf * jvp1
                for _ in 2:probes
                    _rademacher!(zbuf, rng)
                    jvp_k = rec.jvp(xbar, rec.tape[t], zbuf)
                    @. jt_buf += zbuf * jvp_k
                end
                if probes > 1
                    jt_buf .*= one(eltype(jt_buf)) / probes
                end

                view(A, :, t) .= jt_buf
                @views @. B[:, t] = ft - jt_buf * xbar

            elseif jacobian === :diag
                ft = rec.step_fwd(xbar, rec.tape[t])
                jt = jac_diag_via_jvps(rec, xbar, t)
                view(A, :, t) .= jt
                @views @. B[:, t] = ft - jt * xbar

            else
                throw(ArgumentError("jacobian must be :diag or :stoch_diag"))
            end
        end
    end

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
)
    ws = DEERWorkspace(S, vec(s0_in))
    S_out = similar(S)
    deer_update!(ws, S_out, rec, s0_in, S; jacobian=jacobian, damping=damping, probes=probes, rng=rng)
    return S_out
end

"""
Run DEER iterations until convergence.

When no workspace is supplied, one is created automatically. Supplying a
pre-allocated `DEERWorkspace` is the intended path for repeated GPU solves.
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
)
    s0 = vec(s0_in)
    D = length(s0)
    T = length(rec.tape)

    maxiter ≥ 1 || throw(ArgumentError("maxiter must be ≥ 1"))
    tol_abs ≥ 0 || throw(ArgumentError("tol_abs must be ≥ 0"))
    tol_rel ≥ 0 || throw(ArgumentError("tol_rel must be ≥ 0"))
    damping > 0 && damping ≤ 1 || throw(ArgumentError("damping must be in (0,1]"))

    ws = workspace === nothing ? DEERWorkspace(s0, T) : workspace
    size(ws.S_tmp) == (D, T) || throw(ArgumentError("workspace state buffer has wrong size"))

    S = if init === nothing
        S0 = similar(s0, D, T)
        S0 .= reshape(s0, D, 1)
        S0
    else
        size(init) == (D, T) || throw(ArgumentError("init must be size (D,T)"))
        copy(init)
    end
    S_new = ws.S_tmp

    converged = false
    last_metric = Inf
    iters = 0

    for iter in 1:maxiter
        iters = iter
        deer_update!(ws, S_new, rec, s0, S; jacobian=jacobian, damping=damping, probes=probes, rng=rng)

        @. ws.diff_buf = abs(S_new - S)
        Δ_max = maximum(ws.diff_buf)
        @. ws.diff_buf = abs(S_new)
        S_scale = tol_abs + tol_rel * maximum(ws.diff_buf)
        metric = Δ_max / S_scale

        last_metric = metric
        if metric ≤ 1
            converged = true
            S = S_new
            break
        end

        S, S_new = S_new, S
    end

    return_info || return S
    return S,
    (
        converged=converged,
        iters=iters,
        metric=last_metric,
        jacobian=jacobian,
        damping=damping,
        probes=probes,
    )
end

end # module DEER
