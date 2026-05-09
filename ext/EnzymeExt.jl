module EnzymeExt

using ParallelMCMC: pmcmc_matmul, pmcmc_dot, pmcmc_dotsum
using ParallelMCMC.DEER: DEER
using ADTypes: ADTypes
using LinearAlgebra: dot
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore.EnzymeRules:
    EnzymeRules,
    FwdConfig,
    RevConfig,
    Annotation,
    AugmentedReturn,
    needs_primal,
    needs_shadow,
    overwritten,
    width

#=
Tell DEER's reverse-on-grad HVP path how to normalize a plain `AutoEnzyme()`:
fill in `function_annotation=Enzyme.Const` so Enzyme doesn't throw
`EnzymeMutabilityException` on the read-only `_HvpReverseClosure` /
`_BatchHvpReverseClosure` wrappers. If the user already specified
`function_annotation`, respect their choice.
=#
function DEER._hvp_closure_backend(backend::ADTypes.AutoEnzyme{M,A}) where {M,A}
    A === Nothing || return backend  # user already specified function_annotation
    mode = backend.mode === nothing ? Enzyme.set_runtime_activity(Enzyme.Reverse) :
        backend.mode
    return ADTypes.AutoEnzyme(; mode=mode, function_annotation=Enzyme.Const)
end

#=
Native Enzyme rules for the owned wrappers `pmcmc_matmul`, `pmcmc_dot`,
`pmcmc_dotsum`. These exist so that on GPU paths Enzyme treats each as
opaque: primal and cotangents are computed by plain `*`, `'`, `dot`, `sum`,
and broadcast — *outside* Enzyme's IR rewriter. That sidesteps the
"unsupported tag gc-transition" abort Enzyme hits when it tries to lower
`cuMemcpyDtoHAsync_v2` (emitted by every CuArray scalar reduction) or the
cuBLAS bundles inside `*(::CuArray, ::CuArray)`.

For each function we define
  - `EnzymeRules.forward`           (forward-mode JVP)
  - `EnzymeRules.augmented_primal`  (forward sweep of reverse mode)
  - `EnzymeRules.reverse`           (reverse sweep)

Width=1 only — we don't need batch-mode AD here, and DI uses width=1.
=#

# ---------------------------------------------------------------------------
# pmcmc_matmul(A, B) = A * B   —   matrix output (Duplicated)
#
# JVP:      dY  = dA * B + A * dB
# Pullback: dA += dY * B'
#           dB += A'  * dY
# ---------------------------------------------------------------------------

function EnzymeRules.forward(
    config::FwdConfig,
    ::Const{typeof(pmcmc_matmul)},
    RT::Type{<:Annotation},
    A::Annotation,
    B::Annotation,
)
    primal = needs_primal(config) ? pmcmc_matmul(A.val, B.val) : nothing
    shadow = if A isa Const && B isa Const
        nothing
    elseif A isa Const
        pmcmc_matmul(A.val, B.dval)
    elseif B isa Const
        pmcmc_matmul(A.dval, B.val)
    else
        pmcmc_matmul(A.dval, B.val) .+ pmcmc_matmul(A.val, B.dval)
    end
    if RT <: Const
        return needs_primal(config) ? primal : nothing
    elseif RT <: DuplicatedNoNeed
        return shadow
    elseif RT <: Duplicated
        return Duplicated(primal === nothing ? pmcmc_matmul(A.val, B.val) : primal, shadow)
    else
        return needs_primal(config) ? primal : shadow
    end
end

function EnzymeRules.augmented_primal(
    config::RevConfig,
    ::Const{typeof(pmcmc_matmul)},
    RT::Type{<:Annotation},
    A::Annotation,
    B::Annotation,
)
    Y = pmcmc_matmul(A.val, B.val)
    cache_A = overwritten(config)[2] ? copy(A.val) : A.val
    cache_B = overwritten(config)[3] ? copy(B.val) : B.val
    dY = if RT <: Duplicated || RT <: DuplicatedNoNeed
        zero(Y)
    else
        nothing
    end
    primal = needs_primal(config) ? Y : nothing
    return AugmentedReturn(primal, dY, (dY, cache_A, cache_B))
end

function EnzymeRules.reverse(
    config::RevConfig,
    ::Const{typeof(pmcmc_matmul)},
    ::Type{<:Annotation},
    tape,
    A::Annotation,
    B::Annotation,
)
    dY, cache_A, cache_B = tape
    if dY !== nothing
        if !(A isa Const)
            A.dval .+= pmcmc_matmul(dY, transpose(cache_B))
        end
        if !(B isa Const)
            B.dval .+= pmcmc_matmul(transpose(cache_A), dY)
        end
        fill!(dY, zero(eltype(dY)))
    end
    return (nothing, nothing)
end

# ---------------------------------------------------------------------------
# pmcmc_dot(a, b) = dot(a, b)   —   scalar output (Active)
#
# JVP:      dr  = dot(da, b) + dot(a, db)
# Pullback: da += dr * b
#           db += dr * a
# ---------------------------------------------------------------------------

function EnzymeRules.forward(
    config::FwdConfig,
    ::Const{typeof(pmcmc_dot)},
    RT::Type{<:Annotation},
    a::Annotation,
    b::Annotation,
)
    primal = needs_primal(config) ? pmcmc_dot(a.val, b.val) : nothing
    if RT <: Const
        return needs_primal(config) ? primal : nothing
    end
    da_term = a isa Const ? zero(eltype(a.val)) : pmcmc_dot(a.dval, b.val)
    db_term = b isa Const ? zero(eltype(b.val)) : pmcmc_dot(a.val, b.dval)
    tangent = da_term + db_term
    if RT <: DuplicatedNoNeed
        return tangent
    elseif RT <: Duplicated
        return Duplicated(primal === nothing ? pmcmc_dot(a.val, b.val) : primal, tangent)
    else
        return needs_primal(config) ? primal : tangent
    end
end

function EnzymeRules.augmented_primal(
    config::RevConfig,
    ::Const{typeof(pmcmc_dot)},
    ::Type{<:Annotation},
    a::Annotation,
    b::Annotation,
)
    primal = needs_primal(config) ? pmcmc_dot(a.val, b.val) : nothing
    cache_a = (!(b isa Const) && overwritten(config)[2]) ? copy(a.val) : nothing
    cache_b = (!(a isa Const) && overwritten(config)[3]) ? copy(b.val) : nothing
    return AugmentedReturn(primal, nothing, (cache_a, cache_b))
end

function EnzymeRules.reverse(
    config::RevConfig,
    ::Const{typeof(pmcmc_dot)},
    dret,
    tape,
    a::Annotation,
    b::Annotation,
)
    cache_a, cache_b = tape
    if !(dret isa Const)
        dr = dret.val
        if !(a isa Const)
            bv = cache_b !== nothing ? cache_b : b.val
            a.dval .+= dr .* bv
        end
        if !(b isa Const)
            av = cache_a !== nothing ? cache_a : a.val
            b.dval .+= dr .* av
        end
    end
    return (nothing, nothing)
end

# ---------------------------------------------------------------------------
# pmcmc_dotsum(A, B) = sum(A .* B)   —   scalar output (Active)
#
# Same algebra as `pmcmc_dot`, just with matrix args.
# ---------------------------------------------------------------------------

function EnzymeRules.forward(
    config::FwdConfig,
    ::Const{typeof(pmcmc_dotsum)},
    RT::Type{<:Annotation},
    A::Annotation,
    B::Annotation,
)
    primal = needs_primal(config) ? pmcmc_dotsum(A.val, B.val) : nothing
    if RT <: Const
        return needs_primal(config) ? primal : nothing
    end
    dA_term = A isa Const ? zero(eltype(A.val)) : pmcmc_dotsum(A.dval, B.val)
    dB_term = B isa Const ? zero(eltype(B.val)) : pmcmc_dotsum(A.val, B.dval)
    tangent = dA_term + dB_term
    if RT <: DuplicatedNoNeed
        return tangent
    elseif RT <: Duplicated
        return Duplicated(
            primal === nothing ? pmcmc_dotsum(A.val, B.val) : primal, tangent
        )
    else
        return needs_primal(config) ? primal : tangent
    end
end

function EnzymeRules.augmented_primal(
    config::RevConfig,
    ::Const{typeof(pmcmc_dotsum)},
    ::Type{<:Annotation},
    A::Annotation,
    B::Annotation,
)
    primal = needs_primal(config) ? pmcmc_dotsum(A.val, B.val) : nothing
    cache_A = (!(B isa Const) && overwritten(config)[2]) ? copy(A.val) : nothing
    cache_B = (!(A isa Const) && overwritten(config)[3]) ? copy(B.val) : nothing
    return AugmentedReturn(primal, nothing, (cache_A, cache_B))
end

function EnzymeRules.reverse(
    config::RevConfig,
    ::Const{typeof(pmcmc_dotsum)},
    dret,
    tape,
    A::Annotation,
    B::Annotation,
)
    cache_A, cache_B = tape
    if !(dret isa Const)
        dr = dret.val
        if !(A isa Const)
            Bv = cache_B !== nothing ? cache_B : B.val
            A.dval .+= dr .* Bv
        end
        if !(B isa Const)
            Av = cache_A !== nothing ? cache_A : A.val
            B.dval .+= dr .* Av
        end
    end
    return (nothing, nothing)
end

end # module
