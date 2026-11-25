module EnzymeExt

using ParallelMCMC: pmcmc_matmul
using ParallelMCMC.DEER: DEER
using ADTypes: ADTypes
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore.EnzymeRules:
    EnzymeRules,
    FwdConfig,
    RevConfig,
    Annotation,
    AugmentedReturn,
    needs_primal,
    overwritten

# Add the default function annotation without changing the user's mode.
function DEER._normalized_backend(backend::ADTypes.AutoEnzyme{M,A}) where {M,A}
    A === Nothing || return backend
    return ADTypes.AutoEnzyme(; mode=backend.mode, function_annotation=Enzyme.Const)
end

#=
Native Enzyme rules for the owned wrapper `pmcmc_matmul`. They make Enzyme
treat it as opaque: primal and cotangents are computed by plain `*` and
broadcast outside Enzyme's IR rewriter.

Why: forward-mode Enzyme cannot differentiate cuBLAS calls on CuArrays
(CUDA.jl 6.5.1, Enzyme 0.13.221). `A * x` fails in the graph-capture branch of
CUDACore's `CuRefValue{T}(x)`, before cuBLAS is even called:

    IllegalTypeAnalysisException: Enzyme compilation failed due to illegal type analysis.
    Failure within method: closure in CUDACore.initialize_during_capture (graph.jl:147)

Reverse mode handles `*` itself. Drop these rules once forward mode works too.
=#

#=
pmcmc_matmul(A, B) = A * B   —   matrix output (Duplicated)

JVP:      dY  = dA * B + A * dB
Pullback: dA += dY * B'
          dB += A'  * dY
=#

function EnzymeRules.forward(
    config::FwdConfig,
    ::Const{typeof(pmcmc_matmul)},
    RT::Type{<:Annotation},
    A::Annotation,
    B::Annotation,
)
    primal = needs_primal(config) ? pmcmc_matmul(A.val, B.val) : nothing
    shadow = if A isa Const && B isa Const
        #=
        Both args Const → output tangent is structurally zero. We still need
        to return an output-shaped array (Enzyme's shadow-type check rejects
        `nothing` when the caller asked for `Duplicated`/`DuplicatedNoNeed`).
        =#
        zero(primal === nothing ? pmcmc_matmul(A.val, B.val) : primal)
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

end # module
