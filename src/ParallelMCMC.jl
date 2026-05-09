module ParallelMCMC

using AbstractMCMC
using CUDA
using MCMCChains
using LinearAlgebra
using Random
using Statistics

#=
Owned wrappers: identical semantics to their Base counterparts, but provide
stable function identities for backend-specific AD rules in `ext/EnzymeExt.jl`
without committing type piracy on `Base.*` / `Base.dot` / `Base.sum`. User
model code that wants those rules to fire (notably on GPU, where the default
rules trip on cuBLAS gc-transition bundles) should call these instead.

Why both a matmul and reductions: the GPU AD-HVP path runs Enzyme reverse mode
through `gradlogp` *and* through a scalar reduction wrapping it (see DEER's
`_HvpReverseClosure`). The reduction is what actually emits the
`cuMemcpyDtoHAsync_v2` that crashes Enzyme. Owning both the matmul AND the
reduction lets Enzyme treat each as opaque, so neither bundle ever enters
its IR.
=#
pmcmc_matmul(A::AbstractVecOrMat, B::AbstractVecOrMat) = A * B
pmcmc_dot(a::AbstractVector, b::AbstractVector) = dot(a, b)
pmcmc_dotsum(A::AbstractVecOrMat, B::AbstractVecOrMat) = sum(A .* B)

include("MALA/MALA.jl")
include("DEER/DEERScan.jl")
include("DEER/DEER.jl")
include("interface.jl")

export DensityModel
export MALASampler, MALATransition, MALAState
export AdaptiveMALASampler, AdaptiveMALATransition, AdaptiveMALAState
export ParallelMALASampler, ParallelMALATransition, ParallelMALAState
export MALA, DEER
export pmcmc_matmul, pmcmc_dot, pmcmc_dotsum

end
