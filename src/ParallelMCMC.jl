module ParallelMCMC

using AbstractMCMC
using MCMCChains
using LinearAlgebra
using Random
using Statistics

include("MALA/MALA.jl")
include("DEER/DEER.jl")
include("interface.jl")

export DensityModel, MALASampler, MALATransition, MALAState
export MALA, DEER

end
