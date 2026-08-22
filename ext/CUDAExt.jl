module CUDAExt

#=
CUDA's only job in ParallelMCMC is telling the random fills that a `CuArray`
cannot be written element by element. Everything else the samplers do to a
device array — `similar`, broadcast, `copyto!`, the affine scan — goes through
the generic `AbstractArray` interface, so no other method is needed here.
=#

using ParallelMCMC: ParallelMCMC
using CUDA: CuArray

ParallelMCMC.needs_host_staging(::CuArray) = true

end
