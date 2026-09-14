module CUDAExt

using ParallelMCMC: ParallelMCMC
using CUDA: CuArray

ParallelMCMC.needs_host_staging(::CuArray) = true

end
