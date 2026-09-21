#= Device-array stand-in for the host-staging paths, so they can be tested
without a GPU. Scalar `setindex!` errors, as on `CuArray`. Shared by
test-CUDA-Extension.jl and test-Reactant-HVP.jl; the name does not match
`test-*.jl`, so runtests.jl does not include it as a testset. =#

struct StagedArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Base.size(a::StagedArray) = size(a.data)
Base.getindex(a::StagedArray, i::Int...) = a.data[i...]
Base.setindex!(::StagedArray, v, i::Int...) = error("scalar indexing is unsupported")
function Base.similar(a::StagedArray, ::Type{T}, dims::Dims) where {T}
    return StagedArray(similar(a.data, T, dims))
end
Base.copyto!(a::StagedArray, src::AbstractArray) = (copyto!(a.data, src); a)

# `_prepare_model` fills the batched template with `X_template .= x_template`.
# The zero-dim method disambiguates against Base's, so scalar fills also work.
Base.BroadcastStyle(::Type{<:StagedArray}) = Broadcast.ArrayStyle{StagedArray}()
Base.copyto!(a::StagedArray, bc::Broadcast.Broadcasted) = (copyto!(a.data, bc); a)
function Base.copyto!(
    a::StagedArray, bc::Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{0}}
)
    copyto!(a.data, bc)
    return a
end

ParallelMCMC.needs_host_staging(::StagedArray) = true
