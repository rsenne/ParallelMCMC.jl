using Test
using Random
using ParallelMCMC

#=
`needs_host_staging` is the only thing left tying the samplers to a particular
GPU package (#59). These tests pin both halves of that contract: the CPU
default, and what a device array type gets by opting in.
=#

@testset "Host staging defaults to off" begin
    @test !ParallelMCMC.needs_host_staging(zeros(3))
    @test !ParallelMCMC.needs_host_staging(zeros(Float32, 3, 4))
    @test !ParallelMCMC.needs_host_staging(view(zeros(6), 1:3))
end

@testset "Workspaces skip host buffers for CPU arrays" begin
    ws = ParallelMCMC.DEER.DEERWorkspace(randn(Float32, 5), 4)
    @test ws.Zhost === nothing
    @test ws.zhost === nothing
end

#=
Stand-in for a device array: no CuArray needed, so this runs on CI without a
GPU. It refuses `setindex!`, which is what the staged fills exist to avoid.
=#
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

ParallelMCMC.needs_host_staging(::StagedArray) = true

@testset "Opting a device array type in" begin
    x = StagedArray(zeros(Float32, 6))
    @test ParallelMCMC.needs_host_staging(x)

    #= The workspace allocates host mirrors off the same trait, so a template
    that stages gets them and a CPU template does not. =#
    ws = ParallelMCMC.DEER.DEERWorkspace(x, 3)
    @test ws.Zhost isa Matrix{Float32}
    @test size(ws.Zhost) == (6, 3)
    @test ws.zhost isa Vector{Float32}
    @test length(ws.zhost) == 6

    #= Both the buffered and unbuffered fills have to route around setindex!;
    the unbuffered one allocates its own staging buffer. =#
    rng = MersenneTwister(0)
    ParallelMCMC.DEER._rademacher!(x, rng, ws.zhost)
    @test all(v -> abs(v) == 1.0f0, x.data)

    fill!(x.data, 0.0f0)
    ParallelMCMC.DEER._rademacher!(x, MersenneTwister(0))
    @test all(v -> abs(v) == 1.0f0, x.data)

    @test_throws DimensionMismatch ParallelMCMC.DEER._rademacher!(
        x, rng, Vector{Float32}(undef, 2)
    )
end

@testset "MALA noise stages through the host buffer" begin
    ξ, host = ParallelMCMC._make_noise_buffer(StagedArray(zeros(Float32, 4)), Float32, 4)
    @test host isa Vector{Float32}
    ParallelMCMC._randn_like!(MersenneTwister(0), ξ, host)
    @test ξ.data == host

    # Without one there is nowhere to generate the noise, so say so rather than
    # failing later inside setindex!.
    @test_throws ErrorException ParallelMCMC._randn_like!(MersenneTwister(0), ξ, nothing)
end

@testset "CUDAExt loads with CUDA" begin
    cuda_loadable = try
        using CUDA
        true
    catch
        false
    end
    if !cuda_loadable
        @warn "CUDA not loadable — skipping CUDAExt test"
    else
        @test Base.get_extension(ParallelMCMC, :CUDAExt) !== nothing
        # Setting the trait for `CuArray` is the whole of the extension, and it
        # is right about the type whether or not a device is present.
        @test hasmethod(ParallelMCMC.needs_host_staging, Tuple{CUDA.CuArray})
        if CUDA.functional()
            @test ParallelMCMC.needs_host_staging(CUDA.zeros(Float32, 3))
        end
    end
end
