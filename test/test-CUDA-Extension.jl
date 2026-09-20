using Test
using Random
using ParallelMCMC

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

# Device-array stand-in: scalar `setindex!` errors, as on `CuArray`.
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

    ws = ParallelMCMC.DEER.DEERWorkspace(x, 3)
    @test ws.Zhost isa Matrix{Float32}
    @test size(ws.Zhost) == (6, 3)
    @test ws.zhost isa Vector{Float32}
    @test length(ws.zhost) == 6

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

    @test_throws ErrorException ParallelMCMC._randn_like!(MersenneTwister(0), ξ, nothing)
end

@testset "Parallel MALA tape block stages through the host" begin
    D, T, seed = 3, 4, 1234

    tape, Xi, U = ParallelMCMC._make_mala_tape_block(
        MersenneTwister(seed), StagedArray(zeros(Float32, D)), Float32, D, T
    )
    @test Xi isa StagedArray{Float32,2}
    @test size(Xi) == (D, T)
    @test U isa StagedArray{Float32,1}
    @test length(U) == T
    @test length(tape) == T

    ref_tape, ref_Xi, ref_U = ParallelMCMC._make_mala_tape_block(
        MersenneTwister(seed), zeros(Float32, D), Float32, D, T
    )
    @test ref_Xi isa Matrix{Float32}
    @test Xi.data == ref_Xi
    @test U.data == ref_U
    for t in 1:T
        @test collect(tape[t].ξ) == ref_Xi[:, t]
        @test tape[t].u == ref_tape[t].u
        @test 0 <= tape[t].u <= 1
    end
end

@testset "Staging hooks default to plain host arrays" begin
    buf = ParallelMCMC._host_staging_buffer(zeros(3), Float32, (2, 2))
    @test buf isa Matrix{Float32}
    @test size(buf) == (2, 2)

    @test ParallelMCMC._device_array_from_pointer(
        zeros(3), Float64, C_NULL, (3,), "cpu"
    ) === nothing

    x = StagedArray(zeros(Float32, 6))
    buf2 = ParallelMCMC._host_staging_buffer(x, Float64, (3, 2))
    @test buf2 isa Matrix{Float64}
    @test size(buf2) == (3, 2)
    @test ParallelMCMC._device_array_from_pointer(x, Float32, C_NULL, (6,), "cpu") ===
        nothing
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
        @test hasmethod(ParallelMCMC.needs_host_staging, Tuple{CUDA.CuArray})
        if CUDA.functional()
            @test ParallelMCMC.needs_host_staging(CUDA.zeros(Float32, 3))
        end
    end
end

@testset "CUDAExt device buffer hooks" begin
    cuda_functional = try
        using CUDA
        CUDA.functional() && (CUDA.CuArray([1.0f0]); true)
    catch
        false
    end
    if !cuda_functional
        @info "CUDAExt device buffer hooks: CUDA not functional, skipping"
    else
        template = CUDA.zeros(Float32, 3)

        host_buf = ParallelMCMC._host_staging_buffer(template, Float32, (2, 3))
        @test host_buf isa Array{Float32}
        @test size(host_buf) == (2, 3)
        @test CUDA.is_pinned(pointer(host_buf))

        # `CuPtr` does not convert to `Ptr`; go through `UInt` like the hook does.
        src = CUDA.CuArray(Float32[1, 2, 3, 4])
        raw_ptr = Ptr{Cvoid}(UInt(pointer(src)))
        dev = ParallelMCMC._device_array_from_pointer(
            template, Float32, raw_ptr, (4,), "cuda"
        )
        @test dev isa CUDA.CuArray{Float32}
        @test Array(dev) == Array(src)

        # The view aliases the source.
        dev .= 0.0f0
        @test all(iszero, Array(src))

        @test ParallelMCMC._device_array_from_pointer(
            template, Float32, raw_ptr, (4,), "rocm"
        ) === nothing
        @test ParallelMCMC._device_array_from_pointer(
            template, Float32, raw_ptr, (4,), "cpu"
        ) === nothing
    end
end
