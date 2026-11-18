using Test
using Random
using ParallelMCMC
using JLArrays: JLArrays, JLArray
using CUDA: CUDA

const DEER = ParallelMCMC.DEER

#= The scan and convergence-check kernels are KernelAbstractions kernels, so they
run on any GPU array type. JLArrays is a GPU array type that runs on the CPU, so
these tests cover the kernels without a GPU; CUDA repeats them when one works. =#
const _KERNEL_DEVICES = let devs = Any[("JLArray", JLArray)]
    cuda_ok = try
        CUDA.functional() && (CUDA.CuArray([1.0f0]); true)
    catch
        false
    end
    cuda_ok ? push!(devs, ("CuArray", CUDA.CuArray)) :
    @info "GPU kernels: CUDA unavailable, testing on JLArrays only"
    devs
end

@testset "Kernel backend selection" begin
    @test ParallelMCMC._kernel_backend(zeros(3)) === nothing
    @test ParallelMCMC._kernel_backend(zeros(3, 2), zeros(3)) === nothing
    x = JLArray(zeros(Float32, 4, 3))
    @test ParallelMCMC._kernel_backend(x) isa JLArrays.JLBackend
    @test ParallelMCMC._kernel_backend(view(x, :, 1:2)) isa JLArrays.JLBackend
    @test ParallelMCMC._kernel_backend(x, JLArray(zeros(Float32, 4))) isa JLArrays.JLBackend
    # Mixed host and device arrays take the broadcast path, which reports the mismatch.
    @test ParallelMCMC._kernel_backend(x, zeros(Float32, 4)) === nothing
    @test ParallelMCMC._kernel_backend(zeros(Float32, 4), x) === nothing

    @test ParallelMCMC.needs_host_staging(x)
    @test ParallelMCMC.needs_host_staging(view(x, :, 1))
end

@testset "Kernels on $name" for (name, dev) in _KERNEL_DEVICES
    rng = MersenneTwister(11)

    # Shapes cover one chunk, a partial last chunk, D = 1 and T = 1.
    @testset "chunked scan matches the sequential recursion (D=$D, T=$T, $FT)" for (
        D, T, FT
    ) in (
        (3, 1, Float32),
        (7, 33, Float32),
        (1, 1000, Float32),
        (25, 4097, Float32),
        (5, 300, Float64),
    )
        A = FT(0.9) .* rand(rng, FT, D, T) .+ FT(0.05)
        B = randn(rng, FT, D, T)
        s0 = randn(rng, FT, D)
        Ad, Bd, s0d = dev(A), dev(B), dev(s0)
        S = similar(Ad)
        out = DEER.DEERScan.solve_affine_scan_diag!(
            S, Ad, Bd, s0d, DEER.DEERScan.AffineScanWorkspace(Ad)
        )
        @test out === S
        @test Array(S) ≈ DEER.DEERScan.solve_affine_seq(A, B, s0)
    end

    @testset "scan handles D = 0 and T = 0" begin
        for (D, T) in ((0, 5), (4, 0))
            A = dev(zeros(Float32, D, T))
            S = similar(A)
            @test DEER.DEERScan.solve_affine_scan_diag!(
                S, A, A, dev(zeros(Float32, D)), DEER.DEERScan.AffineScanWorkspace(A)
            ) === S
        end
    end

    @testset "scan checks sizes even with check_shapes=false" begin
        A = dev(rand(rng, Float32, 4, 50))
        ws = DEER.DEERScan.AffineScanWorkspace(A)
        @test_throws DimensionMismatch DEER.DEERScan.solve_affine_scan_diag!(
            similar(A),
            A,
            dev(rand(rng, Float32, 4, 49)),
            dev(rand(rng, Float32, 4)),
            ws;
            check_shapes=false,
        )
        @test_throws DimensionMismatch DEER.DEERScan.solve_affine_scan_diag!(
            similar(A),
            A,
            dev(rand(rng, Float32, 4, 50)),
            dev(rand(rng, Float32, 3)),
            ws;
            check_shapes=false,
        )
    end

    @testset "convergence check matches the host ($FT, $(size))" for FT in
                                                                     (Float32, Float64),
        size in ((6, 500), (3, 4), (1000, 300))

        S_new = randn(rng, FT, size...)
        S = randn(rng, FT, size...)
        @test DEER._max_abs_change(dev(S_new), dev(S)) == DEER._max_abs_change(S_new, S)
    end

    @testset "convergence check propagates NaN and checks sizes" begin
        S_new = randn(rng, Float32, 6, 500)
        S = randn(rng, Float32, 6, 500)
        S_new[4, 321] = NaN32
        Δ, m = DEER._max_abs_change(dev(S_new), dev(S))
        @test isnan(Δ) && isnan(m)
        @test DEER._max_abs_change(dev(zeros(Float32, 0, 3)), dev(zeros(Float32, 0, 3))) ==
            (0.0f0, 0.0f0)
        @test_throws DimensionMismatch DEER._max_abs_change(dev(S_new), dev(S[:, 1:10]))
    end
end

@testset "ParallelMALASampler runs on a GPU array type without an extension" begin
    # JLArrays stands in for a GPU with no ParallelMCMC extension (AMDGPU, Metal, ...).
    D = 4
    lp(x) = -0.5f0 * sum(abs2, x)
    lp_batch(X) = vec(-0.5f0 .* sum(abs2, X; dims=1))
    model = DensityModel(
        lp,
        x -> -x,
        D;
        hvp=(x, v) -> -v,
        logdensity_batch=lp_batch,
        grad_logdensity_batch=X -> -X,
        hvp_batch=(X, V) -> -V,
    )
    sampler = ParallelMALASampler(0.3f0; T=32)
    run(x0) = sample(
        MersenneTwister(1),
        model,
        sampler,
        200;
        initial_params=x0,
        progress=false,
        chain_type=Any,
    )
    # Non-interactive sessions disallow scalar indexing of GPU arrays, so this also
    # checks that the sampler never indexes the device arrays element by element.
    samples = run(JLArray(zeros(Float32, D)))
    @test samples[end].x isa JLArray{Float32,1}
    X = reduce(hcat, [Array(s.x) for s in samples])
    X_cpu = reduce(hcat, [s.x for s in run(zeros(Float32, D))])
    # Same tape, so the chains agree up to the DEER tolerance.
    @test X ≈ X_cpu atol = 1.0f-4
end
