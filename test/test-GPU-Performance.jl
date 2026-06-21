using Test
using Random
using LinearAlgebra
using Statistics

using ParallelMCMC
using ADTypes: ADTypes
using CUDA: CUDA

#=
On a problem large enough to amortize CUDA kernel-launch overhead,
ParallelMALASampler on GPU with the batched DEER path is meaningfully faster
than sequential `MALASampler` on CPU.

This is a regression sanity check.

  1. GPU samples/s ≥ 2× CPU samples/s
  2. GPU samples/s ≥ 1500

If GPU is unavailable, the test set is skipped.
=#

const _PERF_GPU_AVAILABLE = try
    CUDA.functional() && (CUDA.CuArray([1.0f0]); true)
catch
    false
end

if !_PERF_GPU_AVAILABLE
    @info "GPU performance test: CUDA not functional — skipping"
else

    #=
    Multivariate Gaussian target — well-conditioned, optimal MALA acceptance from
    any start. Lets ε be set analytically so the chain actually moves and DEER
    does real work.
    =#
    function _perf_logp(β, X, _)
        Xβ = X * β
        N = oftype(zero(eltype(β)), size(X, 1))
        return -oftype(zero(eltype(β)), 0.5) * (sum(abs2, β) + sum(abs2, Xβ) / N)
    end

    function _perf_gradlogp(β, X, _)
        Xβ = X * β
        N = oftype(zero(eltype(β)), size(X, 1))
        return -β .- (X' * Xβ) ./ N
    end

    function _perf_hvp(β, v, X, _)
        Xv = X * v
        N = oftype(zero(eltype(β)), size(X, 1))
        return -v .- (X' * Xv) ./ N
    end

    function _perf_logp_batch(B, X, _)
        XB = X * B
        N = oftype(zero(eltype(B)), size(X, 1))
        return -oftype(zero(eltype(B)), 0.5) .*
               (vec(sum(abs2, B; dims=1)) .+ vec(sum(abs2, XB; dims=1)) ./ N)
    end

    function _perf_gradlogp_batch(B, X, _)
        XB = X * B
        N = oftype(zero(eltype(B)), size(X, 1))
        return -B .- (X' * XB) ./ N
    end

    function _perf_hvp_batch(B, V, X, _)
        XV = X * V
        N = oftype(zero(eltype(B)), size(X, 1))
        return -V .- (X' * XV) ./ N
    end

    function _bench(model, sampler, N; x0, reps=2)
        # Warmup
        sample(
            MersenneTwister(0),
            model,
            sampler,
            sampler isa ParallelMALASampler ? sampler.T : 100;
            progress=false,
            initial_params=x0,
        )
        GC.gc()
        if x0 isa CUDA.CuArray
            CUDA.synchronize()
        end

        ts = Float64[]
        for _ in 1:reps
            GC.gc()
            if x0 isa CUDA.CuArray
                CUDA.synchronize()
            end
            t0 = time_ns()
            sample(
                MersenneTwister(42), model, sampler, N; progress=false, initial_params=x0
            )
            if x0 isa CUDA.CuArray
                CUDA.synchronize()
            end
            push!(ts, (time_ns() - t0) / 1e9)
        end
        return median(ts)
    end

    @testset "GPU performance vs sequential CPU MALA" begin
        #=
        Sized so each per-step Sgemm is large enough to amortize CUDA kernel-
        launch overhead. D<200 is launch-bound and CPU wins; this is the regime
        users care about for the "ParallelMCMC speeds things up" claim.
        =#
        D = 300
        N_data = 4_000
        rng = MersenneTwister(20251231)
        X_cpu = randn(rng, Float32, N_data, D)
        y_cpu = randn(rng, Float32, N_data)        # unused for Gaussian target

        ε = 0.07f0
        T = 1024
        maxiter = 200
        N_samples = 2_000

        # CPU baseline
        logp_cpu = β -> _perf_logp(β, X_cpu, y_cpu)
        gradlogp_cpu = β -> _perf_gradlogp(β, X_cpu, y_cpu)
        cpu_model = DensityModel(logp_cpu, gradlogp_cpu, D)
        cpu_sampler = MALASampler(ε)
        x0_cpu = zeros(Float32, D)

        cpu_time = _bench(cpu_model, cpu_sampler, N_samples; x0=x0_cpu)
        cpu_sps = N_samples / cpu_time
        @info "CPU baseline" cpu_sps cpu_time

        # GPU run (analytical-HVP batched path)
        X_gpu = CUDA.CuMatrix(X_cpu)
        y_gpu = CUDA.CuVector(y_cpu)
        logp_gpu = β -> _perf_logp(β, X_gpu, y_gpu)
        gradlogp_gpu = β -> _perf_gradlogp(β, X_gpu, y_gpu)
        hvp_gpu = (β, v) -> _perf_hvp(β, v, X_gpu, y_gpu)
        logp_batch_gpu = B -> _perf_logp_batch(B, X_gpu, y_gpu)
        gradlogp_batch_gpu = B -> _perf_gradlogp_batch(B, X_gpu, y_gpu)
        hvp_batch_gpu = (B, V) -> _perf_hvp_batch(B, V, X_gpu, y_gpu)

        gpu_model = DensityModel(
            logp_gpu,
            gradlogp_gpu,
            D;
            hvp=hvp_gpu,
            logdensity_batch=logp_batch_gpu,
            grad_logdensity_batch=gradlogp_batch_gpu,
            hvp_batch=hvp_batch_gpu,
        )
        gpu_sampler = ParallelMALASampler(
            ε;
            T=T,
            maxiter=maxiter,
            tol_abs=1.0f-3,
            tol_rel=1.0f-2,
            damping=0.5f0,
            backend=ADTypes.AutoEnzyme(),
        )
        x0_gpu = CUDA.CuArray(x0_cpu)

        gpu_time = _bench(gpu_model, gpu_sampler, N_samples; x0=x0_gpu)
        gpu_sps = N_samples / gpu_time
        speedup = gpu_sps / cpu_sps
        @info "GPU run" gpu_sps gpu_time speedup

        #=
        The thresholds are deliberately loose so flaky shared-cluster GPUs don't
        red the build, but tight enough to catch a genuine perf regression.
        =#
        @test speedup ≥ 2.0
        @test gpu_sps ≥ 1_500.0
    end
end
