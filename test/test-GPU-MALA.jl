using Test
using Random
using LinearAlgebra
using Statistics

using ParallelMCMC
const MALA = ParallelMCMC.MALA

using CUDA: CUDA

# Check if a real GPU is accessible by attempting a small allocation.
# CUDA.functional() only checks that the library loads, not that a device exists.
const CUDA_AVAILABLE = try
    CUDA.CuArray([1.0f0])
    true
catch
    false
end

if !CUDA_AVAILABLE
    @info "No CUDA GPU detected — skipping GPU tests."
else

    # Standard normal: logp = -0.5 ||x||², grad = -x
    logp_batch(X) = vec(-0.5f0 .* sum(abs2, X; dims=1))
    gradlogp_batch(X) = -X

    # Scaled normal: dimension i has variance σᵢ² = i (so std = sqrt(i))
    # logp = -0.5 sum_i x_i²/i,  grad_i = -x_i/i
    function logp_scaled(X)
        D = size(X, 1)
        scales = CUDA.CuArray(Float32.(1:D))         # D-vector on GPU
        return vec(-0.5f0 .* sum(X .^ 2 ./ scales; dims=1))
    end

    function gradlogp_scaled(X)
        D = size(X, 1)
        scales = CUDA.CuArray(Float32.(1:D))
        return -X ./ scales
    end

    @testset "GPU inputs stay on device" begin
        D, N = 4, 32

        X = CUDA.randn(Float32, D, N)
        Ξ = CUDA.randn(Float32, D, N)
        u = CUDA.rand(Float32, N)

        X_next, accepted = MALA.mala_step_batched(
            logp_batch, gradlogp_batch, X, 0.1f0, Ξ, u
        )

        @test X_next isa CUDA.CuArray
        @test accepted isa CUDA.CuArray
        @test size(X_next) == (D, N)
        @test length(accepted) == N
    end

    @testset "GPU eltype preserved (Float32 in → Float32 out)" begin
        D, N = 5, 20
        X = CUDA.randn(Float32, D, N)
        Ξ = CUDA.randn(Float32, D, N)
        u = CUDA.rand(Float32, N)

        X_next, _ = MALA.mala_step_batched(logp_batch, gradlogp_batch, X, 0.1f0, Ξ, u)

        @test eltype(X_next) == Float32
    end

    @testset "GPU and CPU produce identical results (same seed)" begin
        D, N = 3, 16

        rng = MersenneTwister(42)
        X_cpu = randn(rng, Float32, D, N)
        Ξ_cpu = randn(rng, Float32, D, N)
        u_cpu = rand(rng, Float32, N)

        X_gpu = CUDA.CuArray(X_cpu)
        Ξ_gpu = CUDA.CuArray(Ξ_cpu)
        u_gpu = CUDA.CuArray(u_cpu)

        X_next_cpu, acc_cpu = MALA.mala_step_batched(
            logp_batch, gradlogp_batch, X_cpu, 0.1f0, Ξ_cpu, u_cpu
        )
        X_next_gpu, acc_gpu = MALA.mala_step_batched(
            logp_batch, gradlogp_batch, X_gpu, 0.1f0, Ξ_gpu, u_gpu
        )

        @test Array(X_next_gpu) ≈ X_next_cpu atol=1.0f-5
        @test Array(acc_gpu) == acc_cpu
    end

    @testset "GPU single chain (N=1) matches scalar mala_step_full" begin
        D = 4
        rng = MersenneTwister(7)
        x = randn(rng, Float32, D)
        ξ = randn(rng, Float32, D)
        u = rand(rng, Float32)

        # Scalar step on CPU
        logp_scalar = x -> -0.5f0 * dot(x, x)
        grad_scalar = x -> -x
        x_scalar, acc_scalar = MALA.mala_step_full(logp_scalar, grad_scalar, x, 0.1f0, ξ, u)

        # Batched N=1 on GPU
        X_gpu = CUDA.CuArray(reshape(copy(x), D, 1))
        Ξ_gpu = CUDA.CuArray(reshape(copy(ξ), D, 1))
        u_gpu = CUDA.CuArray([u])

        X_next_gpu, acc_gpu = MALA.mala_step_batched(
            logp_batch, gradlogp_batch, X_gpu, 0.1f0, Ξ_gpu, u_gpu
        )

        @test Array(vec(X_next_gpu)) ≈ x_scalar atol=1.0f-5
        @test Bool(Array(acc_gpu)[1]) == acc_scalar
    end

    @testset "GPU rejected chains are unchanged" begin
        D, N = 4, 64
        X = CUDA.randn(Float32, D, N)
        Ξ = CUDA.randn(Float32, D, N)
        # u very close to 1 ⟹ log(u) ≈ 0, forces rejection whenever logα ≤ 0.
        # Some chains may still be accepted (logα > 0 when proposal lands at higher density),
        # but for every rejected chain X_next must exactly equal X.
        u = CUDA.fill(1.0f0 - 1.0f-6, N)

        X_next, accepted = MALA.mala_step_batched(
            logp_batch, gradlogp_batch, X, 0.1f0, Ξ, u
        )

        acc_cpu = Array(accepted)
        X_cpu = Array(X)
        Xn_cpu = Array(X_next)
        rejected = .!acc_cpu

        # Rejected chains must be exactly unchanged.
        @test Xn_cpu[:, rejected] == X_cpu[:, rejected]
        # With u ≈ 1 most should be rejected (not a strict all-zero check).
        @test sum(acc_cpu) < N
    end

    @testset "GPU force all acceptances (u ≈ 0, tiny ε)" begin
        D, N = 4, 64
        X = CUDA.randn(Float32, D, N)
        Ξ = CUDA.randn(Float32, D, N)
        # u very close to 0 ⟹ log(u) → -∞ < any logα ⟹ accept
        u = CUDA.fill(1.0f-7, N)

        _, accepted = MALA.mala_step_batched(logp_batch, gradlogp_batch, X, 0.001f0, Ξ, u)

        @test sum(Array(accepted)) == N
    end

    @testset "GPU acceptance rate in reasonable range" begin
        # With ε=0.1 and standard normal, empirical acceptance rate should be
        # well above 0 and below 1.
        D, N, T = 5, 512, 200

        n_accepted = 0
        X = CUDA.randn(Float32, D, N)
        for _ in 1:T
            Ξ = CUDA.randn(Float32, D, N)
            u = CUDA.rand(Float32, N)
            X, acc = MALA.mala_step_batched(logp_batch, gradlogp_batch, X, 0.1f0, Ξ, u)
            n_accepted += sum(Array(acc))
        end

        rate = n_accepted / (T * N)
        @test 0.3 < rate < 0.99
    end

    @testset "GPU stationary distribution (standard normal)" begin
        D, N, T = 3, 512, 1_000

        X = CUDA.randn(Float32, D, N)
        for _ in 1:T
            Ξ = CUDA.randn(Float32, D, N)
            u = CUDA.rand(Float32, N)
            X, _ = MALA.mala_step_batched(logp_batch, gradlogp_batch, X, 0.1f0, Ξ, u)
        end

        X_cpu = Array(X)
        @test maximum(abs.(vec(mean(X_cpu; dims=2)))) < 0.15
        @test maximum(abs.(vec(var(X_cpu; dims=2)) .- 1.0f0)) < 0.25
    end

    @testset "GPU stationary distribution (scaled normal)" begin
        # Dimension i has variance i; after burn-in each row should have var ≈ i
        D, N, T = 5, 512, 2_000

        X = CUDA.randn(Float32, D, N)
        for _ in 1:T
            Ξ = CUDA.randn(Float32, D, N)
            u = CUDA.rand(Float32, N)
            X, _ = MALA.mala_step_batched(logp_scaled, gradlogp_scaled, X, 0.05f0, Ξ, u)
        end

        X_cpu = Array(X)
        vars = vec(var(X_cpu; dims=2))            # should ≈ [1, 2, 3, 4, 5]
        target = Float32.(1:D)

        @test maximum(abs.(vec(mean(X_cpu; dims=2)))) < 0.3
        @test maximum(abs.(vars .- target) ./ target) < 0.3   # within 30% relative
    end

    @testset "GPU logq_mala_batched matches CPU" begin
        D, N = 4, 16
        rng = MersenneTwister(3)
        Y_cpu = randn(rng, Float32, D, N)
        X_cpu = randn(rng, Float32, D, N)
        G_cpu = randn(rng, Float32, D, N)

        lq_cpu = MALA.logq_mala_batched(Y_cpu, X_cpu, G_cpu, 0.1f0)

        Y_gpu = CUDA.CuArray(Y_cpu)
        X_gpu = CUDA.CuArray(X_cpu)
        G_gpu = CUDA.CuArray(G_cpu)

        lq_gpu = MALA.logq_mala_batched(Y_gpu, X_gpu, G_gpu, 0.1f0)

        @test Array(lq_gpu) ≈ lq_cpu atol=1.0f-4
    end

    @testset "GPU large-scale (D=128, N=1024)" begin
        # Smoke test: just verify it runs and returns correct shapes without OOM.
        D, N = 128, 1024
        X = CUDA.randn(Float32, D, N)
        Ξ = CUDA.randn(Float32, D, N)
        u = CUDA.rand(Float32, N)

        X_next, accepted = MALA.mala_step_batched(
            logp_batch, gradlogp_batch, X, 0.05f0, Ξ, u
        )

        @test size(X_next) == (D, N)
        @test length(accepted) == N
        @test eltype(X_next) == Float32
    end

    @testset "GPU DimensionMismatch errors" begin
        X = CUDA.randn(Float32, 3, 5)
        Ξ_bad = CUDA.randn(Float32, 3, 4)
        u_ok = CUDA.rand(Float32, 5)
        u_bad = CUDA.rand(Float32, 4)

        @test_throws DimensionMismatch MALA.mala_step_batched(
            logp_batch, gradlogp_batch, X, 0.1f0, Ξ_bad, u_ok
        )
        @test_throws DimensionMismatch MALA.mala_step_batched(
            logp_batch, gradlogp_batch, X, 0.1f0, CUDA.randn(Float32, 3, 5), u_bad
        )
    end
end # if CUDA_AVAILABLE
