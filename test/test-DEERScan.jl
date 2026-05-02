using Test
using Random
using LinearAlgebra

using ParallelMCMC
const DEERScan = ParallelMCMC.DEERScan

# GPU is the primary target here.
using CUDA: CUDA

const CUDA_AVAILABLE = try
    CUDA.CuArray([1.0f0])
    true
catch
    false
end

"""
Make a stable diagonal-affine recurrence

    s_t = A[:,t] .* s_{t-1} + B[:,t]

with coefficients small enough that trajectories stay well behaved.
"""
function make_affine_problem(rng::AbstractRNG, D::Int, T::Int; FT=Float32, stable=true)
    A = rand(rng, FT, D, T)
    B = randn(rng, FT, D, T)
    s0 = randn(rng, FT, D)

    if stable
        # keep multipliers safely away from explosive regimes
        A .= FT(0.2) .* (A .- FT(0.5))
    end
    return A, B, s0
end

@testset "DEERScan CPU scan matches sequential reference" begin
    rng = MersenneTwister(11)

    for (D, T) in ((1, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 17), (8, 31), (8, 64))
        A, B, s0 = make_affine_problem(rng, D, T; FT=Float64, stable=true)
        S_ref = DEERScan.solve_affine_seq(A, B, s0)

        ws = DEERScan.AffineScanWorkspace(A)
        S = similar(A)
        out = DEERScan.solve_affine_scan_diag!(S, A, B, s0, ws)

        @test out === S
        @test S ≈ S_ref atol=1e-12 rtol=1e-12

        fill!(S, NaN)
        DEERScan.solve_affine_scan_diag!(S, A, B, s0, ws)
        @test S ≈ S_ref atol=1e-12 rtol=1e-12
    end
end

@testset "DEERScan affine scan validation helpers" begin
    rng = MersenneTwister(12)
    D, T = 4, 9
    A, B, s0 = make_affine_problem(rng, D, T; FT=Float64, stable=true)

    S_seq = DEERScan.solve_affine_seq(A, B, s0)
    S_scan = DEERScan.solve_affine_scan_diag(A, B, s0)

    @test DEERScan.affine_scan_residual(S_seq, A, B, s0) ≤ 1e-12
    @test DEERScan.affine_scan_residual(S_scan, A, B, s0) ≤ 1e-12

    S_bad_first = copy(S_seq)
    S_bad_first[2, 1] += 0.25
    @test DEERScan.affine_scan_residual(S_bad_first, A, B, s0) ≈ 0.25

    S_bad_later = copy(S_seq)
    S_bad_later[3, 6] -= 0.5
    @test DEERScan.affine_scan_residual(S_bad_later, A, B, s0) ≈ 0.5

    chk = DEERScan.check_affine_scan(A, B, s0; atol=1e-12, rtol=1e-12)
    @test chk.ok
    @test chk.max_abs_err ≤ 1e-12
    @test chk.max_rel_err ≤ 1
    @test chk.residual_seq ≤ 1e-12
    @test chk.residual_scan ≤ 1e-12
    @test chk.S_seq ≈ S_seq atol=0 rtol=0
    @test chk.S_scan ≈ S_scan atol=0 rtol=0

    A0 = zeros(Float64, D, 0)
    B0 = similar(A0)
    s00 = randn(rng, D)
    S0 = similar(A0)
    chk0 = DEERScan.check_affine_scan(A0, B0, s00)
    @test DEERScan.affine_scan_residual(S0, A0, B0, s00) == 0.0
    @test chk0.ok
    @test chk0.max_abs_err == 0.0
    @test chk0.max_rel_err == 0.0
    @test size(chk0.S_seq) == (D, 0)
    @test size(chk0.S_scan) == (D, 0)
end

@testset "DEERScan affine scan helper shape validation" begin
    rng = MersenneTwister(13)
    A, B, s0 = make_affine_problem(rng, 3, 5; FT=Float64, stable=true)
    S = DEERScan.solve_affine_seq(A, B, s0)

    @test_throws DimensionMismatch DEERScan.affine_scan_residual(S, B[:, 1:4], B, s0)
    @test_throws DimensionMismatch DEERScan.affine_scan_residual(S[:, 1:4], A, B, s0)
    @test_throws DimensionMismatch DEERScan.affine_scan_residual(S, A, B, randn(rng, 4))
    @test_throws DimensionMismatch DEERScan.check_affine_scan(A, B[:, 1:4], s0)
    @test_throws DimensionMismatch DEERScan.check_affine_scan(A, B, randn(rng, 4))
end

@testset "DEERScan primitive (GPU-first)" begin
    if !CUDA_AVAILABLE
        @info "No CUDA GPU detected — skipping DEERScan GPU tests."
    else
        @testset "GPU scan matches CPU sequential reference" begin
            rng = MersenneTwister(1)

            for (D, T) in ((1, 1), (1, 17), (4, 16), (8, 31), (16, 64))
                A_cpu, B_cpu, s0_cpu = make_affine_problem(
                    rng, D, T; FT=Float32, stable=true
                )

                # CPU oracle: strictly sequential reference
                S_ref = DEERScan.solve_affine_seq(A_cpu, B_cpu, s0_cpu)

                # GPU path under test
                A_gpu = CUDA.CuArray(A_cpu)
                B_gpu = CUDA.CuArray(B_cpu)
                s0_gpu = CUDA.CuArray(s0_cpu)

                S_gpu = DEERScan.solve_affine_scan_diag(A_gpu, B_gpu, s0_gpu)

                @test S_gpu isa CUDA.CuMatrix
                @test size(S_gpu) == (D, T)
                @test Array(S_gpu) ≈ S_ref atol=1.0f-5 rtol=1.0f-5
            end
        end

        @testset "GPU in-place scan matches CPU sequential reference" begin
            rng = MersenneTwister(2)

            D, T = 8, 48
            A_cpu, B_cpu, s0_cpu = make_affine_problem(rng, D, T; FT=Float32, stable=true)
            S_ref = DEERScan.solve_affine_seq(A_cpu, B_cpu, s0_cpu)

            A_gpu = CUDA.CuArray(A_cpu)
            B_gpu = CUDA.CuArray(B_cpu)
            s0_gpu = CUDA.CuArray(s0_cpu)
            S_gpu = similar(A_gpu)

            out = DEERScan.solve_affine_scan_diag!(S_gpu, A_gpu, B_gpu, s0_gpu)

            @test out === S_gpu
            @test S_gpu isa CUDA.CuMatrix
            @test Array(S_gpu) ≈ S_ref atol=1.0f-5 rtol=1.0f-5
        end

        @testset "GPU scan satisfies the recurrence residual" begin
            rng = MersenneTwister(3)

            D, T = 6, 40
            A_cpu, B_cpu, s0_cpu = make_affine_problem(rng, D, T; FT=Float32, stable=true)

            A_gpu = CUDA.CuArray(A_cpu)
            B_gpu = CUDA.CuArray(B_cpu)
            s0_gpu = CUDA.CuArray(s0_cpu)

            S_gpu = DEERScan.solve_affine_scan_diag(A_gpu, B_gpu, s0_gpu)

            # Residual can be checked on CPU after materializing.
            resid = DEERScan.affine_scan_residual(Array(S_gpu), A_cpu, B_cpu, s0_cpu)
            @test resid ≤ 1.0f-5
        end

        @testset "GPU scan agrees with CPU on edge cases" begin
            rng = MersenneTwister(4)

            # zero additive term
            let
                D, T = 5, 20
                A_cpu = rand(rng, Float32, D, T) .* 0.2f0
                B_cpu = zeros(Float32, D, T)
                s0_cpu = randn(rng, Float32, D)

                S_ref = DEERScan.solve_affine_seq(A_cpu, B_cpu, s0_cpu)
                S_gpu = DEERScan.solve_affine_scan_diag(
                    CUDA.CuArray(A_cpu), CUDA.CuArray(B_cpu), CUDA.CuArray(s0_cpu)
                )

                @test Array(S_gpu) ≈ S_ref atol=1.0f-6 rtol=1.0f-6
            end

            # zero multiplicative term => S[:,t] = B[:,t]
            let
                D, T = 7, 25
                A_cpu = zeros(Float32, D, T)
                B_cpu = randn(rng, Float32, D, T)
                s0_cpu = randn(rng, Float32, D)

                S_ref = DEERScan.solve_affine_seq(A_cpu, B_cpu, s0_cpu)
                S_gpu = DEERScan.solve_affine_scan_diag(
                    CUDA.CuArray(A_cpu), CUDA.CuArray(B_cpu), CUDA.CuArray(s0_cpu)
                )

                @test Array(S_gpu) ≈ S_ref atol=1.0f-6 rtol=1.0f-6
                @test Array(S_gpu) ≈ B_cpu atol=1.0f-6 rtol=1.0f-6
            end

            # T = 1
            let
                D, T = 9, 1
                A_cpu = randn(rng, Float32, D, T) .* 0.1f0
                B_cpu = randn(rng, Float32, D, T)
                s0_cpu = randn(rng, Float32, D)

                S_ref = DEERScan.solve_affine_seq(A_cpu, B_cpu, s0_cpu)
                S_gpu = DEERScan.solve_affine_scan_diag(
                    CUDA.CuArray(A_cpu), CUDA.CuArray(B_cpu), CUDA.CuArray(s0_cpu)
                )

                @test size(S_gpu) == (D, 1)
                @test Array(S_gpu) ≈ S_ref atol=1.0f-6 rtol=1.0f-6
            end
        end

        @testset "GPU scan preserves Float32 and stays on device" begin
            rng = MersenneTwister(5)

            D, T = 8, 32
            A_cpu, B_cpu, s0_cpu = make_affine_problem(rng, D, T; FT=Float32, stable=true)

            A_gpu = CUDA.CuArray(A_cpu)
            B_gpu = CUDA.CuArray(B_cpu)
            s0_gpu = CUDA.CuArray(s0_cpu)

            S_gpu = DEERScan.solve_affine_scan_diag(A_gpu, B_gpu, s0_gpu)

            @test S_gpu isa CUDA.CuMatrix
            @test eltype(S_gpu) == Float32
        end

        @testset "GPU scan large test" begin
            rng = MersenneTwister(6)

            D, T = 128, 1024
            A_cpu, B_cpu, s0_cpu = make_affine_problem(rng, D, T; FT=Float32, stable=true)

            A_gpu = CUDA.CuArray(A_cpu)
            B_gpu = CUDA.CuArray(B_cpu)
            s0_gpu = CUDA.CuArray(s0_cpu)

            S_gpu = DEERScan.solve_affine_scan_diag(A_gpu, B_gpu, s0_gpu)

            @test size(S_gpu) == (D, T)
            @test eltype(S_gpu) == Float32
            @test all(isfinite, Array(S_gpu[:, 1:8]))
        end

        @testset "GPU scan shape validation" begin
            rng = MersenneTwister(7)

            A_cpu, B_cpu, s0_cpu = make_affine_problem(rng, 4, 10; FT=Float32, stable=true)

            A_gpu = CUDA.CuArray(A_cpu)
            B_bad_gpu = CUDA.CuArray(randn(Float32, 4, 9))
            s0_gpu = CUDA.CuArray(s0_cpu)

            @test_throws DimensionMismatch DEERScan.solve_affine_scan_diag(
                A_gpu, B_bad_gpu, s0_gpu
            )

            B_gpu = CUDA.CuArray(B_cpu)
            s0_bad_gpu = CUDA.CuArray(randn(Float32, 5))

            @test_throws DimensionMismatch DEERScan.solve_affine_scan_diag(
                A_gpu, B_gpu, s0_bad_gpu
            )
        end

        @testset "GPU scan helper check_affine_scan still agrees with CPU" begin
            rng = MersenneTwister(8)

            D, T = 4, 24
            A_cpu, B_cpu, s0_cpu = make_affine_problem(rng, D, T; FT=Float32, stable=true)

            chk = DEERScan.check_affine_scan(A_cpu, B_cpu, s0_cpu; atol=1.0f-6, rtol=1.0f-6)

            @test chk.ok
            @test chk.max_abs_err ≤ 1.0f-6
            @test chk.residual_seq ≤ 1.0f-6
            @test chk.residual_scan ≤ 1.0f-6
        end
    end
end
