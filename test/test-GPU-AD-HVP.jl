using Test
using Random
using LinearAlgebra
using Statistics

using ParallelMCMC
using ADTypes: ADTypes
using Enzyme
using Mooncake
using Zygote
using CUDA: CUDA

#=
Coverage for the GPU AD-HVP path: when the user supplies a non-Turing model
with `gradlogp` / `grad_logdensity_batch` but no analytical `hvp`/`hvp_batch`,
the sampler should fall back to AD on the user's gradient and run on GPU
without crashing.
=#

const _ADHVP_GPU_AVAILABLE = try
    CUDA.functional() && (CUDA.CuArray([1.0f0]); true)
catch
    false
end

if !_ADHVP_GPU_AVAILABLE
    @info "GPU AD-HVP test: CUDA not functional — skipping"
else

    #=
    Multivariate Gaussian target with X'X/N perturbation:
      logp(β) = -0.5 (||β||^2 + ||Xβ||^2 / N)
    True mean = 0; we'll check the posterior mean is near zero.

    Calls go through `pmcmc_matmul` so Enzyme dispatches to the rule in
    `ext/EnzymeExt.jl`, which keeps the op off Enzyme's reverse path and
    away from the gc-transition abort documented there.
    =#
    function _logp_single(β, X)
        Xβ = pmcmc_matmul(X, β)
        N = oftype(zero(eltype(β)), size(X, 1))
        -oftype(zero(eltype(β)), 0.5) * (sum(abs2, β) + sum(abs2, Xβ) / N)
    end

    function _gradlogp_single(β, X)
        Xβ = pmcmc_matmul(X, β)
        N = oftype(zero(eltype(β)), size(X, 1))
        Y = pmcmc_matmul(transpose(X), Xβ)
        Y = Y ./ N
        Y = Y .+ β
        return -Y
    end

    function _logp_batch(B, X)
        XB = pmcmc_matmul(X, B)
        N = oftype(zero(eltype(B)), size(X, 1))
        -oftype(zero(eltype(B)), 0.5) .*
        (vec(sum(abs2, B; dims=1)) .+ vec(sum(abs2, XB; dims=1)) ./ N)
    end

    function _gradlogp_batch(B, X)
        XB = pmcmc_matmul(X, B)
        N = oftype(zero(eltype(B)), size(X, 1))
        Y = pmcmc_matmul(transpose(X), XB)
        Y = Y ./ N
        Y = Y .+ B
        return -Y
    end

    @testset "GPU AD-HVP: ParallelMALASampler runs without analytical HVP" begin
        D = 20
        N_data = 64
        rng = MersenneTwister(20251231)
        X_cpu = randn(rng, Float32, N_data, D)
        X_gpu = CUDA.CuMatrix(X_cpu)

        model = DensityModel(
            β -> _logp_single(β, X_gpu),
            β -> _gradlogp_single(β, X_gpu),
            D;
            logdensity_batch=B -> _logp_batch(B, X_gpu),
            grad_logdensity_batch=B -> _gradlogp_batch(B, X_gpu),
        )

        sampler = ParallelMALASampler(
            0.05f0;
            T=16,
            maxiter=200,
            tol_abs=1.0f-3,
            tol_rel=1.0f-2,
            damping=0.5f0,
            backend=ADTypes.AutoEnzyme(),
        )

        n_samples, n_burn = 1600, 400
        x0 = CUDA.zeros(Float32, D)

        raw = sample(
            MersenneTwister(42),
            model,
            sampler,
            n_samples;
            initial_params=x0,
            progress=false,
        )
        β_post = vec(
            mean(reduce(hcat, [Array(s.x) for s in raw[(n_burn + 1):end]]); dims=2)
        )

        @test all(isfinite, β_post)
        #=
        Gaussian target → posterior mean is zero. Loose tolerance accounts for MC
        variance with N_eff ~ 100-200; an actual divergence would blow past this.
        =#
        @test maximum(abs, β_post) < 0.4
    end

    # @testset "GPU AD-HVP: matches CPU AD-HVP" begin
    #     D = 12
    #     N_data = 48
    #     rng = MersenneTwister(20251231)
    #     X_cpu = randn(rng, Float32, N_data, D)
    #     X_gpu = CUDA.CuMatrix(X_cpu)

    #     model_cpu = DensityModel(
    #         β -> _logp_single(β, X_cpu),
    #         β -> _gradlogp_single(β, X_cpu),
    #         D;
    #         logdensity_batch=B -> _logp_batch(B, X_cpu),
    #         grad_logdensity_batch=B -> _gradlogp_batch(B, X_cpu),
    #     )
    #     model_gpu = DensityModel(
    #         β -> _logp_single(β, X_gpu),
    #         β -> _gradlogp_single(β, X_gpu),
    #         D;
    #         logdensity_batch=B -> _logp_batch(B, X_gpu),
    #         grad_logdensity_batch=B -> _gradlogp_batch(B, X_gpu),
    #     )

    #     sampler = ParallelMALASampler(
    #         0.05f0;
    #         T=16,
    #         maxiter=200,
    #         tol_abs=1.0f-3,
    #         tol_rel=1.0f-2,
    #         damping=0.5f0,
    #         backend=ADTypes.AutoEnzyme(),
    #     )

    #     n_samples, n_burn = 4000, 1000
    #     raw_cpu = sample(MersenneTwister(7), model_cpu, sampler, n_samples; progress=false)
    #     β_cpu = vec(mean(reduce(hcat, [s.x for s in raw_cpu[(n_burn + 1):end]]); dims=2))

    #     raw_gpu = sample(
    #         MersenneTwister(7),
    #         model_gpu,
    #         sampler,
    #         n_samples;
    #         initial_params=CUDA.zeros(Float32, D),
    #         progress=false,
    #     )
    #     β_gpu = vec(
    #         mean(reduce(hcat, [Array(s.x) for s in raw_gpu[(n_burn + 1):end]]); dims=2)
    #     )

    #     @test maximum(abs, β_cpu .- β_gpu) < 0.25
    # end

    #=
    Mooncake on GPU. The whole point of routing through DI is that swapping
    the backend should keep the same `ParallelMALASampler` API working.
    `_hvp_strategy(::AutoMooncake) = ReverseOnGrad()`, so this exercises the
    reverse-on-grad path through Mooncake's CUDA extension (no custom rules
    of ours involved — Mooncake's defaults handle `*`, broadcast, `sum`,
    `dot` on `CuArray` directly).
    =#
    @testset "GPU AD-HVP: ParallelMALASampler runs without analytical HVP (Mooncake)" begin
        D = 20
        N_data = 64
        rng = MersenneTwister(20251231)
        X_cpu = randn(rng, Float32, N_data, D)
        X_gpu = CUDA.CuMatrix(X_cpu)

        model = DensityModel(
            β -> _logp_single(β, X_gpu),
            β -> _gradlogp_single(β, X_gpu),
            D;
            logdensity_batch=B -> _logp_batch(B, X_gpu),
            grad_logdensity_batch=B -> _gradlogp_batch(B, X_gpu),
        )

        sampler = ParallelMALASampler(
            0.05f0;
            T=16,
            maxiter=200,
            tol_abs=1.0f-3,
            tol_rel=1.0f-2,
            damping=0.5f0,
            backend=ADTypes.AutoMooncake(; config=nothing),
        )

        n_samples, n_burn = 1600, 400
        x0 = CUDA.zeros(Float32, D)

        raw = sample(
            MersenneTwister(42),
            model,
            sampler,
            n_samples;
            initial_params=x0,
            progress=false,
        )
        β_post = vec(
            mean(reduce(hcat, [Array(s.x) for s in raw[(n_burn + 1):end]]); dims=2)
        )

        @test all(isfinite, β_post)
        # Same Gaussian target as the AutoEnzyme test; same loose tolerance.
        @test maximum(abs, β_post) < 0.4
    end

    #=
    Zygote on GPU. Same shape as the Mooncake test: ReverseOnGrad strategy,
    no custom rules of ours involved — Zygote recurses into `pmcmc_matmul`
    and differentiates `A * B` via ChainRules' cuBLAS-backed adjoints.
    =#
    @testset "GPU AD-HVP: ParallelMALASampler runs without analytical HVP (Zygote)" begin
        D = 20
        N_data = 64
        rng = MersenneTwister(20251231)
        X_cpu = randn(rng, Float32, N_data, D)
        X_gpu = CUDA.CuMatrix(X_cpu)

        model = DensityModel(
            β -> _logp_single(β, X_gpu),
            β -> _gradlogp_single(β, X_gpu),
            D;
            logdensity_batch=B -> _logp_batch(B, X_gpu),
            grad_logdensity_batch=B -> _gradlogp_batch(B, X_gpu),
        )

        sampler = ParallelMALASampler(
            0.05f0;
            T=16,
            maxiter=200,
            tol_abs=1.0f-3,
            tol_rel=1.0f-2,
            damping=0.5f0,
            backend=ADTypes.AutoZygote(),
        )

        n_samples, n_burn = 1600, 400
        x0 = CUDA.zeros(Float32, D)

        raw = sample(
            MersenneTwister(42),
            model,
            sampler,
            n_samples;
            initial_params=x0,
            progress=false,
        )
        β_post = vec(
            mean(reduce(hcat, [Array(s.x) for s in raw[(n_burn + 1):end]]); dims=2)
        )

        @test all(isfinite, β_post)
        # Same Gaussian target as the AutoEnzyme/AutoMooncake tests; same tolerance.
        @test maximum(abs, β_post) < 0.4
    end
end # _ADHVP_GPU_AVAILABLE
