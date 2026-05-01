using Test
using Random
using LinearAlgebra
using Statistics
using MCMCChains

using ParallelMCMC

using DynamicPPL
using LogDensityProblems
using LogDensityProblemsAD
using ADTypes
using Distributions: MvNormal, Bernoulli

#=
Bayesian logistic regression:
    β ~ MvNormal(0, I_D)
    y_i | β ~ Bernoulli(sigmoid(X_i β))

Synthetic data with known β_true lets us verify posterior means.

The tests are split into two groups:

Turing integration: uses the DynamicPPL convenience constructor.  These are
CPU-only because DynamicPPL model evaluation does not support GPU arrays.
ParallelMALASampler works fine with Turing on CPU; for GPU you supply a manual
logp/gradlogp that is array-type-agnostic (see below).
=#

const _LR_D = 2
const _LR_N = 60
const _LR_β_true = [1.0, -0.8]

function _lr_data(rng, N, D, β_true)
    X = randn(rng, N, D)
    logits = X * β_true
    y = [rand(rng) < 1 / (1 + exp(-l)) ? 1 : 0 for l in logits]
    return X, y
end

const _LR_X, _LR_y = _lr_data(MersenneTwister(1234), _LR_N, _LR_D, _LR_β_true)

# Array-type-agnostic logp and gradlogp — identical math as the Turing model,
# but expressed as plain broadcasts so they run on CPU or GPU without change.
function _logp_lr(β, X, y)
    logits = X * β
    ll = sum(@. y * (-log1p(exp(-logits))) + (1 - y) * (-log1p(exp(logits))))
    return ll - oftype(ll, 0.5) * sum(abs2, β)
end

function _gradlogp_lr(β, X, y)
    logits = X * β
    p = @. 1 / (1 + exp(-logits))
    return X' * (y .- p) .- β
end

function _hvp_lr(β, v, X, y)
    logits = X * β
    p = @. 1 / (1 + exp(-logits))
    w = @. p * (1 - p)
    return -(X' * (w .* (X * v))) .- v
end

@model function _deer_logistic_regression(X, y)
    D = size(X, 2)
    β ~ MvNormal(zeros(D), I)
    for i in eachindex(y)
        p_i = 1 / (1 + exp(-dot(view(X, i, :), β)))
        y[i] ~ Bernoulli(p_i)
    end
end

function _deer_logistic_turing_density_model()
    return DensityModel(
        _deer_logistic_regression(_LR_X, _LR_y);
        hvp=(β, v) -> _hvp_lr(β, v, _LR_X, _LR_y),
    )
end

# Turing integration tests (CPU)

@testset "ParallelMALASampler Turing logistic: param names extracted correctly" begin
    model = _deer_logistic_turing_density_model()

    @test model.dim == _LR_D
    @test model.param_names == [Symbol("β[1]"), Symbol("β[2]")]
    @test isfinite(model.logdensity(zeros(_LR_D)))
    @test all(isfinite, model.grad_logdensity(zeros(_LR_D)))
end

@testset "ParallelMALASampler Turing logistic: chains output well-formed" begin
    model = _deer_logistic_turing_density_model()
    sampler = ParallelMALASampler(
        0.05; T=16, maxiter=80, tol_abs=1e-4, tol_rel=1e-3, jacobian=:diag, damping=0.5
    )

    chain = sample(
        MersenneTwister(42),
        model,
        sampler,
        400;
        initial_params=zeros(_LR_D),
        chain_type=MCMCChains.Chains,
        progress=false,
    )

    @test chain isa MCMCChains.Chains
    @test size(chain, 1) == 400
    @test Symbol("β[1]") in names(chain, :parameters)
    @test Symbol("β[2]") in names(chain, :parameters)
    @test all(isfinite, Array(chain[:, [Symbol("β[1]"), Symbol("β[2]")], :]))
end

@testset "ParallelMALASampler Turing logistic: posterior sign correct" begin
    model = _deer_logistic_turing_density_model()
    sampler = ParallelMALASampler(
        0.05; T=16, maxiter=80, tol_abs=1e-4, tol_rel=1e-3, jacobian=:diag, damping=0.5
    )

    chain = sample(
        MersenneTwister(99),
        model,
        sampler,
        800;
        initial_params=zeros(_LR_D),
        chain_type=MCMCChains.Chains,
        progress=false,
    )

    post = Array(chain[201:end, [Symbol("β[1]"), Symbol("β[2]")], 1])
    β_mean = vec(mean(post; dims=1))

    @test sign(β_mean[1]) == sign(_LR_β_true[1])
    @test sign(β_mean[2]) == sign(_LR_β_true[2])
    @test abs(β_mean[1] - _LR_β_true[1]) < 0.6
    @test abs(β_mean[2] - _LR_β_true[2]) < 0.6
end

# Statistical correctness: DEER posterior mean matches AdaptiveMALA reference.

@testset "ParallelMALASampler logistic: posterior mean matches AdaptiveMALA reference" begin
    X64, y64 = Float64.(_LR_X), Float64.(_LR_y)
    model = DensityModel(
        β -> _logp_lr(β, X64, y64),
        β -> _gradlogp_lr(β, X64, y64),
        _LR_D;
        hvp=(β, v) -> _hvp_lr(β, v, X64, y64),
    )

    mala_chain = sample(
        MersenneTwister(2025),
        model,
        AdaptiveMALASampler(0.1; n_warmup=1000),
        5000;
        chain_type=MCMCChains.Chains,
        progress=false,
        discard_warmup=true,
    )
    β_mala = vec(mean(Array(mala_chain[:, [Symbol("x[1]"), Symbol("x[2]")], 1]); dims=1))

    deer_chain = sample(
        MersenneTwister(42),
        model,
        ParallelMALASampler(
            0.1;
            T=16,
            maxiter=50,
            tol_abs=1e-4,
            tol_rel=1e-3,
            damping=0.5,
            backend=ADTypes.AutoEnzyme(),
        ),
        800;
        chain_type=MCMCChains.Chains,
        progress=false,
    )
    β_deer = vec(
        mean(Array(deer_chain[201:end, [Symbol("x[1]"), Symbol("x[2]")], 1]); dims=1)
    )

    @test abs(β_deer[1] - β_mala[1]) < 0.25
    @test abs(β_deer[2] - β_mala[2]) < 0.25
end

# GPU tests — same logistic regression model, manual logp/gradlogp on CuArrays.

_cuda_ok_lr = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !_cuda_ok_lr
    @warn "CUDA not available — skipping GPU logistic regression DEER tests"
else
    _X_f32 = Float32.(_LR_X)
    _y_f32 = Float32.(_LR_y)

    @testset "ParallelMALASampler GPU logistic: trajectory matches sequential MALA (same tape)" begin
        rng = MersenneTwister(77)
        T = 20
        ε = 0.05f0

        ξs_cpu = [randn(rng, Float32, _LR_D) for _ in 1:T]
        us = Float32.(rand(rng, T))
        x0_cpu = randn(rng, Float32, _LR_D)

        logp_cpu = β -> _logp_lr(β, _X_f32, _y_f32)
        gradlogp_cpu = β -> _gradlogp_lr(β, _X_f32, _y_f32)

        xs_seq = ParallelMCMC.MALA.run_mala_sequential_taped(
            logp_cpu, gradlogp_cpu, x0_cpu, ε, ξs_cpu, us
        )

        X_gpu = CUDA.CuMatrix(_X_f32)
        y_gpu = CUDA.CuVector(_y_f32)
        logp_gpu = β -> _logp_lr(β, X_gpu, y_gpu)
        gradlogp_gpu = β -> _gradlogp_lr(β, X_gpu, y_gpu)

        tape = [(ξ=CUDA.CuArray(ξs_cpu[t]), u=us[t]) for t in 1:T]
        step_fwd =
            (x, te) ->
                ParallelMCMC.MALA.mala_step_taped(logp_gpu, gradlogp_gpu, x, ε, te.ξ, te.u)
        hvp_fn_gpu = (pt, dir) -> _hvp_lr(pt, dir, X_gpu, y_gpu)
        jvp =
            (x, te, v) -> ParallelMCMC.MALA.mala_step_surrogate_sigmoid_jvp(
                logp_gpu, gradlogp_gpu, x, ε, te.ξ, te.u, v, hvp_fn_gpu
            )
        fwd_and_jvp =
            (x, te, v) -> ParallelMCMC.MALA.mala_step_taped_and_jvp(
                logp_gpu, gradlogp_gpu, x, ε, te.ξ, te.u, v, hvp_fn_gpu
            )

        rec = ParallelMCMC.DEER.TapedRecursion(step_fwd, jvp, tape; fwd_and_jvp=fwd_and_jvp)

        x0_gpu = CUDA.CuArray(x0_cpu)
        S_gpu, info = ParallelMCMC.DEER.solve(
            rec,
            x0_gpu;
            jacobian=:diag,
            damping=0.5,
            tol_abs=1e-5,
            tol_rel=1e-4,
            maxiter=200,
            return_info=true,
        )

        @test info.converged
        @test S_gpu isa CUDA.CuMatrix
        @test size(S_gpu) == (_LR_D, T)
        @test all(isfinite, Array(S_gpu))
        S_ref = reduce(hcat, xs_seq[2:end])
        @test Array(S_gpu) ≈ S_ref rtol=1e-4 atol=1e-5
    end

    @testset "ParallelMALASampler GPU logistic: posterior mean matches CPU" begin
        X_gpu = CUDA.CuMatrix(_X_f32)
        y_gpu = CUDA.CuVector(_y_f32)

        sampler = ParallelMALASampler(
            0.1f0;
            T=16,
            maxiter=50,
            tol_abs=1e-4f0,
            tol_rel=1e-3f0,
            damping=0.5f0,
            backend=ADTypes.AutoEnzyme(),
        )

        model_cpu = DensityModel(
            β -> _logp_lr(β, _X_f32, _y_f32),
            β -> _gradlogp_lr(β, _X_f32, _y_f32),
            _LR_D;
            hvp=(β, v) -> _hvp_lr(β, v, _X_f32, _y_f32),
        )
        model_gpu = DensityModel(
            β -> _logp_lr(β, X_gpu, y_gpu),
            β -> _gradlogp_lr(β, X_gpu, y_gpu),
            _LR_D;
            hvp=(β, v) -> _hvp_lr(β, v, X_gpu, y_gpu),
        )

        n_samples, n_burn = 400, 100

        raw_cpu = sample(MersenneTwister(42), model_cpu, sampler, n_samples; progress=false)
        β_cpu = vec(mean(reduce(hcat, [s.x for s in raw_cpu[(n_burn + 1):end]]); dims=2))

        raw_gpu = sample(
            MersenneTwister(42),
            model_gpu,
            sampler,
            n_samples;
            initial_params=CUDA.CuArray(zeros(Float32, _LR_D)),
            progress=false,
        )
        β_gpu = vec(
            mean(reduce(hcat, [Array(s.x) for s in raw_gpu[(n_burn + 1):end]]); dims=2)
        )

        @test all(isfinite, β_gpu)
        @test abs(β_gpu[1] - β_cpu[1]) < 0.15
        @test abs(β_gpu[2] - β_cpu[2]) < 0.15
    end
end # _cuda_ok_lr
