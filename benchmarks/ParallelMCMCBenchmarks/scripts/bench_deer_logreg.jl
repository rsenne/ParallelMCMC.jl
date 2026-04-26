"""
End-to-end throughput sweep: GPU ParallelMALASampler vs CPU AdaptiveMALA on
Bayesian logistic regression.

This benchmark differs from `new_bench.jl`:

- `new_bench.jl` isolates the raw prebuilt DEER block solve and sweeps T.
- this script times the public `sample(...)` path and sweeps the Parallel MALA
  block length T, which is the number you tune for user-facing runs.

Problem: D=10 features, N=200 observations (synthetic Float32 data).
Measures wall-clock time to collect 1k, 10k, and 100k posterior samples.

Run from the benchmarks/ParallelMCMCBenchmarks directory:
    julia --project scripts/bench_deer_logreg.jl
"""

using Random
using BenchmarkTools
using LinearAlgebra
using Printf
using Statistics

using AbstractMCMC: sample
using ParallelMCMC
using ParallelMCMCBenchmarks

const BayesLogReg = ParallelMCMCBenchmarks.BayesLogReg

function _parse_t_vals()
    raw = get(ENV, "PMCMC_T_VALS", "")
    isempty(strip(raw)) && return [128, 256, 512, 1024, 2048]
    return parse.(Int, strip.(split(raw, ",")))
end

function _label(n::Int)
    n % 1000 == 0 && return "$(div(n, 1000))k"
    return string(n)
end

function _bench_sample(model, sampler, n_samples; reps, x0=nothing)
    kw = x0 === nothing ? (progress=false,) : (progress=false, initial_params=x0)
    return @benchmark(
        sample(MersenneTwister(42), $model, $sampler, $n_samples; $kw...),
        samples=reps,
        evals=1,
    )
end

# Problem setup

rng = MersenneTwister(20251231)
D, N_data = 10, 200

X_f32, y_f32, _ = BayesLogReg.make_data(rng, N_data, D)
X_f32 = Float32.(X_f32)
y_f32 = Float32.(y_f32)

logp_cpu, gradlogp_cpu, hvp_cpu = BayesLogReg.make_problem_with_hvp(X_f32, y_f32)
model_cpu = DensityModel(logp_cpu, gradlogp_cpu, D; hvp=hvp_cpu)

epsilon = 0.1f0
maxiter = 50
tol_abs = 1.0f-4
tol_rel = 1.0f-3
damping = 0.5f0
probes = 1
n_warmup = 500

sample_configs = [(1_000, 8), (10_000, 4), (100_000, 3)]
T_vals = _parse_t_vals()

mala_cpu = AdaptiveMALASampler(epsilon; n_warmup=n_warmup)

const _cuda_ok = get(ENV, "PMCMC_SKIP_GPU", "0") == "1" ? false : try
    using CUDA
    CUDA.functional()
catch
    false
end

if !_cuda_ok
    @warn "CUDA not available - GPU benchmarks will be skipped"
end

println("=" ^ 86)
println("End-to-end ParallelMALASampler T sweep")
println("Model: Bayesian logistic regression  D=$D  N_data=$N_data  Float32")
println("epsilon=$epsilon  maxiter=$maxiter  tol_abs=$tol_abs  tol_rel=$tol_rel")
println("T values: $(join(T_vals, ", "))")
println("=" ^ 86)
println()

cpu_results = Dict{String,BenchmarkTools.Trial}()
gpu_results = Dict{Tuple{String,Int},BenchmarkTools.Trial}()

println("CPU baseline: AdaptiveMALA (n_warmup=$n_warmup)")
for (n_samples, reps) in sample_configs
    label = _label(n_samples)
    println("  [CPU] $label samples")
    trial = _bench_sample(model_cpu, mala_cpu, n_samples; reps=reps)
    show(stdout, MIME("text/plain"), trial)
    println("\n")
    cpu_results[label] = trial
end

if _cuda_ok
    X_gpu = CUDA.CuMatrix(X_f32)
    y_gpu = CUDA.CuVector(y_f32)

    logp_gpu, gradlogp_gpu, hvp_gpu = BayesLogReg.make_problem_with_hvp(X_gpu, y_gpu)
    logp_gpu_batch, gradlogp_gpu_batch, hvp_gpu_batch =
        BayesLogReg.make_problem_batched_with_hvp(X_gpu, y_gpu)

    model_gpu = DensityModel(
        logp_gpu,
        gradlogp_gpu,
        D;
        hvp=hvp_gpu,
        logdensity_batch=logp_gpu_batch,
        grad_logdensity_batch=gradlogp_gpu_batch,
        hvp_batch=hvp_gpu_batch,
    )

    x0_gpu = CUDA.CuArray(zeros(Float32, D))

    println("GPU sweep: ParallelMALASampler")
    for T in T_vals
        deer_gpu = ParallelMALASampler(
            epsilon;
            T=T,
            maxiter=maxiter,
            tol_abs=tol_abs,
            tol_rel=tol_rel,
            damping=damping,
            probes=probes,
        )

        println("  T=$T")
        for (n_samples, reps) in sample_configs
            label = _label(n_samples)
            blocks = cld(n_samples, T)
            println("    [GPU] $label samples ($blocks blocks)")
            trial = _bench_sample(model_gpu, deer_gpu, n_samples; reps=reps, x0=x0_gpu)
            show(stdout, MIME("text/plain"), trial)
            println("\n")
            gpu_results[(label, T)] = trial
        end
    end
end

println("=" ^ 112)
println("Summary: median wall-clock time")
println("=" ^ 112)

if _cuda_ok
    @printf(
        "%-8s  %8s  %8s  %12s  %12s  %12s  %12s  %12s\n",
        "samples",
        "T",
        "blocks",
        "CPU ms",
        "GPU ms",
        "GPU us/samp",
        "GPU ms/block",
        "CPU/GPU",
    )

    for (n_samples, _) in sample_configs
        label = _label(n_samples)
        cpu_ms = median(cpu_results[label]).time / 1e6
        for T in T_vals
            gpu_ms = median(gpu_results[(label, T)]).time / 1e6
            blocks = cld(n_samples, T)
            gpu_us_per_sample = 1000 * gpu_ms / n_samples
            gpu_ms_per_block = gpu_ms / blocks
            speedup = cpu_ms / gpu_ms
            @printf(
                "%-8s  %8d  %8d  %12.1f  %12.1f  %12.3f  %12.3f  %12.2fx\n",
                label,
                T,
                blocks,
                cpu_ms,
                gpu_ms,
                gpu_us_per_sample,
                gpu_ms_per_block,
                speedup,
            )
        end
    end

    println()
    println("Best T by GPU time per sample")
    @printf "%-8s  %8s  %12s  %12s\n" "samples" "best T" "GPU ms" "CPU/GPU"
    for (n_samples, _) in sample_configs
        label = _label(n_samples)
        cpu_ms = median(cpu_results[label]).time / 1e6
        best_T = first(T_vals)
        best_ms = Inf
        for T in T_vals
            gpu_ms = median(gpu_results[(label, T)]).time / 1e6
            if gpu_ms < best_ms
                best_T = T
                best_ms = gpu_ms
            end
        end
        @printf "%-8s  %8d  %12.1f  %12.2fx\n" label best_T best_ms (cpu_ms / best_ms)
    end
else
    @printf "%-8s  %12s\n" "samples" "CPU ms"
    for (n_samples, _) in sample_configs
        label = _label(n_samples)
        cpu_ms = median(cpu_results[label]).time / 1e6
        @printf "%-8s  %12.1f\n" label cpu_ms
    end
end
println()
