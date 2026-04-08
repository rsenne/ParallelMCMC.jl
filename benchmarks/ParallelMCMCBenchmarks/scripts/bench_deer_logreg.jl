"""
Throughput benchmark: GPU ParallelMALASampler vs CPU AdaptiveMALA on Bayesian logistic regression.

The key comparison is GPU DEER vs CPU MALA: DEER parallelises T MALA steps into
an O(log T) parallel-prefix scan on the GPU, so each solve produces T new samples
at roughly the cost of O(log T) sequential MALA steps on GPU hardware.

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

# Problem Setup

rng = MersenneTwister(20251231)
D, N_data = 10, 200

# Float32 throughout
X_f32, y_f32, _ = BayesLogReg.make_data(rng, N_data, D)
X_f32 = Float32.(X_f32)
y_f32 = Float32.(y_f32)

logp_cpu = β -> BayesLogReg.make_problem(X_f32, y_f32)[1](β)
gradlogp_cpu = β -> BayesLogReg.make_problem(X_f32, y_f32)[2](β)

# Rebuild closures once
_logp_cpu, _gradlogp_cpu = BayesLogReg.make_problem(X_f32, y_f32)

model_cpu = DensityModel(_logp_cpu, _gradlogp_cpu, D)

deer_cpu = ParallelMALASampler(
    0.1f0;
    T=16,
    maxiter=50,
    tol_abs=1.0f-4,
    tol_rel=1.0f-3,
    damping=0.5f0,
)
mala_cpu = AdaptiveMALASampler(0.1f0; n_warmup=500)

# CUDA setup

const _cuda_ok = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !_cuda_ok
    @warn "CUDA not available — GPU benchmarks will be skipped"
end

# Benchmark helper

function run_bench(model, sampler, n_samples; reps, label, device, sampler_name, x0=nothing)
    println("  [$device] $sampler_name — $label samples:")
    kw = x0 === nothing ? (progress=false,) : (progress=false, initial_params=x0)
    b = @benchmark(
        sample(MersenneTwister(42), $model, $sampler, $n_samples; $kw...),
        samples=reps,
        evals=1,
    )
    show(stdout, MIME("text/plain"), b)
    println("\n")
    return b
end

configs = [(1_000, 8, "1k"), (10_000, 4, "10k"), (100_000, 3, "100k")]

results = Dict{Tuple{String,String},BenchmarkTools.Trial}()

# CPU MALA baseline

println("=" ^ 65)
println("CPU AdaptiveMALA baseline  (n_warmup=500, Float32)")
println("Model: Bayesian logistic regression  D=$D  N_data=$N_data")
println("=" ^ 65, "\n")

for (n_samples, reps, label) in configs
    results[("CPU-MALA", label)] = run_bench(
        model_cpu, mala_cpu, n_samples; reps, label, device="CPU", sampler_name="MALA"
    )
end

# CPU DEER

println("=" ^ 65)
println("CPU ParallelMALASampler  (T=16, AutoEnzyme, Float32)")
println("Model: Bayesian logistic regression  D=$D  N_data=$N_data")
println("=" ^ 65, "\n")

for (n_samples, reps, label) in configs
    results[("CPU-DEER", label)] = run_bench(
        model_cpu, deer_cpu, n_samples; reps, label, device="CPU", sampler_name="DEER"
    )
end

# GPU DEER

if _cuda_ok
    X_gpu = CUDA.CuMatrix(X_f32)
    y_gpu = CUDA.CuVector(y_f32)
    _logp_gpu, _gradlogp_gpu = BayesLogReg.make_problem(X_gpu, y_gpu)
    _logp_gpu_batch, _gradlogp_gpu_batch = BayesLogReg.make_problem_batched(X_gpu, y_gpu)

    # Provide batched functions so deer_update uses the GPU-efficient batched path
    # (all T trajectory steps processed simultaneously via D×T matrix ops).
    model_gpu = DensityModel(
        _logp_gpu, _gradlogp_gpu, D;
        logdensity_batch=_logp_gpu_batch,
        grad_logdensity_batch=_gradlogp_gpu_batch,
    )

    deer_gpu = ParallelMALASampler(
        0.1f0;
        T=16,
        maxiter=50,
        tol_abs=1.0f-4,
        tol_rel=1.0f-3,
        damping=0.5f0,
    )

    x0_gpu = CUDA.CuArray(zeros(Float32, D))

    println("=" ^ 65)
    println("GPU ParallelMALASampler  (T=16, batched, Float32, CuArrays)")
    println("Model: Bayesian logistic regression  D=$D  N_data=$N_data")
    println("=" ^ 65, "\n")

    for (n_samples, reps, label) in configs
        results[("GPU-DEER", label)] = run_bench(
            model_gpu,
            deer_gpu,
            n_samples;
            reps,
            label,
            device="GPU",
            sampler_name="DEER",
            x0=x0_gpu,
        )
    end
end

# Summary table

println("=" ^ 65)
println("Summary: median wall-clock time (ms)")
println("=" ^ 65)

if _cuda_ok
    @printf "%-10s  %12s  %12s  %12s  %12s  %12s\n" "N samples" "CPU-MALA" "CPU-DEER" "GPU-DEER" "GPU/CPU-MALA" "GPU/CPU-DEER"
    for (_, _, label) in configs
        t_mala = median(results[("CPU-MALA", label)]).time / 1e6
        t_cpu_deer = median(results[("CPU-DEER", label)]).time / 1e6
        t_gpu_deer = median(results[("GPU-DEER", label)]).time / 1e6
        @printf "%-10s  %12.1f  %12.1f  %12.1f  %12.2fx  %12.2fx\n" label t_mala t_cpu_deer t_gpu_deer (
            t_gpu_deer/t_mala
        ) (t_gpu_deer/t_cpu_deer)
    end
else
    @printf "%-10s  %12s  %12s  %12s\n" "N samples" "CPU-MALA" "CPU-DEER" "DEER/MALA"
    for (_, _, label) in configs
        t_mala = median(results[("CPU-MALA", label)]).time / 1e6
        t_cpu_deer = median(results[("CPU-DEER", label)]).time / 1e6
        @printf "%-10s  %12.1f  %12.1f  %12.2fx\n" label t_mala t_cpu_deer (
            t_cpu_deer/t_mala
        )
    end
end
println()
