"""
Throughput benchmark: AdaptiveMALASampler vs ParallelMALASampler (DEER) on
Bayesian linear regression.

Problem: D=16 features, N=200 observations (synthetic Float64 data).
Measures wall-clock time to collect 1k, 5k, and 10k posterior samples.

Run from the benchmarks/ParallelMCMCBenchmarks directory:
    julia --project scripts/bench_mala_bayes.jl
"""

using Random
using BenchmarkTools
using LinearAlgebra
using Printf
using Statistics

using AbstractMCMC: sample
using ParallelMCMC
using ParallelMCMCBenchmarks
const BayesLinReg = ParallelMCMCBenchmarks.BayesLinReg
const MALARunner = ParallelMCMCBenchmarks.MALARunner

# Problem setup

rng = MersenneTwister(20251231)
n, p = 200, 16
X = randn(rng, n, p)
β_true = randn(rng, p)
σ = 1.0
y = X * β_true .+ σ .* randn(rng, n)

logpost, gradlogpost, μ_post, _ = BayesLinReg.make_problem(X, y; σ=σ, τ=10.0)
model = DensityModel(logpost, gradlogpost, p)

# Step-size warmup via low-level runner

println("Tuning step size...")
x_warm, ϵ_tuned = MALARunner.tune_stepsize_mala(
    logpost, gradlogpost, zeros(p), 1e-2; Twarm=3_000, rng=rng
)
@printf "  Tuned ϵ = %.4f\n\n" ϵ_tuned

# Samplers

mala_sampler = AdaptiveMALASampler(ϵ_tuned; n_warmup=500)

deer_sampler = ParallelMALASampler(
    ϵ_tuned; T=64, maxiter=200, tol_abs=1e-6, tol_rel=1e-5, damping=0.5
)

# Benchmark helper

function run_bench(model, sampler, n_samples; reps, label, sampler_name)
    println("  [$sampler_name] $label samples:")
    b = @benchmark(
        sample(MersenneTwister(42), $model, $sampler, $n_samples; progress=false),
        samples=reps,
        evals=1,
    )
    show(stdout, MIME("text/plain"), b)
    println("\n")
    return b
end

configs = [(1_000, 8, "1k"), (5_000, 4, "5k"), (10_000, 3, "10k")]

results = Dict{Tuple{String,String},BenchmarkTools.Trial}()

# AdaptiveMALA baseline

println("=" ^ 60)
println("AdaptiveMALASampler  (n_warmup=500, Float64)")
println("Model: Bayesian linear regression  n=$n  p=$p")
println("=" ^ 60, "\n")
for (n_samples, reps, label) in configs
    results[("MALA", label)] = run_bench(
        model, mala_sampler, n_samples; reps, label, sampler_name="MALA"
    )
end

# ParallelMALA (DEER)

println("=" ^ 60)
println("ParallelMALASampler  (T=64, AutoEnzyme, Float64)")
println("Model: Bayesian linear regression  n=$n  p=$p")
println("=" ^ 60, "\n")
for (n_samples, reps, label) in configs
    results[("DEER", label)] = run_bench(
        model, deer_sampler, n_samples; reps, label, sampler_name="DEER"
    )
end

# Summary table

println("=" ^ 60)
println("Summary: median wall-clock time (ms)")
println("=" ^ 60)
@printf "%-10s  %12s  %12s  %12s\n" "N samples" "MALA" "DEER" "DEER/MALA"
for (_, _, label) in configs
    t_mala = median(results[("MALA", label)]).time / 1e6
    t_deer = median(results[("DEER", label)]).time / 1e6
    @printf "%-10s  %12.1f  %12.1f  %12.2fx\n" label t_mala t_deer (t_deer / t_mala)
end
println()
