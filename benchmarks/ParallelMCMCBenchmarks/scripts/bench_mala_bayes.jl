using Random
using BenchmarkTools
using LinearAlgebra

using ParallelMCMC
using ParallelMCMCBenchmarks
const BayesLinReg = ParallelMCMCBenchmarks.BayesLinReg
const MALARunner = ParallelMCMCBenchmarks.MALARunner

rng = MersenneTwister(20251231)

n, p = 200, 16
X = randn(rng, n, p)
β_true = randn(rng, p)
σ = 1.0
y = X*β_true .+ σ .* randn(rng, n)

logpost, gradlogpost, _, _ = BayesLinReg.make_problem(X, y; σ=σ, τ=10.0)

T = 10_000
ϵ = 0.25
x0 = zeros(Float64, p)

ξs, us = MALARunner.make_tape(rng, p, T)

@btime MALARunner.run_taped_mala_with_accepts($logpost, $gradlogpost, $x0, $ϵ, $ξs, $us);
@profview_allocs MALARunner.run_taped_mala_with_accepts(logpost, gradlogpost, x0, ϵ, ξs, us);
