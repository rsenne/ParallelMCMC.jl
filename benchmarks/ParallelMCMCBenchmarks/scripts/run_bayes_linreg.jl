using Random
using LinearAlgebra
using Statistics

using ParallelMCMC
using ParallelMCMCBenchmarks
const BayesLinReg = ParallelMCMCBenchmarks.BayesLinReg
const MALARunner = ParallelMCMCBenchmarks.MALARunner

rng = MersenneTwister(20251231)

# Synthetic regression problem
n, p = 200, 8
X = randn(rng, n, p)

β_true = randn(rng, p)
σ = 1.0
y = X*β_true .+ σ .* randn(rng, n)

logpost, gradlogpost, μ_post, Σ_post = BayesLinReg.make_problem(X, y; σ=σ, τ=10.0)

# MALA settings
T = 50_000
burn = 10_000

# Start small; tune in warmup
ϵ0 = 1e-3

x0 = zeros(Float64, p)

# Warmup tune
x_warm, ϵ = MALARunner.tune_stepsize_mala(
    logpost, gradlogpost, x0, ϵ0; Twarm=5_000, rng=rng
)
println("Tuned ϵ: ", ϵ)

# Main sampling tape
ξs, us = MALARunner.make_tape(rng, p, T)
xs, accepts = MALARunner.run_taped_mala_with_accepts(
    logpost, gradlogpost, x_warm, ϵ, ξs, us
)
println("Acceptance rate: ", MALARunner.accept_rate(accepts))

# Posterior summaries from samples
Xmat = reduce(hcat, xs[(burn + 1):end])   # p × N
μ̂ = vec(mean(Xmat; dims=2))

# Diagonal posterior std estimate
Xc = Xmat .- μ̂
Σ̂ = (Xc * Xc') ./ (size(Xmat, 2) - 1)

println("\nAnalytic posterior mean (first 5): ", μ_post[1:min(p, 5)])
println("Sample posterior mean  (first 5): ", μ̂[1:min(p, 5)])
println("Mean abs error: ", mean(abs.(μ̂ .- μ_post)))

println("\nAnalytic posterior std (first 5): ", sqrt.(diag(Σ_post))[1:min(p, 5)])
println("Sample posterior std  (first 5): ", sqrt.(diag(Σ̂))[1:min(p, 5)])
println("Std abs error: ", mean(abs.(sqrt.(diag(Σ̂)) .- sqrt.(diag(Σ_post)))))

println("\nDone.")
