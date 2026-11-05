"""
Bayesian logistic regression on the German Credit data, after Section 5.2 of
Zoltowski et al. (2025).

We follow the setup of their Appendix B.1. The covariates are whitened, an intercept
is added, and the prior is N(0, I), so the posterior has 25 dimensions. The chain starts
from a prior draw moved by three MALA steps, and ε = 0.0015 gives about 80% acceptance.
The parallel sampler solves the whole chain as one block (`T = L`) to an absolute
tolerance of 5e-4 in at most `50 + 5L/10⁴` iterations.

Whitening matters. Quasi-DEER uses only the diagonal of each step's Jacobian, and with
standardized but correlated covariates that diagonal misses most of the coupling: DEER
then needs about `T/2` iterations instead of a few dozen.

We time one chain of `L` samples with

- sequential `MALASampler` on the CPU,
- sequential `MALASampler` on the GPU, which is the baseline used in the paper,
- `ParallelMALASampler` on the GPU.

Every method reports the best of `PMCMC_REPS` timed runs after a warm-up run, the
acceptance rate, and the largest gap between its posterior means and those of a long
sequential CPU chain, in posterior standard deviations. Both samplers draw the same
random numbers from the same seed, so DEER also reports its largest distance from the
sequential GPU chain, which should be within the solver tolerance.

Run from the repository root:

    julia --project=benchmarks/ParallelMCMCBenchmarks \\
        benchmarks/ParallelMCMCBenchmarks/scripts/german_credit.jl

The data are downloaded from the UCI repository on the first run and cached in
`tempdir()`. Set `PMCMC_L_VALS` to a comma-separated list to change the sweep, and
`PMCMC_OUT` to a file path to save the timings as CSV for
`docs/figures/german_credit.jl`.
"""

using Downloads
using LinearAlgebra
using Printf
using Random
using Statistics

using CUDA
using ParallelMCMC

const DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
const DATA_PATH = joinpath(tempdir(), "german.data-numeric")

function _parse_ints(name, default)
    let raw = strip(get(ENV, name, ""))
        isempty(raw) ? default : parse.(Int, strip.(split(raw, ",")))
    end
end

const L_VALS = _parse_ints("PMCMC_L_VALS", [1_000, 4_000, 16_000, 64_000])
const REPS = parse(Int, get(ENV, "PMCMC_REPS", "3"))

# Data

function load_german_credit()
    isfile(DATA_PATH) || Downloads.download(DATA_URL, DATA_PATH)
    rows = [
        parse.(Float64, split(line)) for
        line in eachline(DATA_PATH) if !isempty(strip(line))
    ]
    A = reduce(vcat, permutedims.(rows))
    features = A[:, 1:(end - 1)]
    y = A[:, end] .== 1                # 1 = good credit, 2 = bad credit
    features = features .- mean(features; dims=1)
    features = features * inv(sqrt(Symmetric(cov(features))))
    X = hcat(ones(size(features, 1)), features)
    return Float32.(X), Float32.(y)
end

# Log density, gradient and Hessian-vector product, for one point or for the columns
# of a matrix. `softplus(z) = log(1 + exp(z))`.

softplus(z) = max(z, zero(z)) + log1p(exp(-abs(z)))
sigmoid(z) = inv(one(z) + exp(-z))

function logistic_model(X, y)
    D = size(X, 2)
    half = eltype(X)(0.5)

    logp(β) = (Xβ=X * β; sum(y .* Xβ .- softplus.(Xβ)) - half * sum(abs2, β))
    grad(β) = X' * (y .- sigmoid.(X * β)) .- β
    function hvp(β, v)
        p = sigmoid.(X * β)
        return -(X' * (p .* (1 .- p) .* (X * v))) .- v
    end

    function logp_batch(B)
        XB = X * B
        return vec(sum(y .* XB .- softplus.(XB); dims=1)) .-
               half .* vec(sum(abs2, B; dims=1))
    end
    grad_batch(B) = X' * (y .- sigmoid.(X * B)) .- B
    function hvp_batch(B, V)
        P = sigmoid.(X * B)
        return -(X' * (P .* (1 .- P) .* (X * V))) .- V
    end

    sequential = DensityModel(logp, grad, D; hvp=hvp)
    batched = DensityModel(
        logp,
        grad,
        D;
        hvp=hvp,
        logdensity_batch=logp_batch,
        grad_logdensity_batch=grad_batch,
        hvp_batch=hvp_batch,
    )
    return sequential, batched
end

# Timing

function run_chain(model, sampler, L, x0; seed=42)
    chain = sample(
        MersenneTwister(seed),
        model,
        sampler,
        L;
        chain_type=Any,
        progress=false,
        initial_params=x0,
    )
    x0 isa CuArray && CUDA.synchronize()
    return chain
end

function best_time(model, sampler, L, x0)
    run_chain(model, sampler, L, x0; seed=0)
    times = Float64[]
    chain = nothing
    for _ in 1:REPS
        GC.gc()
        t0 = time_ns()
        chain = run_chain(model, sampler, L, x0)
        push!(times, (time_ns() - t0) / 1e9)
    end
    return minimum(times), chain
end

function draws(chain)
    return reduce(hcat, (Array(chain[i].x) for i in 1:length(chain)))
end

#= DEER solves to a tolerance, so a rejected step repeats the previous state only up to
about `tol_abs`. Accepted moves are of order √(2ε) ≈ 0.05. =#
function acceptance(S; atol=1e-4)
    return mean(
        maximum(abs, view(S, :, t) .- view(S, :, t - 1)) > atol for t in 2:size(S, 2)
    )
end

function mean_gap(S, ref_mean, ref_std)
    return maximum(abs.(vec(mean(S; dims=2)) .- ref_mean) ./ ref_std)
end

# Run

CUDA.functional() || error("this benchmark needs a functional CUDA GPU")

X_cpu, y_cpu = load_german_credit()
N_data, D = size(X_cpu)
X_gpu, y_gpu = CuArray(X_cpu), CuArray(y_cpu)

cpu_model, _ = logistic_model(X_cpu, y_cpu)
gpu_seq_model, gpu_batch_model = logistic_model(X_gpu, y_gpu)

epsilon = parse(Float32, get(ENV, "PMCMC_EPSILON", "0.0015"))
mala = MALASampler(epsilon)
function deer(L)
    return ParallelMALASampler(
        epsilon; T=L, maxiter=50 + ceil(Int, 5L / 10^4), tol_abs=5.0f-4, tol_rel=0
    )
end

x0_cpu = run_chain(cpu_model, mala, 3, randn(MersenneTwister(7), Float32, D))[end].x
x0_gpu = CuArray(x0_cpu)

println("German Credit logistic regression: N = $N_data, D = $D, ε = $epsilon")
println("GPU: ", CUDA.name(CUDA.device()), "; CPU BLAS threads: ", BLAS.get_num_threads())
println()

# Reference moments from a long sequential chain, burn-in discarded.
ref_chain = run_chain(cpu_model, mala, 200_000, x0_cpu; seed=1)
ref = draws(ref_chain)[:, 20_001:end]
ref_mean = vec(mean(ref; dims=2))
ref_std = vec(std(ref; dims=2))

results = []
for L in L_VALS
    configs = (
        ("MALA, CPU", cpu_model, mala, x0_cpu),
        ("MALA, GPU", gpu_seq_model, mala, x0_gpu),
        ("DEER MALA, GPU", gpu_batch_model, deer(L), x0_gpu),
    )
    S_seq = nothing
    for (name, model, sampler, x0) in configs
        t, chain = best_time(model, sampler, L, x0)
        S = draws(chain)
        name == "MALA, GPU" && (S_seq = S)
        burn = L ÷ 10
        gap = mean_gap(S[:, (burn + 1):end], ref_mean, ref_std)
        push!(results, (; L, name, t, acc=acceptance(S), gap))
        @printf(
            "L = %6d  %-26s %9.3f s  acc %.2f  mean gap %.2f sd\n",
            L,
            name,
            t,
            acceptance(S),
            gap
        )
    end
    #= `MALASampler` returns the initial point as its first draw and
    `ParallelMALASampler` starts at the first transition, hence the shift. =#
    S_deer = draws(run_chain(gpu_batch_model, deer(L), L, x0_gpu))
    @printf(
        "L = %6d  max |DEER - sequential| = %.1e\n",
        L,
        maximum(abs.(S_deer[:, 1:(end - 1)] .- S_seq[:, 2:end]))
    )
    println()
end

out = get(ENV, "PMCMC_OUT", "")
if !isempty(out)
    open(out, "w") do io
        println(io, "L,method,seconds")
        for r in results
            @printf(io, "%d,\"%s\",%.4f\n", r.L, r.name, r.t)
        end
    end
    println("wrote ", out, "\n")
end

println("Speedup of DEER MALA on the GPU:")
for L in L_VALS
    rs = filter(r -> r.L == L, results)
    seq = only(filter(r -> r.name == "MALA, GPU", rs)).t
    cpu = only(filter(r -> r.name == "MALA, CPU", rs)).t
    par = only(filter(r -> r.name == "DEER MALA, GPU", rs)).t
    @printf("L = %6d  %6.1f× vs GPU MALA, %6.1f× vs CPU MALA\n", L, seq / par, cpu / par)
end
