"""
German Credit with every GPU method compiled by Reactant, for a comparison closer to the
JAX setup of Zoltowski et al. (2025).

`german_credit.jl` times sequential MALA on the GPU as `MALASampler` runs it: each step
waits for the device to make its accept decision on the host. JAX compiles the whole
sequential loop into one device program instead. Here we do the same with Reactant and
time it against `ParallelMALASampler` with `AutoReactant()` derivatives, which compiles
each DEER iteration (#90). Sequential MALA on the CPU is timed too, on the same machine.

The model, step size, starting point, tolerance and iteration limit are those of
`german_credit.jl`. The sequential loop takes its noise from the same random number
stream as `MALASampler`, so the two chains should agree up to Float32 rounding.

Reactant only has a GPU client on Linux, so this runs under Linux or WSL, in an
environment with ParallelMCMC and Reactant:

    julia --project=<env> benchmarks/ParallelMCMCBenchmarks/scripts/german_credit_reactant.jl

`PMCMC_L_VALS`, `PMCMC_REPS` and `PMCMC_EPSILON` work as in `german_credit.jl`.
"""

using Downloads
using LinearAlgebra
using Printf
using Random
using Statistics

using ADTypes
using ParallelMCMC
using Reactant

const DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
const DATA_PATH = joinpath(tempdir(), "german.data-numeric")

function _parse_ints(name, default)
    let raw = strip(get(ENV, name, ""))
        isempty(raw) ? default : parse.(Int, strip.(split(raw, ",")))
    end
end

const L_VALS = _parse_ints("PMCMC_L_VALS", [1_000, 4_000, 16_000, 64_000])
const REPS = parse(Int, get(ENV, "PMCMC_REPS", "10"))

function load_german_credit()
    isfile(DATA_PATH) || Downloads.download(DATA_URL, DATA_PATH)
    rows = [
        parse.(Float64, split(line)) for
        line in eachline(DATA_PATH) if !isempty(strip(line))
    ]
    A = reduce(vcat, permutedims.(rows))
    features = A[:, 1:(end - 1)]
    y = A[:, end] .== 1
    features = features .- mean(features; dims=1)
    features = features * inv(sqrt(Symmetric(cov(features))))
    X = hcat(ones(size(features, 1)), features)
    return Float32.(X), Float32.(y)
end

softplus(z) = max(z, zero(z)) + log1p(exp(-abs(z)))
sigmoid(z) = inv(one(z) + exp(-z))

logp(X, y, β) = (Xβ=X * β; sum(y .* Xβ .- softplus.(Xβ)) - 0.5f0 * sum(abs2, β))
grad(X, y, β) = X' * (y .- sigmoid.(X * β)) .- β

#= The whole chain as one program. Column 1 is the starting point, as in `MALASampler`.
`logU` stays Float64 because `MALASampler` compares `log(rand(rng))` in Float64. =#
function mala_chain(X, y, x0, Ξ, logU, ε)
    S = similar(Ξ)
    S[:, 1] = x0
    # Without the copy, the traced loop writes its final state back into `x0`, so the
    # next call starts where this one ended.
    x = copy(x0)
    lp = logp(X, y, x)
    g = grad(X, y, x)
    @trace for t in 2:size(Ξ, 2)
        x_new = x .+ ε .* g .+ sqrt(2ε) .* Ξ[:, t]
        lp_new = logp(X, y, x_new)
        g_new = grad(X, y, x_new)
        logq_fwd = -sum(abs2, x_new .- x .- ε .* g) / (4ε)
        logq_rev = -sum(abs2, x .- x_new .- ε .* g_new) / (4ε)
        logα = lp_new - lp + logq_rev - logq_fwd
        # With a traced `t` this is a device-side slice, not a host read.
        accept = logα > @allowscalar(logU[t])
        x = ifelse.(accept, x_new, x)
        lp = ifelse(accept, lp_new, lp)
        g = ifelse.(accept, g_new, g)
        S[:, t] = x
    end
    return S
end

# The noise `MALASampler` draws from `MersenneTwister(seed)` for an `L`-sample chain.
function mala_noise(D, L; seed=42)
    rng = MersenneTwister(seed)
    Ξ = zeros(Float32, D, L)
    logU = zeros(Float64, L)
    ξ = Vector{Float32}(undef, D)
    for t in 2:L
        randn!(rng, ξ)
        Ξ[:, t] = ξ
        logU[t] = log(rand(rng))
    end
    return Ξ, logU
end

function best_of(f)
    f()
    times = Float64[]
    out = nothing
    for _ in 1:REPS
        GC.gc()
        t0 = time_ns()
        out = f()
        push!(times, (time_ns() - t0) / 1e9)
    end
    return minimum(times), out
end

function run_chain(model, sampler, L, x0; seed=42)
    return sample(
        MersenneTwister(seed),
        model,
        sampler,
        L;
        chain_type=Any,
        progress=false,
        initial_params=x0,
    )
end
draws(chain) = reduce(hcat, (Array(chain[i].x) for i in 1:length(chain)))
mean_gap(S, m, s) = maximum(abs.(vec(mean(S; dims=2)) .- m) ./ s)

Reactant.set_default_backend("gpu")

X, y = load_german_credit()
N_data, D = size(X)
epsilon = parse(Float32, get(ENV, "PMCMC_EPSILON", "0.0015"))

cpu_model = DensityModel(β -> logp(X, y, β), β -> grad(X, y, β), D)
logp_batch(B) = (Z=X * B; vec(sum(y .* Z .- softplus.(Z); dims=1)) .- 0.5f0 .* vec(sum(abs2, B; dims=1)))
xla_model = DensityModel(
    β -> logp(X, y, β), AutoReactant(), D; logdensity_batch=logp_batch
)

mala = MALASampler(epsilon)
function deer(L)
    return ParallelMALASampler(
        epsilon;
        T=L,
        maxiter=50 + ceil(Int, 5L / 10^4),
        tol_abs=5.0f-4,
        tol_rel=0,
        backend=AutoReactant(),
    )
end

x0 = draws(run_chain(cpu_model, mala, 3, randn(MersenneTwister(7), Float32, D)))[:, end]

println("German Credit logistic regression: N = $N_data, D = $D, ε = $epsilon")
println("XLA client: ", Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
println("CPU BLAS threads: ", BLAS.get_num_threads(), "; best of $REPS runs")
println()

ref = draws(run_chain(cpu_model, mala, 200_000, x0; seed=1))[:, 20_001:end]
ref_mean, ref_std = vec(mean(ref; dims=2)), vec(std(ref; dims=2))

X_r, y_r, x0_r = Reactant.to_rarray(X), Reactant.to_rarray(y), Reactant.to_rarray(x0)
results = []
for L in L_VALS
    Ξ, logU = mala_noise(D, L)
    Ξ_r, logU_r = Reactant.to_rarray(Ξ), Reactant.to_rarray(logU)
    chain_xla = Reactant.with_config(;
        dot_general_precision=Reactant.PrecisionConfig.HIGHEST
    ) do
        @compile mala_chain(X_r, y_r, x0_r, Ξ_r, logU_r, epsilon)
    end

    #= `sample` compiles the DEER iteration once per call, which would put a compile in
    every timed run. Compile once here and time only the block solve, as for the
    compiled sequential loop. =#
    s = deer(L)
    prepped = ParallelMCMC._prepare_model(xla_model, x0, L, s.backend)
    t_compile = @elapsed ws = ParallelMCMC._mala_deer_workspace(prepped, s, x0, nothing)
    solve() = first(
        ParallelMCMC._deer_solve_new_tape(MersenneTwister(42), prepped, s, x0; workspace=ws)
    )

    t_cpu, c_cpu = best_of(() -> run_chain(cpu_model, mala, L, x0))
    t_seq, S_seq = best_of(() -> Array(chain_xla(X_r, y_r, x0_r, Ξ_r, logU_r, epsilon)))
    t_deer, S_deer = best_of(solve)
    S_cpu = draws(c_cpu)
    burn = (L ÷ 10 + 1):L
    for (name, t, S) in (
        ("MALA, CPU", t_cpu, S_cpu),
        ("MALA, GPU, compiled loop", t_seq, S_seq),
        ("DEER MALA, GPU, AutoReactant", t_deer, S_deer),
    )
        push!(results, (; L, name, t))
        gap = mean_gap(S[:, burn], ref_mean, ref_std)
        @printf("L = %6d  %-30s %9.3f s  mean gap %.2f sd\n", L, name, t, gap)
    end
    gaps = vec(maximum(abs.(S_seq .- S_cpu); dims=1))
    @printf(
        "L = %6d  max |compiled loop - MALASampler| = %.1e, first step off by > 1e-3: %s\n",
        L,
        maximum(gaps),
        something(findfirst(>(1.0f-3), gaps), "none")
    )
    @printf("L = %6d  DEER compile time %.1f s, not included above\n", L, t_compile)
    println()
end

println("Speedup of DEER MALA (AutoReactant) on the GPU:")
for L in L_VALS
    rs = filter(r -> r.L == L, results)
    t(name) = only(filter(r -> r.name == name, rs)).t
    par = t("DEER MALA, GPU, AutoReactant")
    @printf(
        "L = %6d  %6.1f× vs compiled GPU MALA, %6.1f× vs CPU MALA\n",
        L,
        t("MALA, GPU, compiled loop") / par,
        t("MALA, CPU") / par
    )
end
