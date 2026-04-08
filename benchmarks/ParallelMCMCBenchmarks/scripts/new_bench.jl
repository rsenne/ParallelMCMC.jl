"""
Raw GPU DEER benchmark for Bayesian logistic regression, with CPU sequential MALA baseline.

This bypasses AbstractMCMC.sample(...) and measures:
    1. GPU DEER block solve:
       x0 -> build tape -> build TapedRecursion -> DEER.solve(rec, x0)
    2. CPU sequential taped MALA for the same block length T
    3. GPU batched logdensity evaluation on the solved trajectory

Interpretation:
- T is the DEER/MALA block length, i.e. the number of taped MALA steps in one block.
- The CPU baseline performs those T steps sequentially.
- The GPU DEER solve computes a full length-T block using the DEER fixed-point solve.
"""

using Random
using BenchmarkTools
using Printf
using Statistics

using ParallelMCMC
using ParallelMCMCBenchmarks
using CUDA

const BayesLogReg = ParallelMCMCBenchmarks.BayesLogReg

rng = MersenneTwister(20251231)

# Start small, then sweep upward if needed.
D = 512
N_data = 131_072

X_f32, y_f32, _ = BayesLogReg.make_data(rng, N_data, D)
X_f32 = Float32.(X_f32)
y_f32 = Float32.(y_f32)

CUDA.functional() || error("CUDA is not functional in this environment")

# CPU and GPU copies of the same problem.
X_cpu = X_f32
y_cpu = y_f32
X_gpu = CUDA.CuMatrix(X_f32)
y_gpu = CUDA.CuVector(y_f32)

logp_cpu, gradlogp_cpu = BayesLogReg.make_problem(X_cpu, y_cpu)
logp_gpu, gradlogp_gpu, hvp_gpu = BayesLogReg.make_problem_with_hvp(X_gpu, y_gpu)
logp_gpu_batch, gradlogp_gpu_batch = BayesLogReg.make_problem_batched(X_gpu, y_gpu)

model_gpu = DensityModel(
    logp_gpu,
    gradlogp_gpu,
    D;
    hvp=hvp_gpu,
    logdensity_batch=logp_gpu_batch,
    grad_logdensity_batch=gradlogp_gpu_batch,
)

# Core DEER settings to test.
T_vals = [8, 16, 32, 64]

# Keep these fixed initially.
epsilon = 0.1f0
maxiter = 50
tol_abs = 1.0f-4
tol_rel = 1.0f-3
damping = 0.5f0
probes = 1

x0_gpu = CUDA.zeros(Float32, D)
x0_cpu = zeros(Float32, D)

"""
Build exactly the same DEER problem that ParallelMALASampler uses internally,
but expose it directly for raw benchmarking.
"""
function build_raw_deer_problem(
    rng::Random.AbstractRNG,
    model::DensityModel,
    x0::AbstractVector;
    epsilon::Float32,
    T::Int,
    maxiter::Int,
    tol_abs::Float32,
    tol_rel::Float32,
    damping::Float32,
    probes::Int,
    cholM=nothing,
    backend=DEER.DEFAULT_BACKEND,
)
    FP = typeof(epsilon)
    D = model.dim

    tape = map(1:T) do _
        ξ = copyto!(similar(x0, D), randn(rng, FP, D))
        ParallelMCMC.MALATapeElement(ξ, FP(rand(rng)))
    end

    rec = ParallelMCMC._build_mala_deer_rec(
        model,
        epsilon,
        tape,
        x0;
        cholM=cholM,
        backend=backend,
    )

    return rec
end

"""
Time one raw DEER block solve, returning the trajectory S (D×T).
"""
function solve_raw_deer_block(
    rng::Random.AbstractRNG,
    model::DensityModel,
    x0::AbstractVector;
    epsilon::Float32,
    T::Int,
    maxiter::Int,
    tol_abs::Float32,
    tol_rel::Float32,
    damping::Float32,
    probes::Int,
    cholM=nothing,
    backend=DEER.DEFAULT_BACKEND,
)
    rec = build_raw_deer_problem(
        rng, model, x0;
        epsilon=epsilon,
        T=T,
        maxiter=maxiter,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        damping=damping,
        probes=probes,
        cholM=cholM,
        backend=backend,
    )

    S = DEER.solve(
        rec,
        x0;
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        maxiter=maxiter,
        jacobian=:stoch_diag,
        damping=damping,
        probes=probes,
        rng=rng,
    )
    return S
end

"""
Run T sequential taped MALA steps on CPU and return the full state path (length T+1).
"""
function solve_seq_mala_block(
    rng::Random.AbstractRNG,
    logp,
    gradlogp,
    x0::AbstractVector;
    epsilon::Float32,
    T::Int,
    cholM=nothing,
)
    FP = typeof(epsilon)
    D = length(x0)
    ξs = [randn(rng, FP, D) for _ in 1:T]
    us = rand(rng, FP, T)
    return ParallelMCMC.MALA.run_mala_sequential_taped(
        logp,
        gradlogp,
        x0,
        epsilon,
        ξs,
        us;
        cholM=cholM,
    )
end

"""
Benchmark just the raw DEER solve.
"""
function bench_raw_deer_solve(model, x0; T, reps)
    println("  [GPU] raw DEER solve, T=$T")

    S_warm = solve_raw_deer_block(
        MersenneTwister(42),
        model,
        x0;
        epsilon=epsilon,
        T=T,
        maxiter=maxiter,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        damping=damping,
        probes=probes,
    )
    CUDA.synchronize()

    b = @benchmark begin
        S = solve_raw_deer_block(
            MersenneTwister(42),
            $model,
            $x0;
            epsilon=$epsilon,
            T=$T,
            maxiter=$maxiter,
            tol_abs=$tol_abs,
            tol_rel=$tol_rel,
            damping=$damping,
            probes=$probes,
        )
        CUDA.synchronize()
        S
    end samples=reps evals=1

    show(stdout, MIME("text/plain"), b)
    println("\n")

    t_ms = median(b).time / 1e6
    samples_per_sec = T / (median(b).time / 1e9)
    @printf "    median solve time: %.3f ms\n" t_ms
    @printf "    implied throughput: %.1f samples/sec\n\n" samples_per_sec

    return b
end

"""
Benchmark T sequential taped MALA steps on CPU.
"""
function bench_seq_mala_cpu(logp, gradlogp, x0; T, reps)
    println("  [CPU] sequential taped MALA, T=$T")

    xs_warm = solve_seq_mala_block(
        MersenneTwister(42),
        logp,
        gradlogp,
        x0;
        epsilon=epsilon,
        T=T,
    )

    b = @benchmark begin
        xs = solve_seq_mala_block(
            MersenneTwister(42),
            $logp,
            $gradlogp,
            $x0;
            epsilon=$epsilon,
            T=$T,
        )
        xs
    end samples=reps evals=1

    show(stdout, MIME("text/plain"), b)
    println("\n")

    t_ms = median(b).time / 1e6
    samples_per_sec = T / (median(b).time / 1e9)
    @printf "    median sequential time: %.3f ms\n" t_ms
    @printf "    implied throughput: %.1f steps/sec\n\n" samples_per_sec

    return b
end

"""
Benchmark batched logdensity over a solved block S.
"""
function bench_block_logdensity(model, x0; T, reps)
    model.logdensity_batch === nothing && error("model.logdensity_batch is required")

    S = solve_raw_deer_block(
        MersenneTwister(42),
        model,
        x0;
        epsilon=epsilon,
        T=T,
        maxiter=maxiter,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        damping=damping,
        probes=probes,
    )
    CUDA.synchronize()

    println("  [GPU] batched logdensity on solved block, T=$T")

    lp = model.logdensity_batch(S)
    CUDA.synchronize()

    b = @benchmark begin
        lp = $model.logdensity_batch($S)
        CUDA.synchronize()
        lp
    end samples=reps evals=1

    show(stdout, MIME("text/plain"), b)
    println("\n")

    t_ms = median(b).time / 1e6
    @printf "    median block logdensity time: %.3f ms\n\n" t_ms

    return b
end

println("=" ^ 72)
println("Raw GPU DEER benchmark + CPU sequential MALA baseline")
println("Model: Bayesian logistic regression   D=$D   N_data=$N_data   Float32")
println("T = block length = number of taped MALA steps in one block")
println("Measures raw GPU DEER block solve throughput, plus CPU sequential baseline")
println("=" ^ 72)
println()

solve_results = Dict{Int,BenchmarkTools.Trial}()
seq_results = Dict{Int,BenchmarkTools.Trial}()
logp_results = Dict{Int,BenchmarkTools.Trial}()

for T in T_vals
    reps = T <= 16 ? 8 : T <= 32 ? 6 : 4
    solve_results[T] = bench_raw_deer_solve(model_gpu, x0_gpu; T=T, reps=reps)
    seq_results[T] = bench_seq_mala_cpu(logp_cpu, gradlogp_cpu, x0_cpu; T=T, reps=reps)
    logp_results[T] = bench_block_logdensity(model_gpu, x0_gpu; T=T, reps=reps)
end

println("=" ^ 104)
println("Summary")
println("=" ^ 104)
@printf "%-8s  %12s  %14s  %12s  %14s  %10s  %14s\n" "T" "gpu solve ms" "gpu samp/sec" "cpu seq ms" "cpu step/sec" "speedup" "block logp ms"

for T in T_vals
    t_gpu_ms = median(solve_results[T]).time / 1e6
    t_gpu_s = median(solve_results[T]).time / 1e9
    gpu_rate = T / t_gpu_s

    t_cpu_ms = median(seq_results[T]).time / 1e6
    t_cpu_s = median(seq_results[T]).time / 1e9
    cpu_rate = T / t_cpu_s

    speedup = t_cpu_ms / t_gpu_ms
    t_logp_ms = median(logp_results[T]).time / 1e6

    @printf "%-8d  %12.3f  %14.1f  %12.3f  %14.1f  %10.2f  %14.3f\n" T t_gpu_ms gpu_rate t_cpu_ms cpu_rate speedup t_logp_ms
end

println()
