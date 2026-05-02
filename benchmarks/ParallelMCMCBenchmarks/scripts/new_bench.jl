"""
Raw GPU DEER benchmark for Bayesian logistic regression, with CPU sequential MALA baseline.

This version isolates the GPU DEER solve by prebuilding:
    1. the taped recursion `rec`
    2. the DEER workspace `ws`

So the timed region measures only:
    DEER.solve(rec, x0; workspace=ws, ...)

An additional builder benchmark is included so build-time allocations are separated
from solve-time allocations.
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

D = 512
N_data = 131_072

X_f32, y_f32, _ = BayesLogReg.make_data(rng, N_data, D)
X_f32 = Float32.(X_f32)
y_f32 = Float32.(y_f32)

CUDA.functional() || error("CUDA is not functional in this environment")

X_cpu = X_f32
y_cpu = y_f32
X_gpu = CUDA.CuMatrix(X_f32)
y_gpu = CUDA.CuVector(y_f32)

logp_cpu, gradlogp_cpu = BayesLogReg.make_problem(X_cpu, y_cpu)
logp_gpu, gradlogp_gpu, hvp_gpu = BayesLogReg.make_problem_with_hvp(X_gpu, y_gpu)
logp_gpu_batch, gradlogp_gpu_batch, hvp_gpu_batch = BayesLogReg.make_problem_batched_with_hvp(
    X_gpu, y_gpu
)

model_gpu = DensityModel(
    logp_gpu,
    gradlogp_gpu,
    D;
    hvp=hvp_gpu,
    logdensity_batch=logp_gpu_batch,
    grad_logdensity_batch=gradlogp_gpu_batch,
    hvp_batch=hvp_gpu_batch,
)

T_vals = [8, 16, 32, 64]

epsilon = 0.1f0
maxiter = 50
tol_abs = 1.0f-4
tol_rel = 1.0f-3
damping = 0.5f0
probes = 1

x0_gpu = CUDA.zeros(Float32, D)
x0_cpu = zeros(Float32, D)

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
        model, epsilon, tape, x0; cholM=cholM, backend=backend
    )

    return rec
end

function build_workspace(x0::AbstractVector, T::Int)
    return DEER.DEERWorkspace(x0, T)
end

function solve_raw_deer_block_prebuilt(
    rng::Random.AbstractRNG,
    rec,
    x0::AbstractVector,
    ws;
    tol_abs::Float32,
    tol_rel::Float32,
    maxiter::Int,
    damping::Float32,
    probes::Int,
)
    return DEER.solve(
        rec,
        x0;
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        maxiter=maxiter,
        jacobian=:stoch_diag,
        damping=damping,
        probes=probes,
        rng=rng,
        workspace=ws,
        copy_result=false,
    )
end

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
        logp, gradlogp, x0, epsilon, ξs, us; cholM=cholM
    )
end

function bench_raw_deer_build(model, x0; T, reps)
    println("  [GPU] build taped recursion only, T=$T")

    rec_warm = build_raw_deer_problem(
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
        rec = build_raw_deer_problem(
            MersenneTwister(42),
            $model,
            $x0;
            epsilon=($epsilon),
            T=($T),
            maxiter=($maxiter),
            tol_abs=($tol_abs),
            tol_rel=($tol_rel),
            damping=($damping),
            probes=($probes),
        )
        CUDA.synchronize()
        rec
    end samples=reps evals=1

    show(stdout, MIME("text/plain"), b)
    println()

    t_ms = median(b).time / 1e6
    @printf "    median build time: %.3f ms\n\n" t_ms
    return b
end

function bench_raw_deer_solve_only(model, x0; T, reps)
    println("  [GPU] raw DEER solve only (prebuilt rec + workspace), T=$T")

    rec = build_raw_deer_problem(
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
    ws = build_workspace(x0, T)

    S_warm = solve_raw_deer_block_prebuilt(
        MersenneTwister(42),
        rec,
        x0,
        ws;
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        maxiter=maxiter,
        damping=damping,
        probes=probes,
    )
    CUDA.synchronize()

    b = @benchmark begin
        S = solve_raw_deer_block_prebuilt(
            MersenneTwister(42),
            $rec,
            $x0,
            $ws;
            tol_abs=($tol_abs),
            tol_rel=($tol_rel),
            maxiter=($maxiter),
            damping=($damping),
            probes=($probes),
        )
        CUDA.synchronize()
        S
    end samples=reps evals=1

    show(stdout, MIME("text/plain"), b)
    println()

    t_ms = median(b).time / 1e6
    samples_per_sec = T / (median(b).time / 1e9)
    @printf "    median solve time: %.3f ms\n" t_ms
    @printf "    implied throughput: %.1f samples/sec\n\n" samples_per_sec

    return b, rec, ws
end

function bench_seq_mala_cpu(logp, gradlogp, x0; T, reps)
    println("  [CPU] sequential taped MALA, T=$T")

    xs_warm = solve_seq_mala_block(
        MersenneTwister(42), logp, gradlogp, x0; epsilon=epsilon, T=T
    )

    b = @benchmark begin
        xs = solve_seq_mala_block(
            MersenneTwister(42), $logp, $gradlogp, $x0; epsilon=($epsilon), T=($T)
        )
        xs
    end samples=reps evals=1

    show(stdout, MIME("text/plain"), b)
    println()

    t_ms = median(b).time / 1e6
    samples_per_sec = T / (median(b).time / 1e9)
    @printf "    median sequential time: %.3f ms\n" t_ms
    @printf "    implied throughput: %.1f steps/sec\n\n" samples_per_sec

    return b
end

function bench_block_logdensity(model, rec, x0, ws; T, reps)
    model.logdensity_batch === nothing && error("model.logdensity_batch is required")

    S = solve_raw_deer_block_prebuilt(
        MersenneTwister(42),
        rec,
        x0,
        ws;
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        maxiter=maxiter,
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
    println()

    t_ms = median(b).time / 1e6
    @printf "    median block logdensity time: %.3f ms\n\n" t_ms

    return b
end

println("=" ^ 72)
println("Raw GPU DEER benchmark + CPU sequential MALA baseline")
println("Model: Bayesian logistic regression   D=$D   N_data=$N_data   Float32")
println("T = block length = number of taped MALA steps in one block")
println("Measures build separately from solve; GPU solve uses prebuilt rec + workspace")
println("=" ^ 72)
println()

build_results = Dict{Int,BenchmarkTools.Trial}()
solve_results = Dict{Int,BenchmarkTools.Trial}()
seq_results = Dict{Int,BenchmarkTools.Trial}()
logp_results = Dict{Int,BenchmarkTools.Trial}()

for T in T_vals
    reps = if T <= 16
        8
    elseif T <= 32
        6
    else
        4
    end
    build_results[T] = bench_raw_deer_build(model_gpu, x0_gpu; T=T, reps=reps)
    solve_results[T], rec, ws = bench_raw_deer_solve_only(model_gpu, x0_gpu; T=T, reps=reps)
    seq_results[T] = bench_seq_mala_cpu(logp_cpu, gradlogp_cpu, x0_cpu; T=T, reps=reps)
    logp_results[T] = bench_block_logdensity(model_gpu, rec, x0_gpu, ws; T=T, reps=reps)
end

println("=" ^ 122)
println("Summary")
println("=" ^ 122)
@printf "%-8s  %12s  %14s  %12s  %14s  %12s  %14s  %10s\n" "T" "build ms" "gpu solve ms" "gpu samp/sec" "cpu seq ms" "cpu step/sec" "block logp ms" "speedup"

for T in T_vals
    t_build_ms = median(build_results[T]).time / 1e6

    t_gpu_ms = median(solve_results[T]).time / 1e6
    t_gpu_s = median(solve_results[T]).time / 1e9
    gpu_rate = T / t_gpu_s

    t_cpu_ms = median(seq_results[T]).time / 1e6
    t_cpu_s = median(seq_results[T]).time / 1e9
    cpu_rate = T / t_cpu_s

    t_logp_ms = median(logp_results[T]).time / 1e6
    speedup = t_cpu_ms / t_gpu_ms

    @printf "%-8d  %12.3f  %14.3f  %14.1f  %12.3f  %12.1f  %14.3f  %10.2f\n" T t_build_ms t_gpu_ms gpu_rate t_cpu_ms cpu_rate t_logp_ms speedup
end

println()
