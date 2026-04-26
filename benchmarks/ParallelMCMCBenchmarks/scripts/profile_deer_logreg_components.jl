"""
Component profile for end-to-end GPU Parallel MALA on Bayesian logistic regression.
"""

using Random
using BenchmarkTools
using Printf
using Statistics

using AbstractMCMC: sample
using ParallelMCMC
using ParallelMCMCBenchmarks
using CUDA

const BayesLogReg = ParallelMCMCBenchmarks.BayesLogReg

const D = parse(Int, get(ENV, "PMCMC_D", "10"))
const N_data = parse(Int, get(ENV, "PMCMC_N_DATA", "200"))
const T = parse(Int, get(ENV, "PMCMC_T", "1024"))
const N_samples = parse(Int, get(ENV, "PMCMC_N_SAMPLES", "10000"))
const reps = parse(Int, get(ENV, "PMCMC_REPS", "6"))

const epsilon = parse(Float32, get(ENV, "PMCMC_EPSILON", "0.1"))
const maxiter = parse(Int, get(ENV, "PMCMC_MAXITER", "50"))
const tol_abs = parse(Float32, get(ENV, "PMCMC_TOL_ABS", "1.0e-4"))
const tol_rel = parse(Float32, get(ENV, "PMCMC_TOL_REL", "1.0e-3"))
const damping = parse(Float32, get(ENV, "PMCMC_DAMPING", "0.5"))
const probes = parse(Int, get(ENV, "PMCMC_PROBES", "1"))

CUDA.functional() || error("CUDA is not functional in this environment")

function make_tape(rng::Random.AbstractRNG, x0::AbstractVector, dim::Int, t_len::Int)
    FP = typeof(epsilon)
    return map(1:t_len) do _
        xi = copyto!(similar(x0, dim), randn(rng, FP, dim))
        ParallelMCMC.MALATapeElement(xi, FP(rand(rng)))
    end
end

function build_problem()
    rng = MersenneTwister(20251231)
    X_f32, y_f32, _ = BayesLogReg.make_data(rng, N_data, D)
    X_f32 = Float32.(X_f32)
    y_f32 = Float32.(y_f32)

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

    x0_gpu = CUDA.zeros(Float32, D)
    ws = DEER.DEERWorkspace(x0_gpu, T)
    return (; model_gpu, x0_gpu, ws)
end

function build_rec(model_gpu, x0_gpu, tape)
    return ParallelMCMC._build_mala_deer_rec(
        model_gpu,
        epsilon,
        tape,
        x0_gpu;
    )
end

function solve_prebuilt(rec, x0_gpu, ws; seed=42, return_info=false)
    out = DEER.solve(
        rec,
        x0_gpu;
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        maxiter=maxiter,
        jacobian=:stoch_diag,
        damping=damping,
        probes=probes,
        rng=MersenneTwister(seed),
        workspace=ws,
        return_info=return_info,
    )
    CUDA.synchronize()
    return out
end

function cache_logps(model_gpu, S)
    logps = Array(model_gpu.logdensity_batch(S))
    CUDA.synchronize()
    return logps
end

function materialize_transitions(S, logps)
    n = size(S, 2)
    out = Vector{ParallelMALATransition}(undef, n)
    for t in 1:n
        out[t] = ParallelMALATransition(S[:, t], logps[t])
    end
    CUDA.synchronize()
    return out
end

function print_trial(name, trial)
    t_ms = median(trial).time / 1e6
    @printf "%-32s  %12.3f  %12d  %12.3f\n" name t_ms median(trial).allocs median(trial).memory / 1024^2
    return t_ms
end

state = build_problem()
model_gpu = state.model_gpu
x0_gpu = state.x0_gpu
ws = state.ws

tape = make_tape(MersenneTwister(42), x0_gpu, D, T)
rec = build_rec(model_gpu, x0_gpu, tape)
S_info, info = solve_prebuilt(rec, x0_gpu, ws; return_info=true)
logps = cache_logps(model_gpu, S_info)
materialize_transitions(S_info, logps)

deer_gpu = ParallelMALASampler(
    epsilon;
    T=T,
    maxiter=maxiter,
    tol_abs=tol_abs,
    tol_rel=tol_rel,
    damping=damping,
    probes=probes,
)

println("=" ^ 96)
println("GPU ParallelMALA component profile")
println("Model: Bayesian logistic regression  D=$D  N_data=$N_data")
println("T=$T  N_samples=$N_samples  blocks=$(cld(N_samples, T))")
println("epsilon=$epsilon  maxiter=$maxiter  tol_abs=$tol_abs  tol_rel=$tol_rel")
println("Warmup solve converged=$(info.converged), iters=$(info.iters), metric=$(info.metric)")
println("=" ^ 96)
println()

println("Detailed BenchmarkTools output")
println("-" ^ 96)

b_tape = @benchmark begin
    tape_local = make_tape(MersenneTwister(42), $x0_gpu, $D, $T)
    CUDA.synchronize()
    tape_local
end samples=reps evals=1
show(stdout, MIME("text/plain"), b_tape)
println("\n")

b_build = @benchmark begin
    rec_local = build_rec($model_gpu, $x0_gpu, $tape)
    CUDA.synchronize()
    rec_local
end samples=reps evals=1
show(stdout, MIME("text/plain"), b_build)
println("\n")

b_solve = @benchmark begin
    S = solve_prebuilt($rec, $x0_gpu, $ws)
    S
end samples=reps evals=1
show(stdout, MIME("text/plain"), b_solve)
println("\n")

b_logps = @benchmark begin
    lp = cache_logps($model_gpu, $S_info)
    lp
end samples=reps evals=1
show(stdout, MIME("text/plain"), b_logps)
println("\n")

b_materialize = @benchmark begin
    ts = materialize_transitions($S_info, $logps)
    ts
end samples=reps evals=1
show(stdout, MIME("text/plain"), b_materialize)
println("\n")

b_sample = @benchmark begin
    xs = sample(
        MersenneTwister(42),
        $model_gpu,
        $deer_gpu,
        $N_samples;
        progress=false,
        initial_params=$x0_gpu,
    )
    CUDA.synchronize()
    xs
end samples=reps evals=1
show(stdout, MIME("text/plain"), b_sample)
println("\n")

println("=" ^ 96)
println("Summary")
println("=" ^ 96)
@printf "%-32s  %12s  %12s  %12s\n" "section" "median ms" "allocs" "memory MiB"
t_tape = print_trial("make_tape", b_tape)
t_build = print_trial("build_rec", b_build)
t_solve = print_trial("DEER.solve prebuilt", b_solve)
t_logps = print_trial("cache block logps", b_logps)
t_materialize = print_trial("materialize transitions", b_materialize)
t_sample = print_trial("sample public path", b_sample)

blocks = cld(N_samples, T)
estimated_block_ms = t_tape + t_build + t_solve + t_logps + t_materialize
estimated_total_ms = blocks * estimated_block_ms

println()
@printf "estimated one-block component sum: %.3f ms\n" estimated_block_ms
@printf "estimated total from components:   %.3f ms (%d blocks)\n" estimated_total_ms blocks
@printf "public sample total:               %.3f ms\n" t_sample
@printf "public sample us/sample:           %.3f\n" (1000 * t_sample / N_samples)
println()
