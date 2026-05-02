"""
Minimal DEER profiling script for ProfileView / VS Code @profview.

Purpose
-------
Build a small GPU DEER problem once, warm it up once, then run exactly one
profiled solve on a smaller problem so the flame graph is readable.

Usage examples
--------------
julia --project minimal_profview_deer.jl

Then inside Julia / VS Code:
    include("minimal_profview_deer.jl")

This file will:
  - build `rec`, `ws`, `x0_gpu`
  - warm up one solve
  - expose `run_profview()` and `run_profview_allocs()`

You can then call:
    run_profview()
or:
    run_profview_allocs()
"""

using Random
using CUDA
using ParallelMCMC
using ParallelMCMCBenchmarks

const BayesLogReg = ParallelMCMCBenchmarks.BayesLogReg

# Smaller / more readable profiling case
D = 64
N_data = 10000
const T = 8

const epsilon = 0.1f0
const maxiter = 10
const tol_abs = 1.0f-4
const tol_rel = 1.0f-3
const damping = 1.0f0
const probes = 1

CUDA.functional() || error("CUDA is not functional in this environment")

function build_raw_deer_problem(
    rng::Random.AbstractRNG,
    model::DensityModel,
    x0::AbstractVector;
    epsilon::Float32,
    T::Int,
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

function build_problem(; seed=20251231)
    rng = MersenneTwister(seed)

    X_f32, y_f32, _ = BayesLogReg.make_data(rng, N_data, D)
    X_f32 = Float32.(X_f32)
    y_f32 = Float32.(y_f32)

    X_gpu = CUDA.CuMatrix(X_f32)
    y_gpu = CUDA.CuVector(y_f32)

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

    x0_gpu = CUDA.zeros(Float32, D)

    rec = build_raw_deer_problem(
        MersenneTwister(42), model_gpu, x0_gpu; epsilon=epsilon, T=T
    )

    ws = DEER.DEERWorkspace(x0_gpu, T)

    return (; model_gpu, x0_gpu, rec, ws)
end

const PROF_STATE = build_problem()

function solve_once(; seed=42)
    rng = MersenneTwister(seed)
    S = DEER.solve(
        PROF_STATE.rec,
        PROF_STATE.x0_gpu;
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        maxiter=maxiter,
        jacobian=:stoch_diag,
        damping=damping,
        probes=probes,
        rng=rng,
        workspace=PROF_STATE.ws,
        copy_result=false,
    )
    CUDA.synchronize()
    return S
end

# Warmup so compilation doesn't dominate the profile
println("Warming up one DEER solve...")
solve_once()
println("Warmup done.")
println()
println("Call one of these in Julia / VS Code:")
println("    run_profview()")
println("    run_profview_allocs()")
println()

function run_profview()
    @profview solve_once()
end

function run_profview_allocs()
    @profview_allocs solve_once()
end
