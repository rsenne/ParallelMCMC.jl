"""
Allocation probe benchmark for GPU DEER logistic regression.

Purpose:
- Separate allocations in the DEER solve path into three pieces:
    1. rec.fwd_and_jvp_batch(Xbar, Z)
    2. DEER.deer_update!(...)
    3. DEER.solve(...; workspace=ws)
- Keep build/setup out of the timed region.
- Reuse a prebuilt recursion and preallocated workspace.

This is the file to run when the question is:
    "Where are the allocations still coming from?"
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
T = 8

X_f32, y_f32, _ = BayesLogReg.make_data(rng, N_data, D)
X_f32 = Float32.(X_f32)
y_f32 = Float32.(y_f32)

CUDA.functional() || error("CUDA is not functional in this environment")

X_gpu = CUDA.CuMatrix(X_f32)
y_gpu = CUDA.CuVector(y_f32)

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

epsilon = 0.1f0
maxiter = 50
tol_abs = 1.0f-4
tol_rel = 1.0f-3
damping = 0.5f0
probes = 1

x0_gpu = CUDA.zeros(Float32, D)

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

    return ParallelMCMC._build_mala_deer_rec(
        model,
        epsilon,
        tape,
        x0;
        cholM=cholM,
        backend=backend,
    )
end

# Prebuild recursion and workspace once.
rec = build_raw_deer_problem(
    MersenneTwister(42),
    model_gpu,
    x0_gpu;
    epsilon=epsilon,
    T=T,
)

ws = DEER.DEERWorkspace(x0_gpu, T)

# Fixed inputs for repeated probes.
S0 = similar(x0_gpu, D, T)
S0 .= reshape(x0_gpu, D, 1)
Xbar = ws.Xbar
Z = ws.Z
DEER._rademacher_matrix!(Z, MersenneTwister(7))
copyto!(view(Xbar, :, 1), x0_gpu)
if T > 1
    Xbar[:, 2:end] .= S0[:, 1:(T - 1)]
end

println("=" ^ 72)
println("Allocation probe for GPU DEER solve path")
println("Model: Bayesian logistic regression   D=$D   N_data=$N_data   T=$T")
println("All timed sections use prebuilt rec + preallocated workspace")
println("=" ^ 72)
println()

# Warmups
if rec.fwd_and_jvp_batch !== nothing
    FT_warm, Jt_warm = rec.fwd_and_jvp_batch(Xbar, Z)
    CUDA.synchronize()
end

Sout_warm = similar(S0)
DEER.deer_update!(
    ws,
    Sout_warm,
    rec,
    x0_gpu,
    S0;
    jacobian=:stoch_diag,
    damping=damping,
    probes=probes,
    rng=MersenneTwister(11),
)
CUDA.synchronize()

Ssolve_warm = DEER.solve(
    rec,
    x0_gpu;
    workspace=ws,
    tol_abs=tol_abs,
    tol_rel=tol_rel,
    maxiter=maxiter,
    jacobian=:stoch_diag,
    damping=damping,
    probes=probes,
    rng=MersenneTwister(12),
)
CUDA.synchronize()

if rec.fwd_and_jvp_batch === nothing
    println("rec.fwd_and_jvp_batch === nothing; skipping direct batch benchmark")
    println()
else
    println("[GPU] rec.fwd_and_jvp_batch(Xbar, Z)")
    b_batch = @benchmark begin
        FT, Jt = $rec.fwd_and_jvp_batch($Xbar, $Z)
        CUDA.synchronize()
        FT, Jt
    end samples=8 evals=1
    show(stdout, MIME("text/plain"), b_batch)
    println("\n")
    @printf "    median batch time: %.3f ms\n\n" (median(b_batch).time / 1e6)
end

println("[GPU] DEER.deer_update!(ws, Sout, rec, x0, S0)")
b_update = @benchmark begin
    DEER.deer_update!(
        $ws,
        $Sout_warm,
        $rec,
        $x0_gpu,
        $S0;
        jacobian=:stoch_diag,
        damping=$damping,
        probes=$probes,
        rng=MersenneTwister(21),
    )
    CUDA.synchronize()
end samples=6 evals=1
show(stdout, MIME("text/plain"), b_update)
println("\n")
@printf "    median deer_update! time: %.3f ms\n\n" (median(b_update).time / 1e6)

println("[GPU] DEER.solve(rec, x0; workspace=ws)")
b_solve = @benchmark begin
    S = DEER.solve(
        $rec,
        $x0_gpu;
        workspace=$ws,
        tol_abs=$tol_abs,
        tol_rel=$tol_rel,
        maxiter=$maxiter,
        jacobian=:stoch_diag,
        damping=$damping,
        probes=$probes,
        rng=MersenneTwister(22),
    )
    CUDA.synchronize()
    S
end samples=4 evals=1
show(stdout, MIME("text/plain"), b_solve)
println("\n")
@printf "    median solve time: %.3f ms\n\n" (median(b_solve).time / 1e6)

println("=" ^ 104)
println("Summary")
println("=" ^ 104)
@printf "%-28s  %12s  %12s\n" "section" "time ms" "allocs"

if rec.fwd_and_jvp_batch !== nothing
    @printf "%-28s  %12.3f  %12d\n" "fwd_and_jvp_batch" (median(b_batch).time / 1e6) median(b_batch).memory == 0 ? median(b_batch).allocs : median(b_batch).allocs
end
@printf "%-28s  %12.3f  %12d\n" "deer_update!" (median(b_update).time / 1e6) median(b_update).allocs
@printf "%-28s  %12.3f  %12d\n" "solve(workspace=ws)" (median(b_solve).time / 1e6) median(b_solve).allocs
println()
