"""
Allocation probe for DEER recursion internals.

Purpose
-------
Isolate allocations in:
  1. rec.step_fwd(x, tape_t)
  2. rec.jvp(x, tape_t, v)
  3. one "stochastic-diag" step body using preallocated zbuf/jt_buf
  4. random probe generation helpers

This is meant to answer:
    "Are the remaining solve allocations coming from the recursion closures
     (step_fwd / jvp), from probe generation, or from DEER scaffolding?"

Assumptions
-----------
- Your local tree includes the patched DEER / DEERScan files.
- Bayesian logistic-regression benchmark setup is available from
  ParallelMCMCBenchmarks.BayesLogReg, as in your earlier scripts.

Notes
-----
- Everything here uses a prebuilt recursion and fixed inputs.
- The timed region excludes tape / recursion construction.
- If `rec.fwd_and_jvp !== nothing`, we benchmark that too.
- If `rec.fwd_and_jvp_batch !== nothing`, we benchmark that too.
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

epsilon = 0.1f0
maxiter = 50
tol_abs = 1.0f-4
tol_rel = 1.0f-3
damping = 0.5f0
probes = 1

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

rec = build_raw_deer_problem(
    MersenneTwister(42),
    model_gpu,
    x0_gpu;
    epsilon=epsilon,
    T=T,
)

tape_t = rec.tape[1]
x = copy(x0_gpu)
v = similar(x0_gpu)
fill!(v, 1f0)

# Buffers for probing the same branch DEER.deer_update! uses.
zbuf = similar(x0_gpu)
jt_buf = similar(x0_gpu)
fill!(zbuf, 1f0)
fill!(jt_buf, 0f0)

# Optional batch-path inputs.
Xbar = similar(x0_gpu, D, T)
Z = similar(x0_gpu, D, T)
Xbar .= reshape(x0_gpu, D, 1)
fill!(Z, 1f0)

"""
A stripped-down copy of the sequential :stoch_diag body for a single time step,
using preallocated zbuf / jt_buf so we can see what still allocates.
"""
function one_stoch_diag_step!(
    rec,
    xbar,
    tape_t,
    zbuf,
    jt_buf;
    probes::Int=1,
    rng::AbstractRNG=Random.default_rng(),
)
    _rademacher!(zbuf, rng)
    ft, jvp1 = if rec.fwd_and_jvp !== nothing
        rec.fwd_and_jvp(xbar, tape_t, zbuf)
    else
        (rec.step_fwd(xbar, tape_t), rec.jvp(xbar, tape_t, zbuf))
    end

    @. jt_buf = zbuf * jvp1
    for _ in 2:probes
        _rademacher!(zbuf, rng)
        jvp_k = rec.jvp(xbar, tape_t, zbuf)
        @. jt_buf += zbuf * jvp_k
    end
    if probes > 1
        jt_buf .*= one(eltype(jt_buf)) / probes
    end
    return ft, jt_buf
end

println("=" ^ 72)
println("Allocation probe for DEER recursion internals")
println("Model: Bayesian logistic regression   D=$D   N_data=$N_data   T=$T")
println("All timed sections use fixed rec/tape/x/v and preallocated buffers")
println("=" ^ 72)
println()

println("[GPU] rec.step_fwd(x, tape_t)")
step_warm = rec.step_fwd(x, tape_t)
CUDA.synchronize()
b_step = @benchmark begin
    y = rec.step_fwd($x, $tape_t)
    CUDA.synchronize()
    y
end samples=8 evals=1
show(stdout, MIME("text/plain"), b_step)
println("\n")
@printf "    median step_fwd time: %.3f ms\n\n" (median(b_step).time / 1e6)

println("[GPU] rec.jvp(x, tape_t, v)")
jvp_warm = rec.jvp(x, tape_t, v)
CUDA.synchronize()
b_jvp = @benchmark begin
    y = rec.jvp($x, $tape_t, $v)
    CUDA.synchronize()
    y
end samples=8 evals=1
show(stdout, MIME("text/plain"), b_jvp)
println("\n")
@printf "    median jvp time: %.3f ms\n\n" (median(b_jvp).time / 1e6)

if rec.fwd_and_jvp !== nothing
    println("[GPU] rec.fwd_and_jvp(x, tape_t, v)")
    fj_warm = rec.fwd_and_jvp(x, tape_t, v)
    CUDA.synchronize()
    b_fj = @benchmark begin
        y = rec.fwd_and_jvp($x, $tape_t, $v)
        CUDA.synchronize()
        y
    end samples=8 evals=1
    show(stdout, MIME("text/plain"), b_fj)
    println("\n")
    @printf "    median fwd_and_jvp time: %.3f ms\n\n" (median(b_fj).time / 1e6)
else
    b_fj = nothing
    println("rec.fwd_and_jvp === nothing; skipping fused single-step benchmark\n")
end

println("[GPU] _rademacher!(zbuf, rng)")
# assume DEER._rademacher! is visible; if not, fall back to local import via module
r_warm = DEER._rademacher!(zbuf, MersenneTwister(7))
CUDA.synchronize()
b_radem = @benchmark begin
    DEER._rademacher!($zbuf, MersenneTwister(7))
    CUDA.synchronize()
end samples=8 evals=1
show(stdout, MIME("text/plain"), b_radem)
println("\n")
@printf "    median _rademacher! time: %.3f ms\n\n" (median(b_radem).time / 1e6)

println("[GPU] one_stoch_diag_step!(...)")
os_warm = one_stoch_diag_step!(rec, x, tape_t, zbuf, jt_buf; probes=probes, rng=MersenneTwister(11))
CUDA.synchronize()
b_one = @benchmark begin
    y = one_stoch_diag_step!($rec, $x, $tape_t, $zbuf, $jt_buf; probes=$probes, rng=MersenneTwister(11))
    CUDA.synchronize()
    y
end samples=8 evals=1
show(stdout, MIME("text/plain"), b_one)
println("\n")
@printf "    median one-step stoch-diag time: %.3f ms\n\n" (median(b_one).time / 1e6)

if rec.fwd_and_jvp_batch !== nothing
    println("[GPU] rec.fwd_and_jvp_batch(Xbar, Z)")
    fb_warm = rec.fwd_and_jvp_batch(Xbar, Z)
    CUDA.synchronize()
    b_fb = @benchmark begin
        y = rec.fwd_and_jvp_batch($Xbar, $Z)
        CUDA.synchronize()
        y
    end samples=8 evals=1
    show(stdout, MIME("text/plain"), b_fb)
    println("\n")
    @printf "    median fwd_and_jvp_batch time: %.3f ms\n\n" (median(b_fb).time / 1e6)
else
    b_fb = nothing
    println("rec.fwd_and_jvp_batch === nothing; skipping batch benchmark\n")
end

println("=" ^ 104)
println("Summary")
println("=" ^ 104)
@printf "%-30s  %12s  %12s\n" "section" "time ms" "allocs"

@printf "%-30s  %12.3f  %12d\n" "step_fwd" (median(b_step).time / 1e6) median(b_step).allocs
@printf "%-30s  %12.3f  %12d\n" "jvp"      (median(b_jvp).time / 1e6) median(b_jvp).allocs
if b_fj !== nothing
    @printf "%-30s  %12.3f  %12d\n" "fwd_and_jvp" (median(b_fj).time / 1e6) median(b_fj).allocs
end
@printf "%-30s  %12.3f  %12d\n" "_rademacher!" (median(b_radem).time / 1e6) median(b_radem).allocs
@printf "%-30s  %12.3f  %12d\n" "one_stoch_diag_step!" (median(b_one).time / 1e6) median(b_one).allocs
if b_fb !== nothing
    @printf "%-30s  %12.3f  %12d\n" "fwd_and_jvp_batch" (median(b_fb).time / 1e6) median(b_fb).allocs
end
println()
