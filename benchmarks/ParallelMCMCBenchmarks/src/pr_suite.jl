module ParallelMCMCPrBenchmarks

using AbstractMCMC: sample
using BenchmarkTools
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

using ParallelMCMC

const MALA = ParallelMCMC.MALA
const DEER = ParallelMCMC.DEER
const DEERScan = ParallelMCMC.DEERScan
if isdefined(parentmodule(@__MODULE__), :BayesLogReg)
    const BayesLogReg = getfield(parentmodule(@__MODULE__), :BayesLogReg)
else
    using ParallelMCMCBenchmarks: BayesLogReg
end

struct BenchmarkCase
    name::String
    description::String
    benchmark::Any
end

function _stdnormal_logp(x)
    return -0.5 * dot(x, x)
end

function _stdnormal_grad(x)
    return -x
end

function _stdnormal_hvp(x, v)
    return -v
end

function _quartic_logp(x)
    return -0.5 * dot(x, x) - 0.05 * sum(abs2.(x) .^ 2)
end

function _quartic_grad(x)
    return @. -x - 0.2 * x^3
end

function _make_stdnormal_rec(rng::AbstractRNG, dim::Int, steps::Int, epsilon::Float64)
    tape = [(noise=randn(rng, dim), u=rand(rng)) for _ in 1:steps]

    step_fwd =
        (x, te) ->
            MALA.mala_step_taped(_stdnormal_logp, _stdnormal_grad, x, epsilon, te.noise, te.u)
    jvp =
        (x, te, v) -> MALA.mala_step_surrogate_sigmoid_jvp(
            _stdnormal_logp,
            _stdnormal_grad,
            x,
            epsilon,
            te.noise,
            te.u,
            v,
            _stdnormal_hvp,
        )
    fwd_and_jvp =
        (x, te, v) -> MALA.mala_step_taped_and_jvp(
            _stdnormal_logp,
            _stdnormal_grad,
            x,
            epsilon,
            te.noise,
            te.u,
            v,
            _stdnormal_hvp,
        )

    return DEER.TapedRecursion(step_fwd, jvp, tape; fwd_and_jvp=fwd_and_jvp)
end

function _mala_step_case()
    rng = MersenneTwister(9001)
    dim = 16
    x = randn(rng, dim)
    x_next = similar(x)
    noise = randn(rng, dim)
    u = rand(rng)
    workspace = MALA.MALAWorkspace(x)
    epsilon = 0.04

    bench = @benchmarkable MALA.mala_step_with_logα!(
        $x_next,
        $workspace,
        $_stdnormal_logp,
        $_stdnormal_grad,
        $x,
        $epsilon,
        $noise,
        $u,
    ) evals = 1

    return BenchmarkCase(
        "mala.step_stdnormal_16d",
        "One allocation-light MALA step with log-acceptance on a 16-D Gaussian target.",
        bench,
    )
end

function _affine_scan_case()
    rng = MersenneTwister(9002)
    dim = 32
    steps = 256
    A = 0.92 .+ 0.02 .* randn(rng, dim, steps)
    B = 0.10 .* randn(rng, dim, steps)
    s0 = randn(rng, dim)
    output = similar(A)
    workspace = DEERScan.AffineScanWorkspace(A)

    bench = @benchmarkable DEERScan.solve_affine_scan_diag!(
        $output, $A, $B, $s0, $workspace
    ) evals = 1

    return BenchmarkCase(
        "deer.affine_scan_32x256",
        "Diagonal affine prefix scan used by quasi-DEER, D=32 and T=256.",
        bench,
    )
end

function _deer_solve_case()
    rng = MersenneTwister(9003)
    dim = 8
    steps = 64
    epsilon = 0.06
    s0 = randn(rng, dim)
    rec = _make_stdnormal_rec(rng, dim, steps, epsilon)
    workspace = DEER.DEERWorkspace(s0, steps)

    bench = @benchmarkable DEER.solve(
        $rec,
        $s0;
        tol_abs=1e-8,
        tol_rel=1e-6,
        maxiter=80,
        jacobian=:diag,
        damping=0.5,
        rng=rng,
        workspace=$workspace,
        copy_result=false,
    ) setup = (rng = MersenneTwister(42)) evals = 1

    return BenchmarkCase(
        "deer.solve_stdnormal_8x64",
        "A full DEER block solve against a taped 8-D Gaussian MALA trajectory, T=64.",
        bench,
    )
end

function _parallel_mala_sample_case()
    rng = MersenneTwister(9004)
    dim = 4
    n_obs = 64
    X, y, _ = BayesLogReg.make_data(rng, n_obs, dim)
    logp, gradlogp, hvp = BayesLogReg.make_problem_with_hvp(X, y)
    model = ParallelMCMC.DensityModel(logp, gradlogp, dim; hvp=hvp)
    sampler = ParallelMCMC.ParallelMALASampler(
        0.045;
        T=16,
        maxiter=60,
        tol_abs=1e-6,
        tol_rel=1e-5,
        jacobian=:diag,
        damping=0.5,
    )
    initial_params = zeros(dim)

    bench = @benchmarkable sample(
        rng,
        $model,
        $sampler,
        64;
        initial_params=$initial_params,
        progress=false,
    ) setup = (rng = MersenneTwister(42)) evals = 1

    return BenchmarkCase(
        "sampler.parallel_mala_logreg_4d_64samples",
        "Public ParallelMALASampler path on a small Bayesian logistic regression problem.",
        bench,
    )
end

function _parallel_mala_sample_no_hvp_case()
    rng = MersenneTwister(9005)
    dim = 4
    model = ParallelMCMC.DensityModel(_quartic_logp, _quartic_grad, dim)
    sampler = ParallelMCMC.ParallelMALASampler(
        0.04;
        T=16,
        maxiter=60,
        tol_abs=1e-6,
        tol_rel=1e-5,
        jacobian=:diag,
        damping=0.5,
    )
    initial_params = zeros(dim)

    bench = @benchmarkable sample(
        rng,
        $model,
        $sampler,
        64;
        initial_params=$initial_params,
        progress=false,
    ) setup = (rng = MersenneTwister(42)) evals = 1

    return BenchmarkCase(
        "sampler.parallel_mala_quartic_no_hvp_4d_64samples",
        "Public ParallelMALASampler path on a non-Gaussian target with no model-provided HVP.",
        bench,
    )
end

function make_pr_benchmarks()
    return [
        _mala_step_case(),
        _affine_scan_case(),
        _deer_solve_case(),
        _parallel_mala_sample_case(),
        _parallel_mala_sample_no_hvp_case(),
    ]
end

function _trial_summary(trial::BenchmarkTools.Trial)
    med = median(trial)
    min_est = minimum(trial)
    return Dict{String,Any}(
        "median_ns" => med.time,
        "minimum_ns" => min_est.time,
        "median_memory_bytes" => med.memory,
        "median_allocs" => med.allocs,
        "samples" => length(trial.times),
    )
end

function _format_time(ns::Real)
    ns < 1e3 && return @sprintf("%.0f ns", ns)
    ns < 1e6 && return @sprintf("%.2f us", ns / 1e3)
    ns < 1e9 && return @sprintf("%.2f ms", ns / 1e6)
    return @sprintf("%.2f s", ns / 1e9)
end

function run_pr_benchmarks(; seconds::Real=0.5, samples::Int=8)
    cases = make_pr_benchmarks()
    results = Dict{String,Any}()

    println("Running ParallelMCMC PR benchmarks")
    println("  Julia: ", VERSION)
    println("  seconds per case: ", seconds)
    println("  minimum samples per case: ", samples)
    println()

    for case in cases
        println("==> ", case.name)
        println("    ", case.description)
        trial = run(case.benchmark; seconds=seconds, samples=samples)
        summary = _trial_summary(trial)
        summary["description"] = case.description
        results[case.name] = summary

        println(
            "    median: ",
            _format_time(summary["median_ns"]),
            "  min: ",
            _format_time(summary["minimum_ns"]),
            "  allocs: ",
            summary["median_allocs"],
            "  memory: ",
            summary["median_memory_bytes"],
            " bytes",
        )
        println()
    end

    return Dict{String,Any}(
        "metadata" => Dict{String,Any}(
            "julia_version" => string(VERSION),
            "seconds" => Float64(seconds),
            "samples" => samples,
        ),
        "benchmarks" => results,
    )
end

function write_results(path::AbstractString, results::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, results; sorted=true)
    end
    return path
end

function write_markdown(path::AbstractString, results::Dict{String,Any})
    benchmarks = results["benchmarks"]
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# ParallelMCMC PR Benchmarks")
        println(io)
        println(io, "| Benchmark | Median | Minimum | Allocs | Memory |")
        println(io, "|---|---:|---:|---:|---:|")
        for name in sort(collect(keys(benchmarks)))
            result = benchmarks[name]
            println(
                io,
                "| `",
                name,
                "` | ",
                _format_time(result["median_ns"]),
                " | ",
                _format_time(result["minimum_ns"]),
                " | ",
                result["median_allocs"],
                " | ",
                result["median_memory_bytes"],
                " B |",
            )
        end
        println(io)
    end
    return path
end

end # module
