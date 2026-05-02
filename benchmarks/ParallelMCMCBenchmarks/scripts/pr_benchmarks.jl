#!/usr/bin/env julia

using ParallelMCMCBenchmarks

const PRSuite = ParallelMCMCBenchmarks.ParallelMCMCPrBenchmarks

function _option(args, name, default=nothing)
    index = findfirst(==(name), args)
    index === nothing && return default
    index < length(args) || error("missing value for $name")
    return args[index + 1]
end

function _has_flag(args, name)
    return any(==(name), args)
end

function _usage()
    return """
    Usage:
      julia --project benchmarks/ParallelMCMCBenchmarks/scripts/pr_benchmarks.jl [options]

    Options:
      --output PATH      Write machine-readable TOML benchmark results.
      --markdown PATH    Write a Markdown summary table.
      --seconds N        Target seconds per benchmark case. Default: 0.5
      --samples N        Minimum samples per benchmark case. Default: 8
      --help             Show this help message.
    """
end

function main(args=ARGS)
    if _has_flag(args, "--help")
        print(_usage())
        return 0
    end

    seconds = parse(Float64, _option(args, "--seconds", get(ENV, "PMCMC_BENCH_SECONDS", "0.5")))
    samples = parse(Int, _option(args, "--samples", get(ENV, "PMCMC_BENCH_SAMPLES", "8")))
    output = _option(args, "--output", "")
    markdown = _option(args, "--markdown", "")

    seconds > 0 || error("--seconds must be positive")
    samples > 0 || error("--samples must be positive")

    results = PRSuite.run_pr_benchmarks(; seconds=seconds, samples=samples)

    if !isempty(output)
        PRSuite.write_results(output, results)
        println("Wrote TOML results to ", output)
    end

    if !isempty(markdown)
        PRSuite.write_markdown(markdown, results)
        println("Wrote Markdown summary to ", markdown)
    end

    return 0
end

exit(main())
