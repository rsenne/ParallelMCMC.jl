#!/usr/bin/env julia

using Printf
using TOML

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
      julia compare_pr_benchmarks.jl --base base.toml --head head.toml [options]

    Options:
      --base PATH        Base-branch benchmark TOML.
      --head PATH        PR-head benchmark TOML.
      --markdown PATH    Write a Markdown comparison table.
      --fail-ratio N     Fail when head/base median time exceeds N. Default: 1.75
      --warn-ratio N     Mark warning when head/base median time exceeds N. Default: 1.25
      --help             Show this help message.
    """
end

function _format_time(ns::Real)
    ns < 1e3 && return @sprintf("%.0f ns", ns)
    ns < 1e6 && return @sprintf("%.2f us", ns / 1e3)
    ns < 1e9 && return @sprintf("%.2f ms", ns / 1e6)
    return @sprintf("%.2f s", ns / 1e9)
end

function _format_ratio(ratio::Real)
    return @sprintf("%.2fx", ratio)
end

function _format_delta(bytes::Real)
    sign = bytes >= 0 ? "+" : "-"
    return sign * string(abs(round(Int, bytes))) * " B"
end

function _status(ratio::Real, warn_ratio::Real, fail_ratio::Real)
    ratio >= fail_ratio && return "regression"
    ratio >= warn_ratio && return "watch"
    ratio <= inv(warn_ratio) && return "faster"
    return "ok"
end

function compare_results(base_path, head_path; warn_ratio=1.25, fail_ratio=1.75)
    base = TOML.parsefile(base_path)
    head = TOML.parsefile(head_path)
    base_benchmarks = base["benchmarks"]
    head_benchmarks = head["benchmarks"]

    names = sort(collect(intersect(keys(base_benchmarks), keys(head_benchmarks))))
    isempty(names) && error("no common benchmark names found")

    rows = Vector{Dict{String,Any}}()
    failed = false

    for name in names
        base_result = base_benchmarks[name]
        head_result = head_benchmarks[name]
        base_ns = Float64(base_result["median_ns"])
        head_ns = Float64(head_result["median_ns"])
        ratio = head_ns / base_ns
        status = _status(ratio, warn_ratio, fail_ratio)
        failed |= status == "regression"

        push!(
            rows,
            Dict{String,Any}(
                "name" => name,
                "base_ns" => base_ns,
                "head_ns" => head_ns,
                "ratio" => ratio,
                "base_allocs" => Int(base_result["median_allocs"]),
                "head_allocs" => Int(head_result["median_allocs"]),
                "memory_delta" =>
                    Float64(head_result["median_memory_bytes"]) -
                    Float64(base_result["median_memory_bytes"]),
                "status" => status,
            ),
        )
    end

    return rows, failed
end

function write_markdown(path, rows; warn_ratio, fail_ratio)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# ParallelMCMC Benchmark Comparison")
        println(io)
        println(
            io,
            "Median runtime is compared against the pull request base commit. ",
            "A benchmark fails the job when `head / base > ",
            fail_ratio,
            "` and is marked `watch` above `",
            warn_ratio,
            "`.",
        )
        println(io)
        println(io, "| Benchmark | Base median | PR median | Ratio | Allocs | Memory delta | Status |")
        println(io, "|---|---:|---:|---:|---:|---:|---|")
        for row in rows
            println(
                io,
                "| `",
                row["name"],
                "` | ",
                _format_time(row["base_ns"]),
                " | ",
                _format_time(row["head_ns"]),
                " | ",
                _format_ratio(row["ratio"]),
                " | ",
                row["base_allocs"],
                " -> ",
                row["head_allocs"],
                " | ",
                _format_delta(row["memory_delta"]),
                " | ",
                row["status"],
                " |",
            )
        end
        println(io)
    end
    return path
end

function print_summary(rows)
    println("ParallelMCMC benchmark comparison")
    for row in rows
        println(
            "  ",
            row["name"],
            ": ",
            _format_time(row["base_ns"]),
            " -> ",
            _format_time(row["head_ns"]),
            " (",
            _format_ratio(row["ratio"]),
            ", ",
            row["status"],
            ")",
        )
    end
end

function main(args=ARGS)
    if _has_flag(args, "--help")
        print(_usage())
        return 0
    end

    base = _option(args, "--base")
    head = _option(args, "--head")
    base === nothing && error("--base is required")
    head === nothing && error("--head is required")

    warn_ratio = parse(Float64, _option(args, "--warn-ratio", get(ENV, "PMCMC_BENCH_WARN_RATIO", "1.25")))
    fail_ratio = parse(Float64, _option(args, "--fail-ratio", get(ENV, "PMCMC_BENCH_FAIL_RATIO", "1.75")))
    markdown = _option(args, "--markdown", "")

    warn_ratio > 1 || error("--warn-ratio must be greater than 1")
    fail_ratio > warn_ratio || error("--fail-ratio must be greater than --warn-ratio")

    rows, failed = compare_results(base, head; warn_ratio=warn_ratio, fail_ratio=fail_ratio)
    print_summary(rows)

    if !isempty(markdown)
        write_markdown(markdown, rows; warn_ratio=warn_ratio, fail_ratio=fail_ratio)
        println("Wrote Markdown comparison to ", markdown)
    end

    return failed ? 1 : 0
end

exit(main())
