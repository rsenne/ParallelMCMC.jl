# ParallelMCMC Benchmarks

This package holds reproducible benchmark workloads for `ParallelMCMC.jl`.

## PR Regression Suite

The pull request workflow runs a small CPU-only suite against both the PR head
and the PR base commit, then compares median runtimes. It covers:

- one allocation-light MALA transition,
- the diagonal affine scan used by quasi-DEER,
- a full DEER block solve on a taped Gaussian MALA trajectory,
- the public `ParallelMALASampler` path on a small Bayesian logistic regression.

Run the same suite locally from the repository root with:

```bash
julia --project=benchmarks/ParallelMCMCBenchmarks \
  benchmarks/ParallelMCMCBenchmarks/scripts/pr_benchmarks.jl \
  --output pr-benchmarks.toml \
  --markdown pr-benchmarks.md
```

To compare two result files:

```bash
julia benchmarks/ParallelMCMCBenchmarks/scripts/compare_pr_benchmarks.jl \
  --base base-benchmarks.toml \
  --head head-benchmarks.toml \
  --markdown benchmark-comparison.md
```

CI marks a benchmark as `watch` above a 1.25x median-time ratio and fails above
1.75x. The thresholds can be adjusted with `PMCMC_BENCH_WARN_RATIO` and
`PMCMC_BENCH_FAIL_RATIO`.

## Manual GPU Sweeps

The existing scripts in `scripts/` are still intended for deeper manual
throughput and GPU investigations. They are not part of the PR gate because
standard GitHub-hosted runners do not provide CUDA hardware.
