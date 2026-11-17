#!/usr/bin/env bash
#
# Runs the test suite, Reactant tests included, on this machine's GPU and uploads
# the coverage to Codecov under the `gpu` flag for the checked-out commit. GitHub CI
# has no GPU, so the CUDA and Reactant-on-CUDA paths are only covered by this run.
#
#   CODECOV_TOKEN=... bash ci/gpu-coverage.sh
#
# Push the commit first: Codecov attaches the upload to its SHA. Optional variables:
# JULIA (default `julia`), ENVDIR (the test environment, default
# ~/.cache/parallelmcmc-gpu-cov), DRY_RUN=1 to stop before uploading.

set -euo pipefail

if [[ -z "${DRY_RUN:-}" ]]; then
    : "${CODECOV_TOKEN:?set CODECOV_TOKEN to the repository upload token}"
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JULIA="${JULIA:-julia}"
ENVDIR="${ENVDIR:-$HOME/.cache/parallelmcmc-gpu-cov}"
# The checkout may be a Windows one seen from WSL: CRLF files and another owner.
git_() { git -c safe.directory="$REPO" -c core.autocrlf=true -C "$REPO" "$@"; }

cd "$REPO"
if [[ -n "$(git_ status --porcelain --untracked-files=no)" ]]; then
    echo "Uncommitted changes: the coverage would not match the commit." >&2
    [[ -n "${DRY_RUN:-}" ]] || exit 1
fi
SHA="$(git_ rev-parse HEAD)"
BRANCH="$(git_ rev-parse --abbrev-ref HEAD)"

# Reactant is only an [extras] entry in test/Project.toml, so Pkg.test would not
# install it. Build the environment outside the repo to leave test/ untouched.
mkdir -p "$ENVDIR"
cp test/Project.toml "$ENVDIR/Project.toml"
"$JULIA" --project="$ENVDIR" -e '
    using Pkg
    Pkg.develop(path=ARGS[1])
    Pkg.add("Reactant")
    Pkg.instantiate()
' "$REPO"

find src ext -name '*.cov' -delete
rm -f lcov.info
# XLA reserves 75% of the GPU by default, which starves the CUDA.jl tests.
export XLA_REACTANT_GPU_MEM_FRACTION="${XLA_REACTANT_GPU_MEM_FRACTION:-0.3}"
PARALLELMCMC_TEST_REACTANT=true "$JULIA" --code-coverage=@"$REPO" \
    --project="$ENVDIR" test/runtests.jl || echo "Tests failed; uploading coverage anyway." >&2

"$JULIA" -e '
    using Pkg
    Pkg.activate(; temp=true)
    Pkg.add("CoverageTools")
    using CoverageTools
    LCOV.writefile("lcov.info", vcat(process_folder("src"), process_folder("ext")))
'
find src ext -name '*.cov' -delete

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "Wrote lcov.info for $SHA; not uploading (DRY_RUN)."
    exit 0
fi

if [[ ! -x ./codecov ]]; then
    curl -fsSL -o codecov https://cli.codecov.io/latest/linux/codecov
    chmod +x codecov
fi
./codecov upload-process \
    --token "$CODECOV_TOKEN" \
    --git-service github \
    --sha "$SHA" \
    --branch "$BRANCH" \
    --flag gpu \
    --file lcov.info \
    --disable-search
rm -f lcov.info
