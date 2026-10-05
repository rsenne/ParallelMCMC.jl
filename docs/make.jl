using ParallelMCMC
using Documenter
using Documenter.Remotes: GitHub

DocMeta.setdocmeta!(ParallelMCMC, :DocTestSetup, :(using ParallelMCMC); recursive=true)

const pages = [
    "Home" => "index.md",
    "Getting started" => "10-getting-started.md",
    "Defining models" => "12-models.md",
    "GPU execution" => "15-gpu.md",
    "Troubleshooting" => "16-troubleshooting.md",
    "How the algorithm works" => "20-algorithms.md",
    "API reference" => "95-reference.md",
    "Contributing" => [
        "Contributor guide" => "90-contributing.md",
        "Developer guide" => "91-developer.md",
    ],
]

makedocs(;
    modules=[ParallelMCMC],
    authors="Ryan Senne <rsenne@bu.edu>",
    repo=GitHub("rsenne", "ParallelMCMC.jl"),
    sitename="ParallelMCMC.jl",
    checkdocs=:none,
    format=Documenter.HTML(;
        canonical="https://rsenne.github.io/ParallelMCMC.jl",
        repolink="https://github.com/rsenne/ParallelMCMC.jl",
        edit_link="main",
    ),
    pages=pages,
)

deploydocs(; repo="github.com/rsenne/ParallelMCMC.jl", devbranch="main")
