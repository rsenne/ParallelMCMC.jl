#=
Regenerate `docs/src/assets/german_credit.png` from the timings in
`german_credit_timings.csv`. The timings come from
`benchmarks/ParallelMCMCBenchmarks/scripts/german_credit.jl` with `PMCMC_OUT` set.

Run from the repository root:

    julia --project=docs/figures docs/figures/german_credit.jl
=#

using CairoMakie
using DelimitedFiles

const DATA = joinpath(@__DIR__, "german_credit_timings.csv")
const OUTPUT = joinpath(@__DIR__, "..", "src", "assets", "german_credit.png")

const SERIES = [
    ("DEER MALA, GPU", "parallel MALA, GPU", :circle),
    ("MALA, CPU", "sequential MALA, CPU", :rect),
    ("MALA, GPU", "sequential MALA, GPU", :utriangle),
]

table = readdlm(DATA, ','; header=true)[1]
function series(method)
    rows = sortslices(table[table[:, 2] .== method, :]; dims=1, by=r -> r[1])
    return Float64.(rows[:, 1]), Float64.(rows[:, 3])
end

set_theme!(
    merge(
        theme_latexfonts(),
        Theme(;
            fontsize=16,
            Axis=(
                xgridvisible=false,
                ygridvisible=false,
                spinewidth=0.8,
                xtickwidth=0.8,
                ytickwidth=0.8,
                xticksize=4,
                yticksize=4,
                yminorticksvisible=true,
                yminorticks=IntervalsBetween(9),
                yminorticksize=2,
                yminortickwidth=0.6,
            ),
            Legend=(framevisible=false,),
        ),
    ),
)
fig = Figure(; size=(560, 380))
ax = Axis(
    fig[1, 1];
    xlabel=L"chain length $L$",
    ylabel="wall-clock time (s)",
    xscale=log10,
    yscale=log10,
    xticks=([1_000, 4_000, 16_000, 64_000], ["1K", "4K", "16K", "64K"]),
)
for (method, label, marker) in SERIES
    L, t = series(method)
    scatterlines!(
        ax,
        L,
        t;
        marker,
        markersize=10,
        linewidth=1.5,
        strokecolor=:white,
        strokewidth=1,
        label,
    )
end
axislegend(ax; position=:lt)

save(OUTPUT, fig; px_per_unit=2)
println("wrote ", normpath(OUTPUT))
