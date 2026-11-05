#=
Regenerate `docs/src/assets/deer_iterations.gif` for the README and the docs landing page.

We fix the noise and uniforms of a 2 000-step MALA chain on a mixture of three Gaussians
placed like the dots of the Julia logo, then run DEER on that tape with the package's own
`DEER.deer_update!`. Each frame is one DEER iteration. The sequential chain is drawn in
grey and the current DEER iterate in blue on top of it. The bottom panel shows
`max |parallel - sequential|` over the coordinates at every step.

Run from the repository root:

    julia --project=docs/figures docs/figures/deer_iterations.jl
=#

using CairoMakie
using LinearAlgebra
using Printf
using Random

using ParallelMCMC
const DEER = ParallelMCMC.DEER
const MALA = ParallelMCMC.MALA

const OUTPUT = joinpath(@__DIR__, "..", "src", "assets", "deer_iterations.gif")

# Target: equal components centred on the three dots of the Julia logo, in the order
# red (bottom left), purple (bottom right), green (top).

const CENTERS = [[-1.05, -0.55], [1.05, -0.55], [0.0, 1.02]]
const LOGO_COLORS = [colorant"#CB3C33", colorant"#9558B2", colorant"#389826"]
const SIGMAS = [0.38, 0.38, 0.38]
const LOG_WEIGHTS = zeros(3)

function log_components(x)
    return [
        LOG_WEIGHTS[k] - log(2π * SIGMAS[k]^2) -
        sum(abs2, x .- CENTERS[k]) / (2SIGMAS[k]^2) for k in eachindex(CENTERS)
    ]
end

function responsibilities(x)
    l = log_components(x)
    m = maximum(l)
    w = exp.(l .- m)
    return w ./ sum(w), m + log(sum(w))
end

logp(x) = last(responsibilities(x))

function gradlogp(x)
    r, _ = responsibilities(x)
    return sum(r[k] .* (CENTERS[k] .- x) ./ SIGMAS[k]^2 for k in eachindex(CENTERS))
end

#= Hessian of a log mixture: Σₖ rₖ (Hₖ + gₖgₖᵀ) - ggᵀ, with Hₖ = -I/σₖ² and gₖ the
component gradients. =#
function hvp(x, v)
    r, _ = responsibilities(x)
    gs = [(CENTERS[k] .- x) ./ SIGMAS[k]^2 for k in eachindex(CENTERS)]
    g = sum(r .* gs)
    hv = sum(r[k] .* (-v ./ SIGMAS[k]^2 .+ gs[k] .* dot(gs[k], v)) for k in eachindex(gs))
    return hv .- g .* dot(g, v)
end

# Chain and DEER iterates

const T = 2_000
const EPSILON = 0.06

rng = MersenneTwister(7)
x0 = [-1.9, 1.5]
tape = [(ξ=randn(rng, 2), u=rand(rng)) for _ in 1:T]

step_fwd(x, s) = MALA.mala_step_taped(logp, gradlogp, x, EPSILON, s.ξ, s.u)
function jvp(x, s, v)
    return MALA.mala_step_surrogate_sigmoid_jvp(
        logp, gradlogp, x, EPSILON, s.ξ, s.u, v, hvp
    )
end
rec = DEER.TapedRecursion(step_fwd, jvp, tape)

sequential = reduce(hcat, accumulate((x, s) -> step_fwd(x, s), tape; init=x0))

function deer_iterates(rec, x0; maxiter=1_000, atol=1e-8)
    S = repeat(x0, 1, T)
    S_new = similar(S)
    ws = DEER.DEERWorkspace(x0, T)
    iterates = [copy(S)]
    for _ in 1:maxiter
        DEER.deer_update!(ws, S_new, rec, x0, S; jacobian=:diag, jacobian_clip=1)
        push!(iterates, copy(S_new))
        maximum(abs, S_new .- S) <= atol && break
        S, S_new = S_new, S
    end
    return iterates
end

iterates = deer_iterates(rec, x0)
error_of(S) = vec(maximum(abs.(S .- sequential); dims=1))
println("DEER converged in ", length(iterates) - 1, " iterations")
@printf("max |DEER - sequential| = %.1e\n", maximum(error_of(last(iterates))))

#= Show every early iteration, where the most happens, then thin the rest so the GIF
stays short. =#
n = length(iterates)
frames = unique(vcat(1:min(n, 30), round.(Int, range(min(n, 30), n; length=40))))

# Figure

const FLOOR = 1e-12
const SEQ = colorant"#b8b8b8"
const PAR = Makie.wong_colors()[1]

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
            ),
        ),
    ),
)

k = Observable(1)
S = @lift iterates[frames[$k]]
err = @lift max.(error_of($S), FLOOR)
title = @lift "DEER iteration $(frames[$k] - 1)"

fig = Figure(; size=(1000, 440), figure_padding=(10, 16, 10, 10))
Label(fig[0, 1], title; fontsize=18, halign=:left, tellwidth=false)

steps = 1:T
pad(v) = (lo=minimum(v); hi=maximum(v); w=hi - lo; (lo - 0.15w, hi + 0.15w))
ax1 = Axis(fig[1, 1]; ylabel=L"x_1", xticklabelsvisible=false)
ax2 = Axis(fig[2, 1]; ylabel=L"x_2", xticklabelsvisible=false)
ax3 = Axis(
    fig[3, 1];
    ylabel="error",
    xlabel="MALA step",
    yscale=log10,
    yticks=([1e-12, 1e-8, 1e-4, 1], [L"10^{-12}", L"10^{-8}", L"10^{-4}", L"10^{0}"]),
)
for (ax, i) in ((ax1, 1), (ax2, 2))
    lines!(ax, steps, sequential[i, :]; color=SEQ, linewidth=2.5)
    lines!(ax, steps, @lift($S[i, :]); color=PAR, linewidth=0.9)
    limits!(ax, 1, T, pad(sequential[i, :])...)
end
lines!(ax3, steps, err; color=PAR, linewidth=0.9)
limits!(ax3, 1, T, FLOOR / 3, 30)
linkxaxes!(ax1, ax2, ax3)
for ax in (ax1, ax2, ax3)
    ax.yticklabelspace = 34.0
end

#= Background: each pixel takes the logo colour of the component that owns it, mixed
into white in proportion to the density. =#
xs = range(-2.2, 2.2; length=360)
ys = range(-1.6, 2.0; length=300)
function tint(x, y)
    r, lp = responsibilities([x, y])
    c = sum(r[j] .* Makie.RGB{Float64}.(LOGO_COLORS)[j] for j in 1:3)
    a = 0.55 * exp(lp - logp(CENTERS[1]))^0.35
    return Makie.RGBf(1 - a + a * c.r, 1 - a + a * c.g, 1 - a + a * c.b)
end
background = [tint(x, y) for x in xs, y in ys]

ax4 = Axis(fig[1:3, 2]; xlabel=L"x_1", ylabel=L"x_2", aspect=DataAspect())
image!(ax4, (xs[1], xs[end]), (ys[1], ys[end]), background)
scatter!(ax4, sequential[1, :], sequential[2, :]; color=SEQ, markersize=5)
scatter!(ax4, @lift($S[1, :]), @lift($S[2, :]); color=PAR, markersize=2.5)
limits!(ax4, -2.2, 2.2, -1.6, 2.0)

Legend(
    fig[0, 2],
    [LineElement(; color=SEQ, linewidth=2.5), LineElement(; color=PAR, linewidth=1.5)],
    ["sequential", "parallel"];
    orientation=:horizontal,
    framevisible=false,
    halign=:right,
    tellwidth=false,
    padding=(0, 0, 0, 0),
)

colsize!(fig.layout, 2, Relative(0.34))
rowgap!(fig.layout, 1, 4)

held = vcat(eachindex(frames), fill(lastindex(frames), 15))
record(fig, OUTPUT, held; framerate=10) do i
    return k[] = i
end
println("wrote ", normpath(OUTPUT))
