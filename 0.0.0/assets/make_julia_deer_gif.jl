#!/usr/bin/env julia

# Regenerate `julia_deer_posterior.gif` for the README and docs landing page.
#
# This script intentionally includes the local MALA and affine-scan primitives
# instead of `using ParallelMCMC`, so the docs asset can be generated without
# downloading GPU/AD artifacts.

using LinearAlgebra
using Printf
using Random

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPO_ROOT, "src", "MALA", "MALA.jl"))
include(joinpath(REPO_ROOT, "src", "DEER", "DEERScan.jl"))

const WIDTH = 560
const HEIGHT = 420
const X_RANGE = (-2.25, 2.25)
const Y_RANGE = (-1.70, 1.78)

const CENTERS = [
    (-1.05, -0.55), # green
    (1.05, -0.55),  # red
    (0.0, 1.02),    # purple
    (0.0, -0.16),   # blue
]
const SIGMAS = [0.38, 0.38, 0.38, 0.25]
const LOG_WEIGHTS = log.([1.0, 1.0, 1.0, 0.50])
const LOGO_COLORS = [
    (56, 152, 38),
    (203, 60, 51),
    (149, 88, 178),
    (64, 99, 216),
]

struct TapeStep
    xi::Vector{Float64}
    u::Float64
end

function log_components(x)
    vals = Vector{Float64}(undef, length(CENTERS))
    for k in eachindex(CENTERS)
        cx, cy = CENTERS[k]
        sig2 = SIGMAS[k]^2
        dx = x[1] - cx
        dy = x[2] - cy
        vals[k] = LOG_WEIGHTS[k] - log(2 * pi * sig2) - 0.5 * (dx^2 + dy^2) / sig2
    end
    return vals
end

function responsibilities_from_logs(vals)
    offset = maximum(vals)
    weights = exp.(vals .- offset)
    total = sum(weights)
    return weights ./ total, offset + log(total)
end

function logposterior(x)
    _, lp = responsibilities_from_logs(log_components(x))
    return lp
end

function gradposterior(x)
    resp, _ = responsibilities_from_logs(log_components(x))
    grad = zeros(2)
    for k in eachindex(CENTERS)
        cx, cy = CENTERS[k]
        sig2 = SIGMAS[k]^2
        grad[1] += resp[k] * (cx - x[1]) / sig2
        grad[2] += resp[k] * (cy - x[2]) / sig2
    end
    return grad
end

function hvp_posterior(x, v)
    resp, _ = responsibilities_from_logs(log_components(x))
    component_grads = Matrix{Float64}(undef, 2, length(CENTERS))
    grad = zeros(2)
    hv = zeros(2)

    for k in eachindex(CENTERS)
        cx, cy = CENTERS[k]
        sig2 = SIGMAS[k]^2
        g1 = (cx - x[1]) / sig2
        g2 = (cy - x[2]) / sig2
        component_grads[1, k] = g1
        component_grads[2, k] = g2
        grad[1] += resp[k] * g1
        grad[2] += resp[k] * g2
        hv[1] -= resp[k] * v[1] / sig2
        hv[2] -= resp[k] * v[2] / sig2
    end

    for k in eachindex(CENTERS)
        dg1 = component_grads[1, k] - grad[1]
        dg2 = component_grads[2, k] - grad[2]
        projection = dg1 * v[1] + dg2 * v[2]
        hv[1] += resp[k] * dg1 * projection
        hv[2] += resp[k] * dg2 * projection
    end

    return hv
end

function make_tape(rng, dim, steps)
    return [TapeStep(randn(rng, dim), rand(rng)) for _ in 1:steps]
end

function make_recursion(tape, epsilon)
    step_fwd =
        (x, step) ->
            MALA.mala_step_taped(logposterior, gradposterior, x, epsilon, step.xi, step.u)
    jvp =
        (x, step, v) -> MALA.mala_step_surrogate_sigmoid_jvp(
            logposterior, gradposterior, x, epsilon, step.xi, step.u, v, hvp_posterior
        )
    return (; step_fwd, jvp, tape)
end

function deer_diag_update!(
    output, A, B, scan_ws, rec, s0, current; damping=0.55
)
    dim, steps = size(current)
    basis = zeros(dim)

    for t in 1:steps
        xbar = t == 1 ? s0 : view(current, :, t - 1)
        ft = rec.step_fwd(xbar, rec.tape[t])

        for j in 1:dim
            fill!(basis, 0.0)
            basis[j] = 1.0
            jv = rec.jvp(xbar, rec.tape[t], basis)
            A[j, t] = jv[j]
        end

        @views B[:, t] .= ft .- A[:, t] .* xbar
    end

    DEERScan.solve_affine_scan_diag!(output, A, B, s0, scan_ws)
    @. output = (1 - damping) * current + damping * output
    return output
end

function record_iterates(rec, s0; steps, maxiter=24, damping=0.55)
    dim = length(s0)
    current = repeat(reshape(s0, dim, 1), 1, steps)
    next_state = similar(current)
    A = similar(current)
    B = similar(current)
    scan_ws = DEERScan.AffineScanWorkspace(A)
    iterates = [copy(current)]
    metrics = Float64[]

    for _ in 1:maxiter
        deer_diag_update!(next_state, A, B, scan_ws, rec, s0, current; damping=damping)
        delta = maximum(abs.(next_state .- current))
        scale = 1e-5 + 1e-4 * maximum(abs.(next_state))
        push!(metrics, delta / scale)
        push!(iterates, copy(next_state))
        current, next_state = next_state, current
    end

    return iterates, metrics
end

clamp_u8(x) = UInt8(clamp(round(Int, x), 0, 255))

function blend_pixel!(image, px, py, color, alpha)
    1 <= px <= WIDTH || return image
    1 <= py <= HEIGHT || return image

    base = image[py, px]
    image[py, px] = (
        clamp_u8((1 - alpha) * base[1] + alpha * color[1]),
        clamp_u8((1 - alpha) * base[2] + alpha * color[2]),
        clamp_u8((1 - alpha) * base[3] + alpha * color[3]),
    )
    return image
end

function data_to_pixel(x, y)
    px = round(Int, 1 + (x - X_RANGE[1]) / (X_RANGE[2] - X_RANGE[1]) * (WIDTH - 1))
    py = round(Int, 1 + (Y_RANGE[2] - y) / (Y_RANGE[2] - Y_RANGE[1]) * (HEIGHT - 1))
    return px, py
end

function pixel_to_data(px, py)
    x = X_RANGE[1] + (px - 1) / (WIDTH - 1) * (X_RANGE[2] - X_RANGE[1])
    y = Y_RANGE[2] - (py - 1) / (HEIGHT - 1) * (Y_RANGE[2] - Y_RANGE[1])
    return [x, y]
end

function weighted_logo_color(resp)
    r = g = b = 0.0
    for k in eachindex(LOGO_COLORS)
        color = LOGO_COLORS[k]
        r += resp[k] * color[1]
        g += resp[k] * color[2]
        b += resp[k] * color[3]
    end
    return (r, g, b)
end

function make_background()
    image = fill((UInt8(252), UInt8(252), UInt8(250)), HEIGHT, WIDTH)
    logps = Matrix{Float64}(undef, HEIGHT, WIDTH)
    maxlogp = -Inf

    for py in 1:HEIGHT, px in 1:WIDTH
        _, lp = responsibilities_from_logs(log_components(pixel_to_data(px, py)))
        logps[py, px] = lp
        maxlogp = max(maxlogp, lp)
    end

    for py in 1:HEIGHT, px in 1:WIDTH
        x = pixel_to_data(px, py)
        resp, _ = responsibilities_from_logs(log_components(x))
        intensity = exp(logps[py, px] - maxlogp)^0.38
        alpha = 0.08 + 0.72 * intensity
        blend_pixel!(image, px, py, weighted_logo_color(resp), alpha)
    end

    return image
end

function draw_disk!(image, cx, cy, radius, color, alpha)
    xmin = floor(Int, cx - radius - 1)
    xmax = ceil(Int, cx + radius + 1)
    ymin = floor(Int, cy - radius - 1)
    ymax = ceil(Int, cy + radius + 1)

    for py in ymin:ymax, px in xmin:xmax
        dist = hypot(px - cx, py - cy)
        if dist <= radius + 0.5
            edge = clamp(radius + 0.5 - dist, 0.0, 1.0)
            blend_pixel!(image, px, py, color, alpha * edge)
        end
    end

    return image
end

function draw_line!(image, p1, p2, radius, color, alpha)
    dx = p2[1] - p1[1]
    dy = p2[2] - p1[2]
    steps = max(1, ceil(Int, 1.7 * max(abs(dx), abs(dy))))

    for i in 0:steps
        t = i / steps
        px = p1[1] + t * dx
        py = p1[2] + t * dy
        draw_disk!(image, px, py, radius, color, alpha)
    end

    return image
end

function draw_trajectory!(image, trajectory; ghost=false)
    steps = size(trajectory, 2)
    points = Vector{Tuple{Int,Int}}(undef, steps)

    for t in 1:steps
        points[t] = data_to_pixel(trajectory[1, t], trajectory[2, t])
    end

    if ghost
        for t in 2:steps
            draw_line!(image, points[t - 1], points[t], 0.9, (25, 31, 40), 0.13)
        end
        for t in 1:3:steps
            draw_disk!(image, points[t][1], points[t][2], 1.45, (25, 31, 40), 0.18)
        end
    else
        for t in 2:steps
            draw_line!(image, points[t - 1], points[t], 1.15, (28, 36, 50), 0.48)
        end
        for t in 1:steps
            draw_disk!(image, points[t][1], points[t][2], 3.0, (28, 36, 50), 0.42)
            draw_disk!(image, points[t][1], points[t][2], 2.05, (247, 166, 38), 0.88)
        end
        draw_disk!(image, points[1][1], points[1][2], 4.2, (255, 255, 255), 0.82)
        draw_disk!(image, points[1][1], points[1][2], 3.0, (56, 152, 38), 0.95)
        draw_disk!(image, points[end][1], points[end][2], 4.2, (255, 255, 255), 0.82)
        draw_disk!(image, points[end][1], points[end][2], 3.0, (203, 60, 51), 0.95)
    end

    return image
end

function draw_rect!(image, x1, y1, x2, y2, color, alpha)
    for py in max(1, y1):min(HEIGHT, y2), px in max(1, x1):min(WIDTH, x2)
        blend_pixel!(image, px, py, color, alpha)
    end
    return image
end

function draw_progress!(image, frame_index, frame_count)
    margin = 44
    y = HEIGHT - 22
    width = WIDTH - 2margin
    filled = round(Int, width * (frame_index - 1) / max(1, frame_count - 1))

    draw_rect!(image, margin, y, margin + width, y + 6, (24, 30, 39), 0.16)
    draw_rect!(image, margin, y, margin + filled, y + 6, (247, 166, 38), 0.90)
    draw_disk!(image, margin + filled, y + 3, 4.0, (247, 166, 38), 0.95)
    return image
end

function write_ppm(path, image)
    open(path, "w") do io
        write(io, "P6\n$WIDTH $HEIGHT\n255\n")
        for py in 1:HEIGHT, px in 1:WIDTH
            color = image[py, px]
            write(io, color[1], color[2], color[3])
        end
    end
    return path
end

function render_frames(frame_dir, iterates, final_trajectory)
    background = make_background()
    frame_paths = String[]

    for (i, trajectory) in enumerate(iterates)
        image = copy(background)
        draw_trajectory!(image, final_trajectory; ghost=true)
        draw_trajectory!(image, trajectory)
        draw_progress!(image, i, length(iterates))

        path = joinpath(frame_dir, @sprintf("frame_%03d.ppm", i))
        write_ppm(path, image)
        push!(frame_paths, path)
    end

    return frame_paths
end

function main()
    rng = MersenneTwister(20260428)
    steps = 1000
    maxiter = 256
    epsilon = 0.095
    damping = 0.55
    x0 = [
    rand(rng) * (X_RANGE[2] - X_RANGE[1]) + X_RANGE[1],
    rand(rng) * (Y_RANGE[2] - Y_RANGE[1]) + Y_RANGE[1],
    ]

    
    tape = make_tape(rng, 2, steps)
    rec = make_recursion(tape, epsilon)
    iterates, metrics = record_iterates(rec, x0; steps=steps, maxiter=maxiter, damping=damping)

    selected = unique(round.(Int, range(0, maxiter; length=256)))
    selected_iterates = [iterates[i + 1] for i in selected]

    noise = [step.xi for step in tape]
    uniforms = [step.u for step in tape]
    sequential = MALA.run_mala_sequential_taped(logposterior, gradposterior, x0, epsilon, noise, uniforms)
    final_trajectory = reduce(hcat, sequential[2:end])

    output = isempty(ARGS) ? joinpath(@__DIR__, "julia_deer_posterior.gif") : ARGS[1]
    mkpath(dirname(output))

    convert = Sys.which("convert")
    convert === nothing && error("ImageMagick `convert` is required to build the GIF")

    mktempdir() do frame_dir
        frame_paths = render_frames(frame_dir, selected_iterates, final_trajectory)
        animation_paths = vcat(frame_paths, fill(last(frame_paths), 8))
        tmp_output = joinpath(frame_dir, "julia_deer_posterior.gif")
        run(`$convert -delay 7 -loop 0 $animation_paths -layers Optimize $tmp_output`)
        cp(tmp_output, output; force=true)
    end

    final_error = maximum(abs.(last(iterates) .- final_trajectory))
    println("wrote ", output)
    println("last DEER metric: ", @sprintf("%.3g", last(metrics)))
    println("max error vs sequential taped MALA: ", @sprintf("%.3g", final_error))
end

main()
