using Test
using Random
using LinearAlgebra

using ParallelMCMC
const MALA = ParallelMCMC.MALA

# Standard normal target in R^d
logp_stdnormal_kernel(x) = -0.5 * dot(x, x)
gradlogp_stdnormal_kernel(x) = -x

# Reference implementation of log q(y|x) for MALA proposal:
# y ~ Normal( x + ϵ∇logp(x), 2ϵ I )
function logq_mala_ref(
    y::AbstractVector, x::AbstractVector, gradlogp_x::AbstractVector, ϵ::Real
)
    μ = x .+ ϵ .* gradlogp_x
    d = length(x)
    r = y .- μ
    return -0.5 * dot(r, r) / (2ϵ) - (d / 2) * log(4π * ϵ)
end

# Build a tape (ξs, us) deterministically
function make_tape(rng::AbstractRNG, D::Int, T::Int)
    ξs = [randn(rng, D) for _ in 1:T]
    us = rand(rng, T)
    return ξs, us
end

@testset "MALA kernel tune-up" begin
    @testset "determinism with fixed tape" begin
        rng = MersenneTwister(20251231)
        D, T = 7, 250
        ϵ = 0.35
        x0 = randn(rng, D)

        # Fixed tape
        ξs, us = make_tape(rng, D, T)

        xs1 = MALA.run_mala_sequential_taped(
            logp_stdnormal, gradlogp_stdnormal, x0, ϵ, ξs, us
        )
        xs2 = MALA.run_mala_sequential_taped(
            logp_stdnormal, gradlogp_stdnormal, x0, ϵ, ξs, us
        )

        @test length(xs1) == T + 1
        @test length(xs2) == T + 1

        # Exact equality should hold because forward recursion is tape-driven
        @test xs1 == xs2

        # Also test step-by-step determinism explicitly
        x = copy(x0)
        for t in 1:T
            x_next_1 = MALA.mala_step_taped(
                logp_stdnormal, gradlogp_stdnormal, x, ϵ, ξs[t], us[t]
            )
            x_next_2 = MALA.mala_step_taped(
                logp_stdnormal, gradlogp_stdnormal, x, ϵ, ξs[t], us[t]
            )
            @test x_next_1 == x_next_2
            x = x_next_1
        end
    end

    @testset "stationary moments on N(0,I)" begin
        rng = MersenneTwister(20251231)
        D = 3
        ϵ = 0.30

        # Keep this CI-friendly. Deterministic via tape.
        T = 25_000
        burn = 5_000

        x0 = randn(rng, D)
        ξs, us = make_tape(rng, D, T)

        xs = MALA.run_mala_sequential_taped(
            logp_stdnormal, gradlogp_stdnormal, x0, ϵ, ξs, us
        )

        # Collect post-burn samples
        X = reduce(hcat, xs[(burn + 1):end])  # D × (T-burn+1)
        N = size(X, 2)

        μ̂ = vec(sum(X; dims=2)) ./ N
        Xc = X .- μ̂
        Σ̂ = (Xc * Xc') ./ (N - 1)

        # Loose-but-informative tolerances for CI stability
        @test maximum(abs.(μ̂)) < 0.08
        @test maximum(abs.(diag(Σ̂) .- 1.0)) < 0.12

        # Off-diagonals should be near zero
        offdiag = Σ̂ .- Diagonal(diag(Σ̂))
        @test maximum(abs.(offdiag)) < 0.10
    end

    @testset "accept-all and reject-all regimes" begin
        rng = MersenneTwister(20251231)
        D = 5
        ϵ = 0.45
        x = randn(rng, D)

        # Find any case with logα < 0 so acceptance probability τ=exp(logα) is strictly < 1
        ξ = randn(rng, D)
        y = MALA.mala_proposal(logp_stdnormal_kernel, gradlogp_stdnormal_kernel, x, ϵ, ξ)
        logα = MALA.mala_logα(logp_stdnormal_kernel, gradlogp_stdnormal_kernel, x, y, ϵ)

        tries = 0
        while !(logα < -1e-8) && tries < 2_000
            ξ = randn(rng, D)
            y = MALA.mala_proposal(
                logp_stdnormal_kernel, gradlogp_stdnormal_kernel, x, ϵ, ξ
            )
            logα = MALA.mala_logα(logp_stdnormal_kernel, gradlogp_stdnormal_kernel, x, y, ϵ)
            tries += 1
        end
        @test logα < -1e-8

        τ = exp(logα)  # acceptance probability since logα < 0
        @test 0.0 < τ < 1.0

        # Force ACCEPT: choose u < τ
        u_accept = τ / 2
        u_accept = max(u_accept, nextfloat(0.0))  # strictly > 0
        @test 0.0 < u_accept < τ
        x_acc = MALA.mala_step_taped(
            logp_stdnormal_kernel, gradlogp_stdnormal_kernel, x, ϵ, ξ, u_accept
        )
        @test x_acc == y

        # Force REJECT: choose u > τ (but still < 1)
        u_reject = (τ + 1.0) / 2.0
        u_reject = min(u_reject, prevfloat(1.0))
        @test τ < u_reject < 1.0
        x_rej = MALA.mala_step_taped(
            logp_stdnormal_kernel, gradlogp_stdnormal_kernel, x, ϵ, ξ, u_reject
        )
        @test x_rej == x

        # Also sanity-check the accept indicator
        a_acc = MALA.mala_accept_indicator(
            logp_stdnormal_kernel, gradlogp_stdnormal_kernel, x, ϵ, ξ, u_accept
        )
        a_rej = MALA.mala_accept_indicator(
            logp_stdnormal_kernel, gradlogp_stdnormal_kernel, x, ϵ, ξ, u_reject
        )
        @test a_acc == 1.0
        @test a_rej == 0.0
    end

    @testset "logα sanity vs independent logq formula" begin
        rng = MersenneTwister(20251231)
        D = 6
        ϵ = 0.40

        x = randn(rng, D)
        ξ = randn(rng, D)

        y = MALA.mala_proposal(logp_stdnormal, gradlogp_stdnormal, x, ϵ, ξ)

        logα_impl = MALA.mala_logα(logp_stdnormal, gradlogp_stdnormal, x, y, ϵ)

        # Independent recomputation:
        gx = gradlogp_stdnormal(x)
        gy = gradlogp_stdnormal(y)

        logq_y_given_x = logq_mala_ref(y, x, gx, ϵ)
        logq_x_given_y = logq_mala_ref(x, y, gy, ϵ)

        logα_ref =
            (logp_stdnormal(y) + logq_x_given_y) - (logp_stdnormal(x) + logq_y_given_x)

        @test isfinite(logα_impl)
        @test isfinite(logα_ref)
        @test logα_impl ≈ logα_ref atol=1e-10 rtol=1e-10
    end
end
