using Test
using Random
using LinearAlgebra
using StatsBase

using ParallelMCMC
const DEER = ParallelMCMC.DEER
const MALA = ParallelMCMC.MALA

logp_stdnormal_B(x) = -0.5 * dot(x, x)
gradlogp_stdnormal_B(x) = -x

# Reference log q(y|x) for MALA:
# y ~ Normal( x + ϵ∇logp(x), 2ϵ I )
function logq_mala_ref_B(y::AbstractVector, x::AbstractVector, gradlogp_x::AbstractVector, ϵ::Real)
    μ = x .+ ϵ .* gradlogp_x
    d = length(x)
    r = y .- μ
    return -0.5 * dot(r, r) / (2ϵ) - (d / 2) * log(4π * ϵ)
end

# Construct a tape for toy affine recursion: b_t vectors
make_affine_tape(rng::AbstractRNG, D::Int, T::Int) = [randn(rng, D) for _ in 1:T]

@testset "Jacobian / surrogate tune-up" begin
    @testset "jac_full matches analytic A on affine recursion" begin
        rng = MersenneTwister(20251231)
        D, T = 6, 5

        # Choose a well-conditioned A with a simple structure
        A = I + 0.1 .* randn(rng, D, D)

        tape = make_affine_tape(rng, D, T)  # b_t

        step_fwd = (x, b) -> A * x + b
        step_lin = step_fwd
        consts = (_x, _b) -> ()
        rec = DEER.TapedRecursion(step_fwd, step_lin, tape; consts=consts, const_example=())

        x0 = randn(rng, D)
        prep = DEER.prepare(rec, x0)

        # Check at a few time points and states
        x = randn(rng, D)
        for t in 1:T
            J = DEER.jac_full(rec, prep, x, t)
            @test size(J) == (D, D)
            @test J ≈ A atol=1e-12 rtol=1e-12
        end
    end

    @testset "jac_diag_stoch approaches diag(jac_full) as probes increase" begin
        rng = MersenneTwister(20251231)
        D, T = 8, 3

        # Make A nontrivial but stable
        A = 0.6I + 0.05 .* randn(rng, D, D)
        tape = make_affine_tape(rng, D, T)

        step_fwd = (x, b) -> A * x + b
        step_lin = step_fwd
        consts = (_x, _b) -> ()
        rec = DEER.TapedRecursion(step_fwd, step_lin, tape; consts=consts, const_example=())

        x0 = randn(rng, D)
        prepJ = DEER.prepare(rec, x0)
        prepPF = DEER.prepare_pushforward(rec, x0)

        x = randn(rng, D)
        t = 1

        J = DEER.jac_full(rec, prepJ, x, t)
        dtrue = diag(J)

        # Compare MSE averaged over multiple independent probe RNG streams
        function mse_for_probes(K::Int; nrep::Int=25)
            mses = zeros(Float64, nrep)
            for r in 1:nrep
                rrng = MersenneTwister(10_000 + 97*r)  # deterministic independent streams
                dest = DEER.jac_diag_stoch(rec, prepPF, x, t; probes=K, rng=rrng)
                mses[r] = mean((dest .- dtrue) .^ 2)
            end
            return mean(mses)
        end

        mse1  = mse_for_probes(1;  nrep=25)
        mse8  = mse_for_probes(8;  nrep=25)
        mse64 = mse_for_probes(64; nrep=25)

        @test mse8  < mse1
        @test mse64 < mse8
    end

    @testset "MALA acceptance gate consistency (forward vs surrogate consts)" begin
        rng = MersenneTwister(20251231)
        D = 5
        ϵ = 0.45

        x = randn(rng, D)
        ξ = randn(rng, D)

        y = MALA.mala_proposal(logp_stdnormal_B, gradlogp_stdnormal_B, x, ϵ, ξ)
        logα = MALA.mala_logα(logp_stdnormal_B, gradlogp_stdnormal_B, x, y, ϵ)

        # Ensure logα < 0 (so τ<1) to force both accept and reject by choosing u.
        tries = 0
        while !(logα < -1e-8) && tries < 2_000
            ξ = randn(rng, D)
            y = MALA.mala_proposal(logp_stdnormal_B, gradlogp_stdnormal_B, x, ϵ, ξ)
            logα = MALA.mala_logα(logp_stdnormal_B, gradlogp_stdnormal_B, x, y, ϵ)
            tries += 1
        end
        @test logα < -1e-8
        τ = exp(logα)
        @test 0.0 < τ < 1.0

        u_acc = max(τ/2, nextfloat(0.0))         # u < τ
        u_rej = min((τ+1.0)/2.0, prevfloat(1.0)) # u > τ

        @test 0.0 < u_acc < τ
        @test τ < u_rej < 1.0

        # Build taped recursion exactly like DEER usage
        tt_acc = (ξ=ξ, u=u_acc)
        tt_rej = (ξ=ξ, u=u_rej)

        step_fwd = (x, tt) -> MALA.mala_step_taped(logp_stdnormal_B, gradlogp_stdnormal_B, x, ϵ, tt.ξ, tt.u)
        step_lin = (x, tt, a) -> MALA.mala_step_surrogate(logp_stdnormal_B, gradlogp_stdnormal_B, x, ϵ, tt.ξ, a)

        consts = (x, tt) -> (MALA.mala_accept_indicator(logp_stdnormal_B, gradlogp_stdnormal_B, x, ϵ, tt.ξ, tt.u),)
        rec = DEER.TapedRecursion(step_fwd, step_lin, [tt_acc, tt_rej]; consts=consts, const_example=(0.0,))

        # consts must match forward accept decision
        a_acc = only(consts(x, tt_acc))
        a_rej = only(consts(x, tt_rej))
        @test a_acc == 1.0
        @test a_rej == 0.0

        # Surrogate semantics: a=1 -> proposal y ; a=0 -> identity x
        @test step_lin(x, tt_acc, 1.0) == y
        @test step_lin(x, tt_acc, 0.0) == x

        # Forward semantics with forced u
        @test step_fwd(x, tt_acc) == y
        @test step_fwd(x, tt_rej) == x
    end


    @testset "DEER solution satisfies recursion defects (MALA taped)" begin
        rng = MersenneTwister(20251231)
        D, T = 6, 80
        ϵ = 0.35

        x0 = randn(rng, D)
        ξs = [randn(rng, D) for _ in 1:T]
        us = rand(rng, T)
        tape = [(ξ=ξs[t], u=us[t]) for t in 1:T]

        step_fwd = (x, tt) -> MALA.mala_step_taped(logp_stdnormal_B, gradlogp_stdnormal_B, x, ϵ, tt.ξ, tt.u)
        step_lin = (x, tt, a) -> MALA.mala_step_surrogate(logp_stdnormal_B, gradlogp_stdnormal_B, x, ϵ, tt.ξ, a)
        consts = (x, tt) -> (MALA.mala_accept_indicator(logp_stdnormal_B, gradlogp_stdnormal_B, x, ϵ, tt.ξ, tt.u),)

        rec = DEER.TapedRecursion(step_fwd, step_lin, tape; consts=consts, const_example=(0.0,))

        S, info = DEER.solve(
            rec,
            x0;
            jacobian=:diag,
            damping=0.5,
            tol_abs=1e-12,
            tol_rel=1e-10,
            maxiter=200,
            return_info=true,
        )

        # Defect check: S[:,t] should equal step_fwd(S_prev, tape[t])
        s_prev = x0
        max_def = 0.0
        for t in 1:T
            s_next = step_fwd(s_prev, tape[t])
            def = norm(S[:, t] .- s_next)
            max_def = max(max_def, def)
            s_prev = S[:, t]
        end

        @test max_def ≤ 1e-9
        @test info.converged
    end
end
