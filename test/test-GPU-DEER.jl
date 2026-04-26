using Test
using Random
using LinearAlgebra
using Statistics

using ParallelMCMC
import ParallelMCMC: DEER, MALA
using ADTypes: ADTypes
using ForwardDiff: ForwardDiff

cuda_ok = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !cuda_ok
    @warn "CUDA not available or not functional — skipping GPU DEER tests"
else

    # Standard normal target — simple element-wise ops, compatible with CuArray{Dual}
    _logp(x) = -0.5f0 * sum(abs2, x)
    _gradlogp(x) = -x

    #=
    Collect GPU-compatible backends to test.
    DifferentiationInterface abstracts over backends; any backend whose
    pushforward works on CuArrays (i.e. uses broadcast rather than scalar
    indexing) is valid.
    =#
    _gpu_backends = Pair{String,ADTypes.AbstractADType}[]
    if !Sys.iswindows()
        try
            using Enzyme: Enzyme
            push!(_gpu_backends, "Enzyme" => ADTypes.AutoEnzyme())
        catch
            @warn "Enzyme not loadable — skipping Enzyme GPU DEER tests"
        end
    end

    # Convenience alias used by tests that don't need to sweep over all backends.
    _gpu_backend = last(first(_gpu_backends))

    @testset "solve_affine_scan_diag CPU vs GPU" begin
        rng = MersenneTwister(1)
        D, T = 4, 16

        A_cpu = randn(rng, Float32, D, T)
        B_cpu = randn(rng, Float32, D, T)
        s0_cpu = randn(rng, Float32, D)

        S_cpu = DEER.solve_affine_scan_diag(A_cpu, B_cpu, s0_cpu)

        A_gpu = CUDA.CuArray(A_cpu)
        B_gpu = CUDA.CuArray(B_cpu)
        s0_gpu = CUDA.CuArray(s0_cpu)

        S_gpu = DEER.solve_affine_scan_diag(A_gpu, B_gpu, s0_gpu)

        @test S_gpu isa CUDA.CuMatrix
        @test size(S_gpu) == (D, T)
        @test Array(S_gpu) ≈ S_cpu atol=1e-5
    end

    #=
    Backend-parametric convergence test: run DEER.solve on GPU with each
    available backend and verify the solution matches the sequential CPU
    reference.  This is the primary test that the AD abstraction layer
    (DifferentiationInterface) is backend-agnostic.
    =#
    function _make_gpu_rec(tape, ε, backend)
        step_fwd = (x, te) -> MALA.mala_step_taped(_logp, _gradlogp, x, ε, te.ξ, te.u)
        hvp_fn = (pt, dir) -> DEER._hvp_nopre(_gradlogp, backend, pt, dir)
        jvp = (x, te, v) -> MALA.mala_step_surrogate_sigmoid_jvp(
            _logp, _gradlogp, x, ε, te.ξ, te.u, v, hvp_fn
        )
        fwd_and_jvp = (x, te, v) -> MALA.mala_step_taped_and_jvp(
            _logp, _gradlogp, x, ε, te.ξ, te.u, v, hvp_fn
        )
        return DEER.TapedRecursion(step_fwd, jvp, tape; fwd_and_jvp=fwd_and_jvp)
    end

    @testset "DEER.solve GPU backend=$bname (stoch_diag)" for (bname, backend) in _gpu_backends
        rng = MersenneTwister(7)
        D, T = 4, 32
        ε = 0.05f0

        ξs_cpu = [randn(rng, Float32, D) for _ in 1:T]
        us = Float32.(rand(rng, T))
        x0_cpu = randn(rng, Float32, D)

        xs_seq = MALA.run_mala_sequential_taped(_logp, _gradlogp, x0_cpu, ε, ξs_cpu, us)

        tape = [(ξ=CUDA.CuArray(ξs_cpu[t]), u=us[t]) for t in 1:T]
        rec = _make_gpu_rec(tape, ε, backend)

        x0_gpu = CUDA.CuArray(x0_cpu)
        S_gpu, info = DEER.solve(
            rec,
            x0_gpu;
            jacobian=:stoch_diag,
            probes=2,
            rng=MersenneTwister(123),
            damping=0.5,
            tol_abs=1e-5,
            tol_rel=1e-4,
            maxiter=400,
            return_info=true,
        )

        @test info.converged
        @test S_gpu isa CUDA.CuMatrix
        @test size(S_gpu) == (D, T)
        @test all(isfinite, Array(S_gpu))

        S_ref = reduce(hcat, xs_seq[2:end])
        @test Array(S_gpu) ≈ S_ref rtol=1e-3 atol=1e-4
    end

    @testset "ParallelMALASampler initial step on GPU" begin
        rng = MersenneTwister(7)
        D = 3
        model = DensityModel(_logp, _gradlogp, D)
        sampler = ParallelMALASampler(
            0.05f0; T=8, maxiter=200, jacobian=:diag, backend=_gpu_backend
        )
        x0_gpu = CUDA.CuArray(randn(rng, Float32, D))

        trans, state = ParallelMCMC.AbstractMCMC.step(
            rng, model, sampler; initial_params=x0_gpu
        )

        @test trans isa ParallelMALATransition
        @test state isa ParallelMALAState
        @test trans.x isa CUDA.CuVector
        @test length(trans.x) == D
        @test isfinite(Float32(trans.logp))
        @test size(state.trajectory) == (D, 8)
        @test state.trajectory isa CUDA.CuMatrix
        @test length(state.logps) == 8
        @test state.workspace isa DEER.DEERWorkspace
        @test state.t == 1
    end

    @testset "ParallelMALASampler sequential steps advance trajectory index on GPU" begin
        rng = MersenneTwister(42)
        D = 3
        model = DensityModel(_logp, _gradlogp, D)
        sampler = ParallelMALASampler(
            0.05f0; T=8, maxiter=200, jacobian=:diag, backend=_gpu_backend
        )
        x0_gpu = CUDA.CuArray(randn(rng, Float32, D))

        trans, state = ParallelMCMC.AbstractMCMC.step(
            rng, model, sampler; initial_params=x0_gpu
        )
        @test state.t == 1

        trans2, state2 = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
        @test state2.t == 2
        @test trans2.x isa CUDA.CuVector
        @test Array(trans2.x) ≈ Array(state.trajectory[:, 2])

        trans3, state3 = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state2)
        @test state3.t == 3
    end

    @testset "ParallelMALASampler re-solves at trajectory boundary on GPU" begin
        rng = MersenneTwister(99)
        D, T = 3, 4
        model = DensityModel(_logp, _gradlogp, D)
        sampler = ParallelMALASampler(
            0.05f0; T=T, maxiter=200, jacobian=:diag, backend=_gpu_backend
        )
        x0_gpu = CUDA.CuArray(randn(rng, Float32, D))

        _, state = ParallelMCMC.AbstractMCMC.step(
            rng, model, sampler; initial_params=x0_gpu
        )
        for _ in 1:(T - 1)
            _, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
        end
        @test state.t == T

        # t == T triggers a re-solve; t resets to 1 and reuses workspace
        _, state_new = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
        @test state_new.t == 1
        @test state_new.trajectory isa CUDA.CuMatrix
        @test state_new.workspace === state.workspace
        @test length(state_new.logps) == T
    end

    @testset "ParallelMALASampler stationary distribution on GPU (standard normal)" begin
        D = 3
        model = DensityModel(_logp, _gradlogp, D)
        # stoch_diag: one JVP per probe — cheaper and GPU-friendly
        sampler = ParallelMALASampler(
            0.1f0;
            T=32,
            maxiter=100,
            jacobian=:stoch_diag,
            probes=1,
            damping=0.5,
            backend=_gpu_backend,
        )

        rng = MersenneTwister(2025)
        # Collect raw ParallelMALATransitions to avoid MCMCChains GPU array copy issues
        raw = sample(
            rng,
            model,
            sampler,
            3_000;
            initial_params=CUDA.CuArray(zeros(Float32, D)),
            progress=false,
        )

        xs = reduce(hcat, [Array(s.x) for s in raw])   # D×3000

        burn = 300
        post = xs[:, burn:end]
        mu = vec(mean(post; dims=2))
        vars = vec(var(post; dims=2))

        @test maximum(abs.(mu)) < 0.2
        @test maximum(abs.(vars .- 1.0)) < 0.3
    end
end # cuda_ok
