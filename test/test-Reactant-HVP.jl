using Test
using Random
using LinearAlgebra
using Statistics
using FlexiChains

using ParallelMCMC
using ADTypes

# A quartic target exposes missing second-order terms that a Gaussian would hide.
logp_r(x) = -0.25 * sum(abs2, x)^2
gradlogp_r(x) = -sum(abs2, x) .* x
hvp_r(x, v) = -(sum(abs2, x) .* v .+ 2 .* dot(x, v) .* x)
logp_batch_r(X) = vec(-0.25 .* sum(abs2, X; dims=1) .^ 2)
gradlogp_batch_r(X) = -X .* sum(abs2, X; dims=1)
hvp_batch_r(X, V) = -(sum(abs2, X; dims=1) .* V .+ 2 .* sum(X .* V; dims=1) .* X)
logp_r32(x) = -0.25f0 * sum(abs2, x)^2
logp_batch_r32(X) = vec(-0.25f0 .* sum(abs2, X; dims=1) .^ 2)

logp_gauss(x) = -0.5 * sum(abs2, x)
gradlogp_gauss(x) = -x
hvp_gauss(x, v) = -v

const D_R = 4
const CT_R = FlexiChains.FlexiChain{Symbol}

# This file only runs when PARALLELMCMC_TEST_REACTANT opts in (see runtests.jl)

using Reactant: Reactant
using Enzyme: Enzyme

isdefined(@__MODULE__, :StagedArray) || include(joinpath(@__DIR__, "staged_array.jl"))

const REACTANT_EXT = Base.get_extension(ParallelMCMC, :ReactantExt)

# "unknown" keeps the load failure reportable as a test below
const REACTANT_PLATFORM =
    REACTANT_EXT === nothing ? "unknown" : REACTANT_EXT._reactant_client_platform()

# `set_default_backend` names the client kind; `platform_name` names the device.
_reactant_backend_name(platform) = platform in ("cuda", "rocm") ? "gpu" : platform

#= `ReactantCall` frees the output buffer before it returns, so a test that needs
to look at that buffer has to run the compiled function itself. Returns the
uploaded inputs too, since they have to be freed alongside the output. =#
function _run_compiled(call, args...)
    inputs = map(REACTANT_EXT._upload, args, call.stages)
    return inputs, call.compiled(inputs...)
end

function _release_compiled!(inputs, out)
    foreach(REACTANT_EXT._free!, inputs)
    REACTANT_EXT._free!(out)
    return nothing
end

#= Float32 logistic regression whose derivatives are all matrix products. TF32
products put the batched derivatives about 3e-4 off the Float64 reference (#83);
full Float32 is near 1e-6. =#
function _test_matmul_precision()
    rng = MersenneTwister(83)
    N, D, T = 512, 16, 64
    A = randn(rng, Float32, N, D) ./ sqrt(Float32(D))
    y = Float32.(rand(rng, N) .< 0.5)
    softplus(z) = max(z, zero(z)) + log1p(exp(-abs(z)))
    logp(b) = sum(y .* (A * b) .- softplus.(A * b)) - 0.5f0 * sum(abs2, b)
    function logp_batch(B)
        Z = A * B
        return vec(sum(y .* Z .- softplus.(Z); dims=1)) .-
               0.5f0 .* vec(sum(abs2, B; dims=1))
    end

    B = 0.3f0 .* randn(rng, Float32, D, T)
    V = randn(rng, Float32, D, T)
    Ad, yd, Bd, Vd = Float64.(A), Float64.(y), Float64.(B), Float64.(V)
    P = inv.(1 .+ exp.(-(Ad * Bd)))
    G_ref = Ad' * (yd .- P) .- Bd
    Hv_ref = -(Ad' * (P .* (1 .- P) .* (Ad * Vd))) .- Vd

    model = DensityModel(
        logp,
        AutoReactant(),
        D;
        logdensity_batch=logp_batch,
        grad_logdensity_batch=AutoReactant(),
        hvp_batch=AutoReactant(),
    )
    m_p = ParallelMCMC._prepare_model(model, B[:, 1], T, nothing)
    relerr(a, b) = maximum(abs, a .- b) / maximum(abs, b)
    @test relerr(m_p.grad_logdensity_batch(B), G_ref) < 1e-5
    @test relerr(m_p.hvp_batch(B, V), Hv_ref) < 1e-5
end

@testset "Reactant HVP" begin
    @testset "extension is loaded" begin
        @test Base.get_extension(ParallelMCMC, :ReactantExt) !== nothing
    end

    @testset "a non-default AutoReactant mode is rejected" begin
        bad = AutoReactant(; mode=AutoEnzyme(; mode=Enzyme.Forward))
        model_grad = DensityModel(logp_r, bad, D_R)
        @test_throws ArgumentError ParallelMCMC._prepare_model(
            model_grad, zeros(D_R), 8, nothing
        )

        model_hvp = DensityModel(logp_r, gradlogp_r, D_R; hvp=bad)
        @test_throws ArgumentError ParallelMCMC._prepare_model(
            model_hvp, zeros(D_R), 8, nothing
        )

        @test DensityModel(logp_r, AutoReactant(), D_R) isa DensityModel
    end

    @testset "HVP matches analytic" begin
        rng = MersenneTwister(71)
        x = randn(rng, D_R)
        v = randn(rng, D_R)

        @testset "forward over user gradient (sampler backend)" begin
            model = DensityModel(logp_r, gradlogp_r, D_R)
            m_p = ParallelMCMC._prepare_model(model, x, 8, AutoReactant())
            @test m_p.hvp(x, v) ≈ hvp_r(x, v)
        end

        @testset "forward-over-reverse from logp alone (both slots AutoReactant)" begin
            model = DensityModel(logp_r, AutoReactant(), D_R; hvp=AutoReactant())
            m_p = ParallelMCMC._prepare_model(model, x, 8, nothing)
            @test m_p.grad_logdensity(x) ≈ gradlogp_r(x)
            @test m_p.hvp(x, v) ≈ hvp_r(x, v)
        end

        @testset "batched slots" begin
            T = 8
            X = randn(rng, D_R, T)
            V = randn(rng, D_R, T)
            Hv_cols = reduce(hcat, [hvp_r(X[:, t], V[:, t]) for t in 1:T])

            model = DensityModel(
                logp_r,
                AutoReactant(),
                D_R;
                logdensity_batch=logp_batch_r,
                grad_logdensity_batch=AutoReactant(),
                hvp=AutoReactant(),
                hvp_batch=AutoReactant(),
            )
            m_p = ParallelMCMC._prepare_model(model, X[:, 1], T, nothing)
            @test m_p.grad_logdensity_batch(X) ≈ gradlogp_batch_r(X)
            @test m_p.hvp_batch(X, V) ≈ Hv_cols
        end

        @testset "traceable value, gradient and HVP evaluator" begin
            T = 8
            X = randn(rng, D_R, T)
            V = randn(rng, D_R, T)

            model = DensityModel(
                logp_r,
                AutoReactant(),
                D_R;
                logdensity_batch=logp_batch_r,
                hvp=AutoReactant(),
            )
            m_p = ParallelMCMC._prepare_model(model, X[:, 1], T, nothing)
            vgh = m_p.value_grad_hvp_batch
            @test vgh isa REACTANT_EXT.ValueGradHVPBatch
            Xr, Vr = Reactant.to_rarray(X), Reactant.to_rarray(V)
            lp, G, Hv = map(Array, Reactant.@compile(vgh(Xr, Vr))(Xr, Vr))
            @test lp ≈ logp_batch_r(X)
            @test G ≈ gradlogp_batch_r(X)
            @test Hv ≈ hvp_batch_r(X, V)

            # Without an AD-derived batched gradient, logdensity_batch is never
            # traced, so it may not be traceable.
            model_cb = DensityModel(
                logp_r,
                gradlogp_r,
                D_R;
                hvp=hvp_r,
                logdensity_batch=logp_batch_r,
                grad_logdensity_batch=gradlogp_batch_r,
                hvp_batch=AutoReactant(),
            )
            m_cb = ParallelMCMC._prepare_model(model_cb, X[:, 1], T, nothing)
            @test m_cb.value_grad_hvp_batch === nothing

            model_hb = DensityModel(
                logp_r,
                AutoReactant(),
                D_R;
                hvp=AutoReactant(),
                logdensity_batch=logp_batch_r,
                hvp_batch=hvp_batch_r,
            )
            m_hb = ParallelMCMC._prepare_model(model_hb, X[:, 1], T, nothing)
            @test m_hb.value_grad_hvp_batch === nothing
        end

        @testset "forward over user batched gradient (hvp_batch=AutoReactant())" begin
            T = 8
            X = randn(rng, D_R, T)
            V = randn(rng, D_R, T)
            Hv_cols = reduce(hcat, [hvp_r(X[:, t], V[:, t]) for t in 1:T])

            model = DensityModel(
                logp_r,
                gradlogp_r,
                D_R;
                hvp=hvp_r,
                logdensity_batch=logp_batch_r,
                grad_logdensity_batch=gradlogp_batch_r,
                hvp_batch=AutoReactant(),
            )
            m_p = ParallelMCMC._prepare_model(model, X[:, 1], T, nothing)
            @test m_p.hvp_batch(X, V) ≈ Hv_cols
        end

        @testset "edge shapes" begin
            @testset "D=1" begin
                x1 = randn(rng, 1)
                v1 = randn(rng, 1)
                model = DensityModel(logp_r, AutoReactant(), 1; hvp=AutoReactant())
                m_p = ParallelMCMC._prepare_model(model, x1, 8, nothing)
                @test m_p.grad_logdensity(x1) ≈ gradlogp_r(x1)
                @test m_p.hvp(x1, v1) ≈ hvp_r(x1, v1)
            end

            @testset "T=1" begin
                T = 1
                X = reshape(x, D_R, T)
                V = reshape(v, D_R, T)
                model = DensityModel(
                    logp_r,
                    AutoReactant(),
                    D_R;
                    logdensity_batch=logp_batch_r,
                    grad_logdensity_batch=AutoReactant(),
                    hvp=AutoReactant(),
                    hvp_batch=AutoReactant(),
                )
                m_p = ParallelMCMC._prepare_model(model, x, T, nothing)
                @test m_p.grad_logdensity_batch(X) ≈ gradlogp_batch_r(X)
                @test m_p.hvp_batch(X, V) ≈ reshape(hvp_r(x, v), D_R, T)
            end
        end

        # DEER passes column views to the vector callables.
        @testset "SubArray inputs" begin
            model = DensityModel(logp_r, AutoReactant(), D_R; hvp=AutoReactant())
            m_p = ParallelMCMC._prepare_model(model, x, 8, nothing)
            S = randn(rng, D_R, 2)
            xv = view(S, :, 1)
            vv = view(S, :, 2)
            @test m_p.grad_logdensity(xv) isa Vector{Float64}
            @test m_p.grad_logdensity(xv) ≈ gradlogp_r(S[:, 1])
            @test m_p.hvp(xv, vv) ≈ hvp_r(S[:, 1], S[:, 2])
        end

        @testset "Float32 on CPU" begin
            x32 = Float32.(x)
            v32 = Float32.(v)
            model = DensityModel(logp_r32, AutoReactant(), D_R; hvp=AutoReactant())
            m_p = ParallelMCMC._prepare_model(model, x32, 8, nothing)

            g = m_p.grad_logdensity(x32)
            @test eltype(g) === Float32
            @test g ≈ gradlogp_r(x32)

            Hv = m_p.hvp(x32, v32)
            @test eltype(Hv) === Float32
            @test Hv ≈ hvp_r(x32, v32)
        end

        @testset "Float32 matrix products at full precision" begin
            _test_matmul_precision()
        end
    end

    @testset "documented caveat: captured data is frozen at preparation time" begin
        data = [1.0, 1.0, 1.0, 1.0]
        f(x) = -0.5 * sum(abs2, x .- data)
        model = DensityModel(f, AutoReactant(), D_R)
        m_p = ParallelMCMC._prepare_model(model, zeros(D_R))

        x0 = zeros(D_R)
        g_before = m_p.grad_logdensity(x0)
        @test g_before ≈ [1.0, 1.0, 1.0, 1.0]

        data .= 5.0

        g_after = m_p.grad_logdensity(x0)
        @test g_after ≈ [1.0, 1.0, 1.0, 1.0]
    end

    # Finite samples do not establish HVP correctness, so test the posterior and
    # compare against an analytic HVP on the same noise tape.
    @testset "end-to-end sampling with AutoReactant" begin
        @testset "posterior mean recovery (standard Gaussian, mean 0)" begin
            model = DensityModel(logp_gauss, AutoReactant(), D_R)
            s = ParallelMALASampler(0.3; T=16, backend=AutoReactant())
            n_samples, n_burn = 2000, 500
            chain = sample(
                MersenneTwister(72), model, s, n_samples; chain_type=CT_R, progress=false
            )
            @test size(chain) == (n_samples, 1)

            xs = chain[:x]
            @test all(x -> all(isfinite, x), xs)
            post_mean = vec(mean(reduce(hcat, xs[(n_burn + 1):end]); dims=2))
            @test maximum(abs, post_mean) < 0.3
        end

        @testset "matches analytic-HVP DEER on the same noise tape" begin
            s = ParallelMALASampler(0.1; T=16, backend=AutoReactant())
            model_r = DensityModel(logp_gauss, AutoReactant(), D_R)
            model_an = DensityModel(logp_gauss, gradlogp_gauss, D_R; hvp=hvp_gauss)

            c_r = sample(
                MersenneTwister(99), model_r, s, 64; chain_type=CT_R, progress=false
            )
            c_an = sample(
                MersenneTwister(99), model_an, s, 64; chain_type=CT_R, progress=false
            )
            @test c_r[:x] ≈ c_an[:x]
        end

        # Probes come from a Reactant RNG, so only the first block matches the host path.
        model_rb = DensityModel(
            logp_r, AutoReactant(), D_R; logdensity_batch=logp_batch_r, hvp=AutoReactant()
        )
        model_an = DensityModel(
            logp_r,
            gradlogp_r,
            D_R;
            hvp=hvp_r,
            logdensity_batch=logp_batch_r,
            grad_logdensity_batch=gradlogp_batch_r,
            hvp_batch=hvp_batch_r,
        )
        x0_b = fill(0.5, D_R)
        first_block(model, s) = sample(
            MersenneTwister(86),
            model,
            s,
            s.T + 1;
            chain_type=CT_R,
            progress=false,
            initial_params=x0_b,
        )[:x]

        @testset "compiled DEER iterations match analytic derivatives on the same tape" begin
            s = ParallelMALASampler(0.05; T=16)
            @test all(
                isapprox.(first_block(model_rb, s), first_block(model_an, s); atol=1e-4)
            )

            L = [1.0 0 0 0; 0.3 0.8 0 0; -0.2 0.1 1.2 0; 0 0.2 0 0.9]
            s_M = ParallelMALASampler(0.05; T=16, cholM=cholesky(Symmetric(L * L')))
            @test all(
                isapprox.(first_block(model_rb, s_M), first_block(model_an, s_M); atol=1e-4)
            )
        end

        @testset "the compiled solver is built once per chain and rebuilt when stale" begin
            rng = MersenneTwister(87)
            s = ParallelMALASampler(0.05; T=8)
            _, st = ParallelMCMC.AbstractMCMC.step(rng, model_rb, s; initial_params=x0_b)
            _, st = ParallelMCMC.AbstractMCMC.step(rng, model_rb, s, st)
            solver = st.workspace
            @test solver isa REACTANT_EXT.ReactantDEERSolver
            for _ in 1:s.T
                _, st = ParallelMCMC.AbstractMCMC.step(rng, model_rb, s, st)
            end
            @test st.t == 1  # a second block was solved
            @test st.workspace === solver

            # Tolerances are not compiled in.
            s_tol = ParallelMALASampler(0.05; T=8, tol_abs=1e-8)
            for _ in 1:s.T
                _, st = ParallelMCMC.AbstractMCMC.step(rng, model_rb, s_tol, st)
            end
            @test st.workspace === solver

            s2 = ParallelMALASampler(0.03; T=8)
            for _ in 1:s.T
                _, st = ParallelMCMC.AbstractMCMC.step(rng, model_rb, s2, st)
            end
            @test st.workspace !== solver
            @test st.workspace.iteration.ε == 0.03
        end

        @testset "Float32 initial point with a Float64 step size" begin
            s = ParallelMALASampler(0.05; T=8)
            chain = sample(
                MersenneTwister(88),
                model_rb,
                s,
                3s.T;
                chain_type=CT_R,
                progress=false,
                initial_params=Float32.(x0_b),
            )
            @test all(x -> all(isfinite, x), chain[:x])
        end
    end

    # `StagedArray` has no `_copy_from_device_pointer!` method, so results come
    # back through a plain host download, as a `CuArray` would on the CPU client.
    @testset "staged template" begin
        rng = MersenneTwister(74)
        x0 = StagedArray(randn(rng, Float32, D_R))
        model = DensityModel(logp_r32, AutoReactant(), D_R; hvp=AutoReactant())
        m_p = ParallelMCMC._prepare_model(model, x0, 8, nothing)

        x1 = StagedArray(randn(rng, Float32, D_R))
        v1 = StagedArray(randn(rng, Float32, D_R))
        x2 = StagedArray(randn(rng, Float32, D_R))
        v2 = StagedArray(randn(rng, Float32, D_R))

        g1 = m_p.grad_logdensity(x1)
        Hv1 = m_p.hvp(x1, v1)
        @test g1 isa StagedArray{Float32,1}
        @test eltype(g1) === Float32
        @test g1.data ≈ gradlogp_r(x1.data)
        @test Hv1 isa StagedArray{Float32,1}
        @test Hv1.data ≈ hvp_r(x1.data, v1.data)

        g2 = m_p.grad_logdensity(x2)
        Hv2 = m_p.hvp(x2, v2)
        @test g2.data ≈ gradlogp_r(x2.data)
        @test Hv2.data ≈ hvp_r(x2.data, v2.data)

        # Results must not alias the reused stage.
        @test g1 !== g2
        @test g1.data !== g2.data
        @test Hv1 !== Hv2
        @test Hv1.data !== Hv2.data

        # Everything above would also pass if no stage had been allocated.
        @test only(m_p.grad_logdensity.stages) isa Vector{Float32}
        @test size(only(m_p.grad_logdensity.stages)) == (D_R,)
        @test all(s -> s isa Vector{Float32}, m_p.hvp.stages)

        # XLA only ever sees the stage, so a short input has to be rejected here.
        @test_throws DimensionMismatch m_p.grad_logdensity(
            StagedArray(randn(rng, Float32, D_R - 1))
        )

        @testset "batched slots" begin
            T = 8
            model_b = DensityModel(
                logp_r32,
                AutoReactant(),
                D_R;
                logdensity_batch=logp_batch_r32,
                grad_logdensity_batch=AutoReactant(),
                hvp=AutoReactant(),
                hvp_batch=AutoReactant(),
            )
            m_pb = ParallelMCMC._prepare_model(model_b, x0, T, nothing)

            X1 = StagedArray(randn(rng, Float32, D_R, T))
            V1 = StagedArray(randn(rng, Float32, D_R, T))

            Gb = m_pb.grad_logdensity_batch(X1)
            @test Gb isa StagedArray{Float32,2}
            @test eltype(Gb) === Float32
            @test Gb.data ≈ gradlogp_batch_r(X1.data)

            Hvb = m_pb.hvp_batch(X1, V1)
            Hv_cols = reduce(hcat, [hvp_r(X1.data[:, t], V1.data[:, t]) for t in 1:T])
            @test Hvb isa StagedArray{Float32,2}
            @test Hvb.data ≈ Hv_cols
        end
    end

    @testset "eager buffer free survives an aliased result" begin
        ext = REACTANT_EXT
        rng = MersenneTwister(75)
        for template in (randn(rng, Float32, D_R), StagedArray(randn(rng, Float32, D_R)))
            call = ext._compiled(identity, template)
            xs = [typeof(template)(randn(rng, Float32, D_R)) for _ in 1:50]
            ys = map(call, xs)
            GC.gc(true)
            @test all(((x, y),) -> collect(y) == collect(x), zip(xs, ys))
        end
    end

    # The free has no observable flag; this checks that a throwing call frees
    # its inputs without a later GC finalizing them a second time.
    @testset "a throwing call still frees its buffers" begin
        ext = REACTANT_EXT
        rng = MersenneTwister(77)
        for template in (randn(rng, Float32, D_R), StagedArray(randn(rng, Float32, D_R)))
            call = ext._compiled(identity, template)
            failing = ext.ReactantCall(
                _ -> error("compiled call failed"),
                call.stages,
                call.template,
                call.platform,
            )
            for _ in 1:20
                @test_throws ErrorException("compiled call failed") failing(template)
            end
            GC.gc(true)
            @test collect(call(template)) == collect(template)
        end
    end

    @testset "a host-resident output declines the pointer path" begin
        x0 = StagedArray(randn(MersenneTwister(76), Float32, D_R))
        model = DensityModel(logp_r32, AutoReactant(), D_R)
        call = ParallelMCMC._prepare_model(model, x0).grad_logdensity
        inputs, out = _run_compiled(call, x0)
        GC.@preserve out begin
            if call.platform == "cpu"
                @test REACTANT_EXT._device_pointer(out) === nothing
                res = REACTANT_EXT._download(call.template, out, call.platform)
                @test res isa StagedArray{Float32,1}
                @test res.data ≈ gradlogp_r(x0.data)
            else
                @info "device pointer guard: not on the CPU client" platform = call.platform
            end
        end
        _release_compiled!(inputs, out)
    end

    reactant_gpu_ok = try
        using CUDA: CUDA
        CUDA.functional() && (CUDA.CuArray([1.0f0]); true)
    catch
        false
    end

    if !reactant_gpu_ok
        @info "Reactant HVP test: CUDA not functional — skipping CuArray boundary"
    else
        # This tests CuArray marshalling, not the XLA execution device.
        @testset "CuArray boundary" begin
            rng = MersenneTwister(73)
            x_h = randn(rng, Float32, D_R)
            v_h = randn(rng, Float32, D_R)
            x_d = CUDA.CuArray(x_h)
            v_d = CUDA.CuArray(v_h)

            model = DensityModel(logp_r32, AutoReactant(), D_R; hvp=AutoReactant())
            m_p = ParallelMCMC._prepare_model(model, x_d, 8, nothing)

            g = m_p.grad_logdensity(x_d)
            @test g isa CUDA.CuArray
            @test Array(g) ≈ gradlogp_r(x_h)

            Hv = m_p.hvp(x_d, v_d)
            @test Hv isa CUDA.CuArray
            @test Array(Hv) ≈ hvp_r(x_h, v_h)

            # Repeated calls must not alias the reused stages.
            x_h2 = randn(rng, Float32, D_R)
            v_h2 = randn(rng, Float32, D_R)
            x_d2 = CUDA.CuArray(x_h2)
            v_d2 = CUDA.CuArray(v_h2)

            g2 = m_p.grad_logdensity(x_d2)
            Hv2 = m_p.hvp(x_d2, v_d2)
            @test g2 isa CUDA.CuArray
            @test Array(g2) ≈ gradlogp_r(x_h2)
            @test Hv2 isa CUDA.CuArray
            @test Array(Hv2) ≈ hvp_r(x_h2, v_h2)

            @test g !== g2
            @test Hv !== Hv2
        end

        gpu_platform = try
            Reactant.set_default_backend("gpu")
            REACTANT_EXT._reactant_client_platform()
        catch err
            @info "Reactant HVP test: selecting a GPU XLA client failed" exception = (
                err, catch_backtrace()
            )
            "unavailable"
        end

        try
            if gpu_platform != "cuda"
                @info "Reactant HVP test: skipping the device-to-device download" platform =
                    gpu_platform
            else
                @testset "device-to-device download (CUDA XLA client)" begin
                    rng = MersenneTwister(76)

                    @testset "vector output" begin
                        x_h = randn(rng, Float32, D_R)
                        v_h = randn(rng, Float32, D_R)
                        x_d = CUDA.CuArray(x_h)
                        v_d = CUDA.CuArray(v_h)

                        model = DensityModel(
                            logp_r32, AutoReactant(), D_R; hvp=AutoReactant()
                        )
                        m_p = ParallelMCMC._prepare_model(model, x_d, 8, nothing)
                        call = m_p.grad_logdensity
                        @test call.platform == "cuda"

                        inputs, out = _run_compiled(call, x_d)
                        GC.@preserve out begin
                            # The branch needs an unsharded PJRT buffer.
                            @test out isa Reactant.ConcretePJRTArray
                            ptr = REACTANT_EXT._device_pointer(out)
                            @test ptr isa Ptr{Cvoid}
                            @test ptr != C_NULL

                            dest = similar(x_d, eltype(out), size(out))
                            @test ParallelMCMC._copy_from_device_pointer!(
                                dest, ptr, call.platform
                            )
                            @test Array(dest) ≈ gradlogp_r(x_h)

                            # The same branch, reached through `_download`.
                            g = REACTANT_EXT._download(call.template, out, call.platform)
                            @test g isa CUDA.CuArray{Float32,1}
                            @test Array(g) ≈ gradlogp_r(x_h)
                        end
                        _release_compiled!(inputs, out)

                        # And end to end, through the call object.
                        g2 = m_p.grad_logdensity(x_d)
                        @test g2 isa CUDA.CuArray{Float32,1}
                        @test Array(g2) ≈ gradlogp_r(x_h)

                        Hv = m_p.hvp(x_d, v_d)
                        @test Hv isa CUDA.CuArray{Float32,1}
                        @test Array(Hv) ≈ hvp_r(x_h, v_h)
                    end

                    @testset "matrix output" begin
                        T = 8
                        X_h = randn(rng, Float32, D_R, T)
                        V_h = randn(rng, Float32, D_R, T)
                        X_d = CUDA.CuArray(X_h)
                        V_d = CUDA.CuArray(V_h)

                        model = DensityModel(
                            logp_r32,
                            AutoReactant(),
                            D_R;
                            logdensity_batch=logp_batch_r32,
                            grad_logdensity_batch=AutoReactant(),
                            hvp=AutoReactant(),
                            hvp_batch=AutoReactant(),
                        )
                        m_p = ParallelMCMC._prepare_model(
                            model, CUDA.CuArray(X_h[:, 1]), T, nothing
                        )
                        call = m_p.grad_logdensity_batch
                        @test call.platform == "cuda"

                        inputs, out = _run_compiled(call, X_d)
                        GC.@preserve out begin
                            @test out isa Reactant.ConcretePJRTArray
                            @test REACTANT_EXT._device_pointer(out) isa Ptr{Cvoid}

                            G = REACTANT_EXT._download(call.template, out, call.platform)
                            @test G isa CUDA.CuArray{Float32,2}
                            @test size(G) == (D_R, T)
                            @test Array(G) ≈ gradlogp_batch_r(X_h)
                        end
                        _release_compiled!(inputs, out)

                        Hvb = m_p.hvp_batch(X_d, V_d)
                        @test Hvb isa CUDA.CuArray{Float32,2}
                        @test size(Hvb) == (D_R, T)
                        @test Array(Hvb) ≈
                            reduce(hcat, [hvp_r(X_h[:, t], V_h[:, t]) for t in 1:T])
                    end

                    # The GPU client is where XLA would otherwise pick TF32.
                    @testset "Float32 matrix products at full precision" begin
                        _test_matmul_precision()
                    end

                    @testset "results outlive the buffers they came from" begin
                        model = DensityModel(logp_r32, AutoReactant(), D_R)
                        m_p = ParallelMCMC._prepare_model(
                            model, CUDA.CuArray(randn(rng, Float32, D_R))
                        )
                        xs = [CUDA.CuArray(randn(rng, Float32, D_R)) for _ in 1:32]
                        gs = map(m_p.grad_logdensity, xs)
                        GC.gc(true)
                        @test all(((x, g),) -> Array(g) ≈ gradlogp_r(Array(x)), zip(xs, gs))
                    end

                    @testset "an unsupported destination falls back to the host" begin
                        x0 = StagedArray(randn(rng, Float32, D_R))
                        model = DensityModel(logp_r32, AutoReactant(), D_R)
                        call = ParallelMCMC._prepare_model(model, x0).grad_logdensity
                        @test call.platform == "cuda"
                        res = call(x0)
                        @test res isa StagedArray{Float32,1}
                        @test res.data ≈ gradlogp_r(x0.data)
                    end
                end
            end
        finally
            REACTANT_PLATFORM == "unknown" ||
                Reactant.set_default_backend(_reactant_backend_name(REACTANT_PLATFORM))
        end
    end
end
