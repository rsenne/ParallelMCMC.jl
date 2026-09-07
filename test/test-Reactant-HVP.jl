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
logp_r32(x) = -0.25f0 * sum(abs2, x)^2

logp_gauss(x) = -0.5 * sum(abs2, x)
gradlogp_gauss(x) = -x
hvp_gauss(x, v) = -v

const D_R = 4
const CT_R = FlexiChains.FlexiChain{Symbol}

# This file only runs when PARALLELMCMC_TEST_REACTANT opts in (see runtests.jl)

using Reactant: Reactant
using Enzyme: Enzyme

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
        end
    end
end
