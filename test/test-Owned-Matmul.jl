using Test
using Random
using LinearAlgebra

using ParallelMCMC
using ADTypes
using Enzyme
import ParallelMCMC.DEER as DEER
const DI = DEER.DI

#=
Tests for the owned wrappers `pmcmc_matmul`, `pmcmc_dot`, `pmcmc_dotsum` and
the native Enzyme rules in `ext/EnzymeExt.jl`. We check:
  1. function values match the Base counterparts
  2. forward-mode JVPs match the analytical formulas
  3. reverse-mode pullbacks (gradients) match the analytical formulas

The reverse cases are critical: they're what makes the GPU AD-HVP path work
without crashing on `cuMemcpyDtoHAsync_v2` gc-transition bundles.
=#

const _AD_FWD = ADTypes.AutoEnzyme(; mode=Enzyme.Forward)
const _AD_REV = ADTypes.AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const
)

# ============================================================
# pmcmc_matmul
# ============================================================

@testset "pmcmc_matmul value matches Base.*" begin
    rng = MersenneTwister(0)
    A = randn(rng, 4, 3)
    B = randn(rng, 3, 5)
    b = randn(rng, 3)

    @test pmcmc_matmul(A, B) ≈ A * B
    @test pmcmc_matmul(A, b) ≈ A * b
end

@testset "Enzyme forward JVP on pmcmc_matmul matches analytical" begin
    rng = MersenneTwister(1)
    M, K, N = 5, 4, 3
    A = randn(rng, M, K)
    B = randn(rng, K, N)
    dA = randn(rng, M, K)
    dB = randn(rng, K, N)

    dY_expected = dA * B + A * dB

    f1 = ((A_, B_),) -> pmcmc_matmul(A_, B_)
    prep = DI.prepare_pushforward(f1, _AD_FWD, (A, B), ((dA, dB),); strict=Val(false))
    dY = first(DI.pushforward(f1, prep, _AD_FWD, (A, B), ((dA, dB),)))

    @test dY ≈ dY_expected rtol=1e-12 atol=1e-12
end

@testset "Enzyme forward JVP on pmcmc_matmul: matrix-vector" begin
    rng = MersenneTwister(2)
    M, K = 6, 4
    A = randn(rng, M, K)
    b = randn(rng, K)
    dA = randn(rng, M, K)
    db = randn(rng, K)

    dy_expected = dA * b + A * db

    f1 = ((A_, b_),) -> pmcmc_matmul(A_, b_)
    prep = DI.prepare_pushforward(f1, _AD_FWD, (A, b), ((db, db),); strict=Val(false))  # eltype-stable
    prep = DI.prepare_pushforward(f1, _AD_FWD, (A, b), ((dA, db),); strict=Val(false))
    dy = first(DI.pushforward(f1, prep, _AD_FWD, (A, b), ((dA, db),)))

    @test dy ≈ dy_expected rtol=1e-12 atol=1e-12
end

@testset "Enzyme reverse pullback on pmcmc_matmul (via grad of dot)" begin
    # f(A, B) = dot(pmcmc_matmul(A, B), w)  is scalar; dA = w * B', dB = A' * w
    rng = MersenneTwister(3)
    M, K = 5, 4
    A = randn(rng, M, K)
    b = randn(rng, K)
    w = randn(rng, M)

    f = ((A_, b_),) -> pmcmc_dot(pmcmc_matmul(A_, b_), w)
    prep = DI.prepare_gradient(f, _AD_REV, (A, b); strict=Val(false))
    g = DI.gradient(f, prep, _AD_REV, (A, b))

    dA_expected = w * b'
    db_expected = A' * w
    @test g[1] ≈ dA_expected rtol=1e-12 atol=1e-12
    @test g[2] ≈ db_expected rtol=1e-12 atol=1e-12
end

# ============================================================
# pmcmc_dot
# ============================================================

@testset "pmcmc_dot value matches LinearAlgebra.dot" begin
    rng = MersenneTwister(10)
    a = randn(rng, 7)
    b = randn(rng, 7)
    @test pmcmc_dot(a, b) ≈ dot(a, b)
end

@testset "Enzyme reverse pullback on pmcmc_dot" begin
    # f(a, b) = pmcmc_dot(a, b);  da = b, db = a
    rng = MersenneTwister(11)
    a = randn(rng, 6)
    b = randn(rng, 6)

    f = ((a_, b_),) -> pmcmc_dot(a_, b_)
    prep = DI.prepare_gradient(f, _AD_REV, (a, b); strict=Val(false))
    g = DI.gradient(f, prep, _AD_REV, (a, b))

    @test g[1] ≈ b rtol=1e-12 atol=1e-12
    @test g[2] ≈ a rtol=1e-12 atol=1e-12
end

# ============================================================
# pmcmc_dotsum
# ============================================================

@testset "pmcmc_dotsum value matches sum(A .* B)" begin
    rng = MersenneTwister(20)
    A = randn(rng, 3, 4)
    B = randn(rng, 3, 4)
    @test pmcmc_dotsum(A, B) ≈ sum(A .* B)
end

@testset "Enzyme reverse pullback on pmcmc_dotsum" begin
    # f(A, B) = pmcmc_dotsum(A, B);  dA = B, dB = A
    rng = MersenneTwister(21)
    A = randn(rng, 4, 3)
    B = randn(rng, 4, 3)

    f = ((A_, B_),) -> pmcmc_dotsum(A_, B_)
    prep = DI.prepare_gradient(f, _AD_REV, (A, B); strict=Val(false))
    g = DI.gradient(f, prep, _AD_REV, (A, B))

    @test g[1] ≈ B rtol=1e-12 atol=1e-12
    @test g[2] ≈ A rtol=1e-12 atol=1e-12
end
