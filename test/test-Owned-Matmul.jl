using Test
using Random
using LinearAlgebra

using ParallelMCMC
using ADTypes
using Enzyme
import ParallelMCMC.DEER as DEER
const DI = DEER.DI

#=
Tests for the owned wrapper `pmcmc_matmul` and its native Enzyme rules in
`ext/EnzymeExt.jl`. We check:
  1. function values match `Base.*`
  2. forward-mode JVPs match the analytical formulas
  3. reverse-mode pullbacks (gradients) match the analytical formulas

The forward cases are why the rules exist: forward-mode Enzyme cannot
differentiate `*` on CuArrays (see `ext/EnzymeExt.jl`).
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
    prep = DI.prepare_pushforward(f1, _AD_FWD, (A, B), ((dA, dB),))
    dY = first(DI.pushforward(f1, prep, _AD_FWD, (A, B), ((dA, dB),)))

    @test dY ≈ dY_expected rtol = 1e-12 atol = 1e-12
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
    prep = DI.prepare_pushforward(f1, _AD_FWD, (A, b), ((dA, db),))
    dy = first(DI.pushforward(f1, prep, _AD_FWD, (A, b), ((dA, db),)))

    @test dy ≈ dy_expected rtol = 1e-12 atol = 1e-12
end

@testset "Enzyme reverse pullback on pmcmc_matmul (via grad of dot)" begin
    # f(A, B) = dot(pmcmc_matmul(A, B), w)  is scalar; dA = w * B', dB = A' * w
    rng = MersenneTwister(3)
    M, K = 5, 4
    A = randn(rng, M, K)
    b = randn(rng, K)
    w = randn(rng, M)

    f = ((A_, b_),) -> dot(pmcmc_matmul(A_, b_), w)
    prep = DI.prepare_gradient(f, _AD_REV, (A, b))
    g = DI.gradient(f, prep, _AD_REV, (A, b))

    dA_expected = w * b'
    db_expected = A' * w
    @test g[1] ≈ dA_expected rtol = 1e-12 atol = 1e-12
    @test g[2] ≈ db_expected rtol = 1e-12 atol = 1e-12
end

@testset "Enzyme Const-arg forward JVP — pmcmc_matmul" begin
    rng = MersenneTwister(100)
    M, K, N = 5, 4, 3
    A = randn(rng, M, K)
    B = randn(rng, K, N)
    dA = randn(rng, M, K)
    dB = randn(rng, K, N)

    # Const(A), Duplicated(B): dY = A * dB
    (dY1,) = Enzyme.autodiff(
        Enzyme.Forward,
        pmcmc_matmul,
        Enzyme.Duplicated,
        Enzyme.Const(A),
        Enzyme.Duplicated(B, dB),
    )
    @test dY1 ≈ A * dB rtol = 1e-12 atol = 1e-12

    # Duplicated(A), Const(B): dY = dA * B
    (dY2,) = Enzyme.autodiff(
        Enzyme.Forward,
        pmcmc_matmul,
        Enzyme.Duplicated,
        Enzyme.Duplicated(A, dA),
        Enzyme.Const(B),
    )
    @test dY2 ≈ dA * B rtol = 1e-12 atol = 1e-12

    # Const(A), Const(B): tangent is zero
    (dY0,) = Enzyme.autodiff(
        Enzyme.Forward, pmcmc_matmul, Enzyme.Duplicated, Enzyme.Const(A), Enzyme.Const(B)
    )
    @test all(iszero, dY0)
end

@testset "Enzyme Const-arg reverse pullback — pmcmc_matmul" begin
    # f(A, B) = dot(pmcmc_matmul(A, B), w);  dA = w * B', dB = A' * w
    rng = MersenneTwister(101)
    M, K = 5, 4
    A = randn(rng, M, K)
    b = randn(rng, K)
    w = randn(rng, M)

    # Const(A): only B accumulates; expect db = A' * w
    db_buf = zero(b)
    Enzyme.autodiff(
        Enzyme.Reverse,
        (A_, B_, w_) -> dot(pmcmc_matmul(A_, B_), w_),
        Enzyme.Active,
        Enzyme.Const(A),
        Enzyme.Duplicated(b, db_buf),
        Enzyme.Const(w),
    )
    @test db_buf ≈ A' * w rtol = 1e-12 atol = 1e-12

    # Const(b): only A accumulates; expect dA = w * b'
    dA_buf = zero(A)
    Enzyme.autodiff(
        Enzyme.Reverse,
        (A_, B_, w_) -> dot(pmcmc_matmul(A_, B_), w_),
        Enzyme.Active,
        Enzyme.Duplicated(A, dA_buf),
        Enzyme.Const(b),
        Enzyme.Const(w),
    )
    @test dA_buf ≈ w * b' rtol = 1e-12 atol = 1e-12
end
