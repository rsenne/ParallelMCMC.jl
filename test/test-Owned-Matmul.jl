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

@testset "Enzyme forward JVP on pmcmc_dot matches analytical" begin
    # d/dt pmcmc_dot(a + t*da, b + t*db)|t=0 = dot(da, b) + dot(a, db)
    rng = MersenneTwister(12)
    a  = randn(rng, 6)
    b  = randn(rng, 6)
    da = randn(rng, 6)
    db = randn(rng, 6)

    dval_expected = dot(da, b) + dot(a, db)

    f1 = ((a_, b_),) -> pmcmc_dot(a_, b_)
    prep = DI.prepare_pushforward(f1, _AD_FWD, (a, b), ((da, db),); strict=Val(false))
    dval = first(DI.pushforward(f1, prep, _AD_FWD, (a, b), ((da, db),)))

    @test dval ≈ dval_expected rtol=1e-12 atol=1e-12
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

@testset "Enzyme forward JVP on pmcmc_dotsum matches analytical" begin
    # d/dt pmcmc_dotsum(A + t*dA, B + t*dB)|t=0 = sum(dA .* B) + sum(A .* dB)
    rng = MersenneTwister(22)
    A  = randn(rng, 4, 3)
    B  = randn(rng, 4, 3)
    dA = randn(rng, 4, 3)
    dB = randn(rng, 4, 3)

    dval_expected = sum(dA .* B) + sum(A .* dB)

    f1 = ((A_, B_),) -> pmcmc_dotsum(A_, B_)
    prep = DI.prepare_pushforward(f1, _AD_FWD, (A, B), ((dA, dB),); strict=Val(false))
    dval = first(DI.pushforward(f1, prep, _AD_FWD, (A, B), ((dA, dB),)))

    @test dval ≈ dval_expected rtol=1e-12 atol=1e-12
end

# ============================================================
# Enzyme Const-annotation paths
#
# DI doesn't expose Enzyme `Const` directly on arguments — it activates the
# whole gradient subject. The rule bodies all special-case `a isa Const` /
# `B isa Const`; these tests hit those branches via `Enzyme.autodiff`.
# ============================================================

@testset "Enzyme Const-arg forward JVP — pmcmc_matmul" begin
    rng = MersenneTwister(100)
    M, K, N = 5, 4, 3
    A  = randn(rng, M, K); B  = randn(rng, K, N)
    dA = randn(rng, M, K); dB = randn(rng, K, N)

    # Const(A), Duplicated(B): dY = A * dB
    (dY1,) = Enzyme.autodiff(
        Enzyme.Forward, pmcmc_matmul, Enzyme.Duplicated,
        Enzyme.Const(A), Enzyme.Duplicated(B, dB),
    )
    @test dY1 ≈ A * dB rtol=1e-12 atol=1e-12

    # Duplicated(A), Const(B): dY = dA * B
    (dY2,) = Enzyme.autodiff(
        Enzyme.Forward, pmcmc_matmul, Enzyme.Duplicated,
        Enzyme.Duplicated(A, dA), Enzyme.Const(B),
    )
    @test dY2 ≈ dA * B rtol=1e-12 atol=1e-12

    # Const(A), Const(B): tangent is zero
    (dY0,) = Enzyme.autodiff(
        Enzyme.Forward, pmcmc_matmul, Enzyme.Duplicated,
        Enzyme.Const(A), Enzyme.Const(B),
    )
    @test all(iszero, dY0)
end

@testset "Enzyme Const-arg reverse pullback — pmcmc_matmul" begin
    # f(A, B) = pmcmc_dot(pmcmc_matmul(A, B), w);  dA = w * B', dB = A' * w
    rng = MersenneTwister(101)
    M, K = 5, 4
    A = randn(rng, M, K); b = randn(rng, K); w = randn(rng, M)

    # Const(A): only B accumulates; expect db = A' * w
    db_buf = zero(b)
    Enzyme.autodiff(
        Enzyme.Reverse,
        (A_, B_, w_) -> pmcmc_dot(pmcmc_matmul(A_, B_), w_),
        Enzyme.Active,
        Enzyme.Const(A), Enzyme.Duplicated(b, db_buf), Enzyme.Const(w),
    )
    @test db_buf ≈ A' * w rtol=1e-12 atol=1e-12

    # Const(b): only A accumulates; expect dA = w * b'
    dA_buf = zero(A)
    Enzyme.autodiff(
        Enzyme.Reverse,
        (A_, B_, w_) -> pmcmc_dot(pmcmc_matmul(A_, B_), w_),
        Enzyme.Active,
        Enzyme.Duplicated(A, dA_buf), Enzyme.Const(b), Enzyme.Const(w),
    )
    @test dA_buf ≈ w * b' rtol=1e-12 atol=1e-12
end

@testset "Enzyme Const-arg forward JVP — pmcmc_dot" begin
    rng = MersenneTwister(110)
    a  = randn(rng, 6); b  = randn(rng, 6)
    da = randn(rng, 6); db = randn(rng, 6)

    (dv1,) = Enzyme.autodiff(
        Enzyme.Forward, pmcmc_dot, Enzyme.Duplicated,
        Enzyme.Const(a), Enzyme.Duplicated(b, db),
    )
    @test dv1 ≈ dot(a, db) rtol=1e-12 atol=1e-12

    (dv2,) = Enzyme.autodiff(
        Enzyme.Forward, pmcmc_dot, Enzyme.Duplicated,
        Enzyme.Duplicated(a, da), Enzyme.Const(b),
    )
    @test dv2 ≈ dot(da, b) rtol=1e-12 atol=1e-12

    (dv0,) = Enzyme.autodiff(
        Enzyme.Forward, pmcmc_dot, Enzyme.Duplicated,
        Enzyme.Const(a), Enzyme.Const(b),
    )
    @test dv0 == 0
end

@testset "Enzyme Const-arg reverse pullback — pmcmc_dot" begin
    rng = MersenneTwister(111)
    a = randn(rng, 6); b = randn(rng, 6)

    da_buf = zero(a)
    Enzyme.autodiff(
        Enzyme.Reverse, pmcmc_dot, Enzyme.Active,
        Enzyme.Duplicated(a, da_buf), Enzyme.Const(b),
    )
    @test da_buf ≈ b rtol=1e-12 atol=1e-12

    db_buf = zero(b)
    Enzyme.autodiff(
        Enzyme.Reverse, pmcmc_dot, Enzyme.Active,
        Enzyme.Const(a), Enzyme.Duplicated(b, db_buf),
    )
    @test db_buf ≈ a rtol=1e-12 atol=1e-12
end

@testset "Enzyme Const-arg forward JVP — pmcmc_dotsum" begin
    rng = MersenneTwister(120)
    A  = randn(rng, 4, 3); B  = randn(rng, 4, 3)
    dA = randn(rng, 4, 3); dB = randn(rng, 4, 3)

    (dv1,) = Enzyme.autodiff(
        Enzyme.Forward, pmcmc_dotsum, Enzyme.Duplicated,
        Enzyme.Const(A), Enzyme.Duplicated(B, dB),
    )
    @test dv1 ≈ sum(A .* dB) rtol=1e-12 atol=1e-12

    (dv2,) = Enzyme.autodiff(
        Enzyme.Forward, pmcmc_dotsum, Enzyme.Duplicated,
        Enzyme.Duplicated(A, dA), Enzyme.Const(B),
    )
    @test dv2 ≈ sum(dA .* B) rtol=1e-12 atol=1e-12

    (dv0,) = Enzyme.autodiff(
        Enzyme.Forward, pmcmc_dotsum, Enzyme.Duplicated,
        Enzyme.Const(A), Enzyme.Const(B),
    )
    @test dv0 == 0
end

@testset "Enzyme Const-arg reverse pullback — pmcmc_dotsum" begin
    rng = MersenneTwister(121)
    A = randn(rng, 4, 3); B = randn(rng, 4, 3)

    dA_buf = zero(A)
    Enzyme.autodiff(
        Enzyme.Reverse, pmcmc_dotsum, Enzyme.Active,
        Enzyme.Duplicated(A, dA_buf), Enzyme.Const(B),
    )
    @test dA_buf ≈ B rtol=1e-12 atol=1e-12

    dB_buf = zero(B)
    Enzyme.autodiff(
        Enzyme.Reverse, pmcmc_dotsum, Enzyme.Active,
        Enzyme.Const(A), Enzyme.Duplicated(B, dB_buf),
    )
    @test dB_buf ≈ A rtol=1e-12 atol=1e-12
end
