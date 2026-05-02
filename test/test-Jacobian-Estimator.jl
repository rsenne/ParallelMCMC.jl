using Test
using Random
using LinearAlgebra
using StatsBase

using ParallelMCMC
const DEER = ParallelMCMC.DEER
const MALA = ParallelMCMC.MALA

logp_stdnormal_B(x) = -0.5 * dot(x, x)
gradlogp_stdnormal_B(x) = -x

struct TaggedVector{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.IndexStyle(::Type{<:TaggedVector}) = IndexLinear()
Base.size(x::TaggedVector) = size(x.data)
Base.axes(x::TaggedVector) = axes(x.data)
Base.getindex(x::TaggedVector, i::Int) = x.data[i]
Base.setindex!(x::TaggedVector, v, i::Int) = setindex!(x.data, v, i)
function Base.similar(x::TaggedVector, ::Type{T}, dims::Dims{1}) where {T}
    TaggedVector(Vector{T}(undef, dims[1]))
end
function Base.similar(x::TaggedVector, ::Type{T}, n::Int) where {T}
    TaggedVector(Vector{T}(undef, n))
end
function Base.similar(x::TaggedVector, dims::Dims{1})
    TaggedVector(Vector{eltype(x)}(undef, dims[1]))
end
Base.similar(x::TaggedVector, n::Int) = TaggedVector(Vector{eltype(x)}(undef, n))
Base.copy(x::TaggedVector) = TaggedVector(copy(x.data))

# Reference log q(y|x) for MALA:
# y ~ Normal( x + ϵ∇logp(x), 2ϵ I )
function logq_mala_ref_B(
    y::AbstractVector, x::AbstractVector, gradlogp_x::AbstractVector, ϵ::Real
)
    μ = x .+ ϵ .* gradlogp_x
    d = length(x)
    r = y .- μ
    return -0.5 * dot(r, r) / (2ϵ) - (d / 2) * log(4π * ϵ)
end

# Construct a tape for toy affine recursion: b_t vectors
make_affine_tape(rng::AbstractRNG, D::Int, T::Int) = [randn(rng, D) for _ in 1:T]

@testset "Jacobian / surrogate tune-up" begin
    @testset "jac_diag_via_jvps gives exact diagonal on affine step" begin
        rng = MersenneTwister(20251231)
        D, T = 6, 5

        A = I + 0.1 .* randn(rng, D, D)
        tape = make_affine_tape(rng, D, T)

        # Explicit JVP of f(x) = A*x + b is just v -> A*v.
        step_fwd = (x, b) -> A * x + b
        jvp = (x, b, v) -> A * v
        rec = DEER.TapedRecursion(step_fwd, jvp, tape)

        x = randn(rng, D)
        for t in 1:T
            d = DEER.jac_diag_via_jvps(rec, x, t)
            @test size(d) == (D,)
            @test d ≈ diag(A) atol=1e-12 rtol=1e-12
        end
    end

    @testset "jac_diag_stoch approaches diag(J) as probes increase (affine step)" begin
        rng = MersenneTwister(20251231)
        D, T = 8, 3

        A = 0.6I + 0.05 .* randn(rng, D, D)
        tape = make_affine_tape(rng, D, T)

        step_fwd = (x, b) -> A * x + b
        jvp = (x, b, v) -> A * v
        rec = DEER.TapedRecursion(step_fwd, jvp, tape)

        x = randn(rng, D)
        t = 1
        dtrue = diag(A)

        # Compare MSE averaged over multiple independent probe RNG streams
        function mse_for_probes(K::Int; nrep::Int=25)
            mses = zeros(Float64, nrep)
            for r in 1:nrep
                rrng = MersenneTwister(10_000 + 97*r)
                dest = DEER.jac_diag_stoch(rec, x, t; probes=K, rng=rrng)
                mses[r] = mean((dest .- dtrue) .^ 2)
            end
            return mean(mses)
        end

        mse1 = mse_for_probes(1; nrep=25)
        mse8 = mse_for_probes(8; nrep=25)
        mse64 = mse_for_probes(64; nrep=25)

        @test mse8 < mse1
        @test mse64 < mse8
    end

    @testset "jac_diag_stoch default probe uses x-like storage" begin
        x = TaggedVector([1.0, 2.0, 3.0])
        tape = [1]
        step_fwd = (x, t) -> x
        jvp = function (x, t, v)
            v isa TaggedVector || throw(ArgumentError("expected TaggedVector tangent"))
            return copy(v)
        end
        rec = DEER.TapedRecursion(step_fwd, jvp, tape)

        d = DEER.jac_diag_stoch(rec, x, 1; rng=MersenneTwister(1))

        @test d isa TaggedVector
        @test collect(d) == ones(3)
    end

    @testset "Rademacher and DEER argument validation paths" begin
        rng = MersenneTwister(20260203)
        z = zeros(Float64, 12)
        out = DEER._rademacher!(z, rng)
        @test out === z
        @test all(abs.(z) .== 1)

        D, T = 3, 4
        tape = collect(1:T)
        step_fwd = (x, t) -> 0.8 .* x .+ t
        jvp = (x, t, v) -> 0.8 .* v
        rec = DEER.TapedRecursion(step_fwd, jvp, tape)
        x = randn(rng, D)

        @test_throws ArgumentError DEER.jac_diag_stoch(
            rec, x, 1; probes=0, rng=MersenneTwister(1)
        )
        @test_throws ArgumentError DEER.jac_diag_stoch(
            rec, x, 1; zbuf=zeros(D + 1), rng=MersenneTwister(1)
        )

        S = randn(rng, D, T)
        S_out = similar(S)
        s0 = randn(rng, D)
        ws = DEER.DEERWorkspace(S, s0)

        @test_throws ArgumentError DEER.deer_update!(
            ws, S_out, rec, s0, S; jacobian=:full
        )
        @test_throws ArgumentError DEER.deer_update!(
            ws, S_out, rec, s0, S; damping=0.0
        )
        @test_throws ArgumentError DEER.deer_update!(
            ws, S_out, rec, s0, S; probes=0
        )
        @test_throws ArgumentError DEER.deer_update!(
            ws, randn(rng, D, T + 1), rec, s0, S
        )

        @test_throws ArgumentError DEER.solve(rec, s0; maxiter=0)
        @test_throws ArgumentError DEER.solve(rec, s0; tol_abs=-1.0)
        @test_throws ArgumentError DEER.solve(rec, s0; tol_rel=-1.0)
        @test_throws ArgumentError DEER.solve(rec, s0; init=randn(rng, D, T + 1))
    end

    @testset "batched stoch_diag uses z .* Jz in DEER update" begin
        rng = MersenneTwister(20251231)
        D, T = 7, 13

        Atrue = 0.65 .+ 0.1 .* randn(rng, D, T)
        Btrue = 0.2 .* randn(rng, D, T)
        s0 = randn(rng, D)
        S_in = randn(rng, D, T)

        step_fwd = (x, t) -> view(Atrue, :, t) .* x .+ view(Btrue, :, t)
        jvp = (x, t, v) -> view(Atrue, :, t) .* v
        fwd_and_jvp_batch = (Xbar, Z) -> (Atrue .* Xbar .+ Btrue, Atrue .* Z)
        rec = DEER.TapedRecursion(
            step_fwd, jvp, collect(1:T); fwd_and_jvp_batch=fwd_and_jvp_batch
        )

        ws = DEER.DEERWorkspace(S_in, s0)
        S_out = similar(S_in)
        DEER.deer_update!(
            ws, S_out, rec, s0, S_in; jacobian=:stoch_diag, probes=1, rng=MersenneTwister(1)
        )

        S_ref = DEER.solve_affine_seq(Atrue, Btrue, s0)
        @test S_out ≈ S_ref atol=1e-12 rtol=1e-12
    end

    @testset "DEER solution satisfies recursion defects (MALA taped, :diag)" begin
        rng = MersenneTwister(20251231)
        D, T = 6, 80
        ϵ = 0.35

        x0 = randn(rng, D)
        ξs = [randn(rng, D) for _ in 1:T]
        us = rand(rng, T)
        tape = [(ξ=ξs[t], u=us[t]) for t in 1:T]

        step_fwd =
            (x, tt) -> MALA.mala_step_taped(
                logp_stdnormal_B, gradlogp_stdnormal_B, x, ϵ, tt.ξ, tt.u
            )
        hvp_fn =
            (pt, dir) ->
                DEER._hvp_nopre(gradlogp_stdnormal_B, DEER.DEFAULT_BACKEND, pt, dir)
        jvp =
            (x, tt, v) -> MALA.mala_step_surrogate_sigmoid_jvp(
                logp_stdnormal_B, gradlogp_stdnormal_B, x, ϵ, tt.ξ, tt.u, v, hvp_fn
            )

        rec = DEER.TapedRecursion(step_fwd, jvp, tape)

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
