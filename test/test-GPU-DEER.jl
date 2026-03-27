using Test
using Random
using LinearAlgebra

using ParallelMCMC
import ParallelMCMC: DEER
import ADTypes

cuda_ok = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !cuda_ok
    @warn "CUDA not available or not functional — skipping GPU DEER tests"
else

_logp_gpu(x) = -0.5f0 * sum(abs2, x)
_gradlogp_gpu(x) = -x

@testset "solve_affine_scan_diag CPU vs GPU" begin
    rng = MersenneTwister(1)
    D, T = 4, 16

    A_cpu  = randn(rng, Float32, D, T)
    B_cpu  = randn(rng, Float32, D, T)
    s0_cpu = randn(rng, Float32, D)

    S_cpu = DEER.solve_affine_scan_diag(A_cpu, B_cpu, s0_cpu)

    A_gpu  = CUDA.cu(A_cpu)
    B_gpu  = CUDA.cu(B_cpu)
    s0_gpu = CUDA.cu(s0_cpu)

    S_gpu = DEER.solve_affine_scan_diag(A_gpu, B_gpu, s0_gpu)

    @test S_gpu isa CUDA.CuMatrix
    @test size(S_gpu) == (D, T)
    @test Array(S_gpu) ≈ S_cpu  atol=1e-5
end

enzyme_ok = try
    using Enzyme
    true
catch
    false
end

if !enzyme_ok
    @warn "Enzyme not available — skipping GPU AD DEER tests"
else


# @testset "DEER.solve on GPU with AutoEnzyme" begin
#     rng = MersenneTwister(42)
#     D, T = 3, 8

#     ε    = 0.05f0
#     tape = map(1:T) do _
#         ξ_gpu = CUDA.cu(randn(rng, Float32, D))
#         u     = Float32(rand(rng))
#         MALATapeElement(ξ_gpu, u)
#     end

#     rec = DEER.TapedRecursion(
#         (x, te) -> MALA.mala_step_taped(_logp_gpu, _gradlogp_gpu, x, ε, te.ξ, te.u),
#         (x, te, a) -> MALA.mala_step_surrogate(_logp_gpu, _gradlogp_gpu, x, ε, te.ξ, a),
#         tape;
#         consts        = (x, te) -> (MALA.mala_accept_indicator(_logp_gpu, _gradlogp_gpu, x, ε, te.ξ, te.u),),
#         const_example = (0.0f0,),
#         backend       = ADTypes.AutoEnzyme(),
#     )

#     s0_gpu = CUDA.cu(randn(rng, Float32, D))
#     S, info = DEER.solve(rec, s0_gpu; jacobian=:stoch_diag, maxiter=20, return_info=true)

#     @test S isa CUDA.CuMatrix
#     @test size(S) == (D, T)
#     @test all(isfinite, Array(S))
# end

# @testset "DEERSampler interface on GPU" begin
#     rng = MersenneTwister(7)
#     D   = 3

#     model   = DensityModel(_logp_gpu, _gradlogp_gpu, D)
#     sampler = DEERSampler(0.05f0; T=8, maxiter=20, jacobian=:stoch_diag, backend=ADTypes.AutoEnzyme())

#     x0_gpu = CUDA.cu(randn(rng, Float32, D))
#     trans, state = ParallelMCMC.AbstractMCMC.step(rng, model, sampler; initial_params=x0_gpu)

#     @test trans isa DEERTransition
#     @test state isa DEERState
#     @test trans.x isa CUDA.CuVector
#     @test length(trans.x) == D
#     @test isfinite(Float32(trans.logp))
#     @test size(state.trajectory) == (D, 8)
#     @test state.trajectory isa CUDA.CuMatrix

#     # Second step consumes from the cached trajectory
#     trans2, state2 = ParallelMCMC.AbstractMCMC.step(rng, model, sampler, state)
#     @test state2.t == 2
#     @test trans2.x isa CUDA.CuVector
# end

end # enzyme_ok
end # cuda_ok
