# Tests of the 3D CTMRG (src/utility/ctmrg3D_helpers.jl, ctmrg3D_main.jl) that need no cluster expansion: rotations of the 27 tensors and exactly solvable PEPOs.
# Offered as an ADDITION (not in runtests.jl): add it with
#     @time @safetestset "CTMRG3D" begin include("ctmrg3d.jl") end
# Everything here is small (chi_env <= 4), most of the time is compilation of contract_onesite.
#   (1) the rotations rot_ud_90, rot_ns_90, rot_we_90 (random tensors of the right spaces): fourth power = identity, and the closed 27-tensor network is invariant under a rotation
#       of the environment together with the same permutation of the legs of M;
#   (2) a product state (bond dimension 1): <sigma^x> = tanh(beta h);
#   (3) a classical Ising chain along z (bond dimension 2): <sigma^z> = sinh(h) / sqrt(sinh(h)^2 + exp(-4K)), after a FIXED number of iterations;
#   (4) the classical 2D Ising model in the three planes (Onsager), a measurement of the error of the U*sqrt(S) projectors, see the comments there.
using TensorKit
using TensorKitTensors
using MPSKit
using ClusterExpansions
using LinearAlgebra
using Random
using Test
const CE = ClusterExpansions

Random.seed!(20261105)

# Expectation value of a one-site operator after a FIXED number of CTMRG iterations. The stopping rule of MPSKit.leading_boundary(::CTMRG3D) looks only at the corner spectrum of C[1];
# for rank-one corners (decoupled directions) it stops after miniter + 1 iterations while the faces A are still far from converged (see development notes), so the number of iterations is fixed here.
function ctmrg3d_value(PEPO, obs, χp, χe; iterations, west_updates = :auto)
    O = (@tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]; flip(permute(O, ((), (1, 2, 3, 4, 5, 6))), [4 5 6]))
    M = (@tensor M[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 2; -1 -2 -3 -4 -5 -6] * obs[2; 1]; flip(permute(M, ((), (1, 2, 3, 4, 5, 6))), [4 5 6]))
    env = CE.CTMRG3D(O; trunc_env = truncrank(χe), trunc_bpeps = truncrank(χp), Vbp = ℂ^χp, Vv = ℂ^χe, miniter = iterations, maxiter = iterations, tol = 0.0, west_updates)
    for _ in 1:iterations
        CE.ctmrgstep(env)
    end
    return CE.contract_onesite(env, M) / CE.contract_onesite(env, O)
end

# classical Ising PEPO O[s s'; u n e d s w] = δ_{s s'} e^{h s} Π B[s, a], B = sqrt([[e^K, e^-K], [e^-K, e^K]]), exact with bond dimension 2 in every direction (K = 0: rank-one, uncoupled bond).
# CTMRG3D needs the same virtual space on all six legs, so uncoupled directions also have dimension 2.
function bondmat(K)
    λ, V = eigen(Symmetric([exp(K) exp(-K); exp(-K) exp(K)]))
    return V * Diagonal(sqrt.(max.(λ, 0.0))) * V'
end
function ising_pepo(Kz, Ky, Kx, h)
    Bz, By, Bx = bondmat(Kz), bondmat(Ky), bondmat(Kx)
    arr = zeros(2, 2, 2, 2, 2, 2, 2, 2)
    for σ in 1:2, u in 1:2, n in 1:2, e in 1:2, d in 1:2, s in 1:2, w in 1:2
        arr[σ, σ, u, n, e, d, s, w] = exp(h * (σ == 1 ? 1.0 : -1.0)) * Bz[σ, u] * Bz[σ, d] * By[σ, n] * By[σ, s] * Bx[σ, e] * Bx[σ, w]
    end
    return TensorMap(arr, ℂ^2 ⊗ (ℂ^2)', ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)' ⊗ (ℂ^2)')
end
sz = TensorMap(Float64[1 0; 0 -1], ℂ^2 ← ℂ^2)

@testset "rotations of the 27 tensors" begin
    Vp, Vbp, Vv = ℂ^2, ℂ^2, ℂ^3
    T = ComplexF64
    O = randn(T, one(Vp) ← Vp^6)
    env = CE.CTMRG3D(O; Vp, Vbp, Vv)
    for i in 1:6;  env.A[i] = randn(T, space(env.A[1])); end
    for i in 1:12; env.E[i] = randn(T, space(env.E[1])); end
    for i in 1:8;  env.C[i] = randn(T, space(env.C[1])); end
    M = randn(T, one(Vp) ← Vp^6)
    perms = Dict(:ud => (1, 6, 2, 4, 3, 5), :ns => (6, 2, 1, 3, 5, 4), :we => (5, 1, 3, 2, 4, 6))      # the permutations of env.O in rot_*_90
    rots = Dict(:ud => CE.rot_ud_90, :ns => CE.rot_ns_90, :we => CE.rot_we_90)
    val0 = CE.contract_onesite(env, M)
    relerr(a, b) = abs(a - b) / max(abs(a), abs(b))

    for k in (:ud, :ns, :we)
        e = deepcopy(env)
        for _ in 1:4; rots[k](e); end
        @test all(norm(x - y) == 0 for (X, Y) in ((e.C, env.C), (e.E, env.E), (e.A, env.A)) for (x, y) in zip(X, Y))      # fourth power = identity
        @test norm(e.O - env.O) == 0

        e = deepcopy(env); rots[k](e)
        @test relerr(CE.contract_onesite(e, permute(M, ((), perms[k]))), val0) < 1e-12          # invariance of the closed network
        @test relerr(CE.contract_onesite(e, M), val0) > 1e-6                                    # negative control: without permuting M the value does change
    end
    # random sequences of rotations
    for _ in 1:4
        e = deepcopy(env); Mr = M
        for _ in 1:rand(3:6)
            k = rand((:ud, :ns, :we)); rots[k](e); Mr = permute(Mr, ((), perms[k]))
        end
        @test relerr(CE.contract_onesite(e, Mr), val0) < 1e-12
    end
end

@testset "product state: <σˣ> = tanh(βh)" begin
    β, h = 0.2, 2.0
    ρ = exp(β * h * [0.0 1.0; 1.0 0.0])
    O = TensorMap(reshape(ρ, 2, 2, 1, 1, 1, 1, 1, 1), ℂ^2 ⊗ (ℂ^2)', ℂ^1 ⊗ ℂ^1 ⊗ ℂ^1 ⊗ (ℂ^1)' ⊗ (ℂ^1)' ⊗ (ℂ^1)')
    sx = TensorMap([0.0 1.0; 1.0 0.0], ℂ^2 ← ℂ^2)
    m = only(MPSKit.expectation_value(O, [sx], ℂ^1, ℂ^1; miniter = 3, maxiter = 30, tol = 1.0e-10))
    @test m ≈ tanh(β * h) atol = 1.0e-12
end

@testset "classical Ising chain along z (exact, bond dimension 2)" begin
    for (K, h) in ((0.5, 0.3), (1.0, 0.1))
        exact = sinh(h) / sqrt(sinh(h)^2 + exp(-4K))
        m = ctmrg3d_value(ising_pepo(K, 0.0, 0.0, h), sz, 2, 4; iterations = 160)
        @test real(m) ≈ exact atol = 1.0e-10
    end
end

@testset "classical 2D Ising model in the three planes (Onsager, K = 0.6)" begin
    K = 0.6
    onsager = (1 - sinh(2K)^-4)^(1 / 8)
    ms = Float64[]
    for Ks in ((0.0, K, K), (K, 0.0, K), (K, K, 0.0))             # (z, y, x) couplings: the N-E, U-E and U-N plane
        push!(ms, abs(real(ctmrg3d_value(ising_pepo(Ks..., 0.0), sz, 3, 3; iterations = 60))))
    end
    # the three orientations are the same physical problem
    @test maximum(ms) - minimum(ms) < 1.0e-8
    # Error of the environment with the U*sqrt(S) projectors (src/utility/ctmrg3D_helpers.jl, get_projectors_corner/_edge): 1.5e-3 at chi = 3, 4, 5, independent of chi, while a plain
    # isometry corner transfer matrix reaches 1e-9 at chi = 3 (numpy check). This is a measurement of the present state, NOT a statement that this is the right value.
    # In a night test with U*S^p instead of U*sqrt(S) (monkey patch, development/night/e16-e18): this plane test is exact (8e-10) only for p = 0 (plain isometry), while for the genuine
    # 3D PEPO of test/bpeps.jl p = 0 does not converge and p >= 0.85 agrees with the bPEPS value to 1e-16 (p = 0.5: -3.7e-5). So no single exponent is right in both; if you change the
    # projectors, the @test_broken below will report 'unexpectedly passed' and should be turned into a @test.
    @test abs(ms[1] - onsager) < 3.0e-3
    @test_broken abs(ms[1] - onsager) < 1.0e-6
end
