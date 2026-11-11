# Test of the variational boundary PEPS (src/utility/observables_cubic_bpeps.jl) on a small 3D Ising PEPO (cluster expansion, g = 2, β = 0.2, D = 2).
# Small sizes only (χpeps = 3, χenv = 8); most of the time is compilation.
# What is tested:
#   (1) the layout of the 2D PEPO tensor,
#   (2) the gradient of the cost function with fixed environments (differentiate_environment = false, the default) agrees with a finite difference of the cost
#       with frozen environments: this tests the automatic differentiation of the network values. It does NOT test that this gradient is the gradient of the
#       total cost (environments re-converged): that is only approximately true, see the comment in observables_cubic_bpeps.jl (section 4) and
#       development/checks/check_bpeps_gradient.jl, which prints the numbers,
#   (3) bpeps_optimize runs (a few LBFGS steps, from the power-method start and warm-started), gives a finite cost, a finite observable, and the nested "bPEPS: ..." timers.
# NOT tested here (a test that the LBFGS cost decreases would be unreliable): the 2D CTMRG of this small case converges to different fixed points for different random
# starting environments, so the cost function is not reproducible enough for such a test.
# Opt-in (not run by default, needs a long first Zygote compile): the gradient through the CTMRG (differentiate_environment = true),
#   switch it on with the environment variable BPEPS_TEST_FIXEDPOINT=1, for example in PowerShell: $env:BPEPS_TEST_FIXEDPOINT = "1"
# NOT tested here: the C4vCTMRG + RotateReflect path (needs a C4v symmetric PEPO), Z2-graded spaces.
using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using TimerOutputs
using KrylovKit
using Test
using Random
using LinearAlgebra
const CE = ClusterExpansions

Random.seed!(1234)

# 3D PEPO of the transverse-field Ising model at β = 0.2 (four steps of Δβ = 0.05, truncated to D = 2)
setprecision(128)
D, χpeps, χenv = 2, 3, 8
ce_alg = ising_operators(1.0, 2.0, 0.0; T = BigFloat, symmetry = nothing)
ΔO = evolution_operator_cubic(ce_alg, 0.05; T_conv = Float64)
O = evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)
trunc_alg = NoEnvTruncation(truncrank(D))
for _ in 1:4
    global O
    O, = approximate_state((O, ΔO), trunc_alg; rot_invariance = false)
end
obs = SpinOperators.σˣ(Float64, Trivial)

@testset "2D PEPO tensor" begin
    O2 = CE.bpeps_pepo_tensor(O)
    @test numout(O2) == 2 && numin(O2) == 4          # (DOWN, UP) physical pair, (N, E, S, W) in-plane
    @test scalartype(O2) <: Complex                  # always complex, see bpeps_pepo_tensor
    @test dim(codomain(O2)) == dim(domain(O)[1]) * dim(domain(O)[4])      # the physical legs of the 2D problem are the DOWN and UP legs of the 3D PEPO
    M2 = CE.bpeps_pepo_tensor(O, id(domain(obs)))    # inserting the identity does not change the tensor
    @test M2 ≈ O2
end

# power-method start (20 steps) and a perturbed PEPS ψ1 = ψ0 + 0.1 η at which the gradient is not zero
O2 = CE.bpeps_pepo_tensor(O)
T = InfinitePEPO(O2)
ψ0 = PEPSKit.peps_normalize(CE.bpeps_initial_state(O2, T, ℂ^χpeps; maxiter_init = 20))
randdir() = (η = InfinitePEPS(map(t -> (x = similar(t); randn!(x); x), ψ0.A)); η / norm(η))
ψ1 = PEPSKit.peps_normalize(ψ0 + 0.1 * randdir())
η2 = randdir()
alg = SimultaneousCTMRG(; tol = 1.0e-8, maxiter = 200, verbosity = 0)
cost_value(ψ, e2, e3) = -log(abs(network_value(InfiniteSquareNetwork(ψ, T), e3) / network_value(InfiniteSquareNetwork(ψ), e2)))

@testset "gradient with fixed environments" begin
    env2 = CE.bpeps_initial_environment(alg, InfiniteSquareNetwork(ψ1), ℂ^χenv)
    env3 = CE.bpeps_initial_environment(alg, InfiniteSquareNetwork(ψ1, T), ℂ^χenv)
    costfun = CE.generate_bpeps_cost_function(T; boundary_alg = alg)       # differentiate_environment = false is the default
    # the cost function converges the CTMRG and stores the result in env2, env3 (in place); these are the environments at which the gradient is taken
    E, g = costfun((ψ1, env2, env3))
    @test isfinite(E)
    @test E ≈ cost_value(ψ1, env2, env3) atol = 1.0e-8
    dd = PEPSKit.real_inner(ψ1, g, η2)                # directional derivative <g, η2> (PEPSKit convention, the one used by LBFGS)
    @test abs(dd) > 1.0e-8                            # not at a stationary point

    ε = 1.0e-3
    frozen(ψ) = cost_value(ψ, env2, env3)
    fd_frozen = (frozen(ψ1 + ε * η2) - frozen(ψ1 - ε * η2)) / (2ε)
    @test abs(fd_frozen - dd) < 1.0e-4 * abs(dd)       # (2) the automatic differentiation is correct for fixed environments
end

@testset "bpeps_optimize, observable, timers" begin
    reset_timer!(CE.to)
    kw = (; boundary_tol = 1.0e-8, boundary_maxiter = 50, optimizer_maxiter = 2)
    res = CE.bpeps_optimize(O, ℂ^χpeps, ℂ^χenv; kw..., maxiter_init = 10)
    @test isfinite(res.cost)
    @test res.numfg >= 1

    m = only(CE.bpeps_observables(O, [obs], res))
    @info "bpeps test: cost = $(res.cost), numfg = $(res.numfg), <σˣ> = $m"
    @test isfinite(m)
    @test abs(imag(m)) < 1.0e-2
    @test 0 < real(m) < 1

    # nested timers: LBFGS > cost function > {the two CTMRG contractions, forward pass, backward pass}
    to = CE.to
    @test haskey(to, "bPEPS: LBFGS optimization")
    lb = to["bPEPS: LBFGS optimization"]
    @test haskey(lb, "bPEPS: cost function (value + gradient)")
    cf = lb["bPEPS: cost function (value + gradient)"]
    @test haskey(cf, "bPEPS: n2 leading_boundary (CTMRG of <psi|psi>)") && haskey(cf, "bPEPS: n3 leading_boundary (CTMRG of <psi|O|psi>)")
    @test haskey(cf, "bPEPS: forward pass") && haskey(cf, "bPEPS: backward pass (gradient of the network values)")
    @test haskey(cf["bPEPS: forward pass"], "bPEPS: network_value and cost")
    @test TimerOutputs.ncalls(cf) == res.numfg          # one cost function call per function evaluation

    # warm start from the result of the first call (state0), as warm_start = true in the timing script does
    res2 = CE.bpeps_optimize(O, ℂ^χpeps, ℂ^χenv; kw..., optimizer_maxiter = 1, state0 = (res.psi, res.env2, res.env3))
    @test isfinite(res2.cost)
    m2 = only(CE.bpeps_observables(O, [obs], res2))
    @test isfinite(m2)
end

# Opt-in: the gradient through the CTMRG (PEPSKit.hook_pullback, fixed-point gradient). The first call compiles the pullback of a CTMRG iteration with Zygote:
# minutes on a normal machine (I could not finish it in my sandbox, so this test has never been run by me and its tolerance is a guess).
if get(ENV, "BPEPS_TEST_FIXEDPOINT", "0") == "1"
    @testset "gradient through the CTMRG (differentiate_environment = true)" begin
        env2 = CE.bpeps_initial_environment(alg, InfiniteSquareNetwork(ψ1), ℂ^χenv)
        env3 = CE.bpeps_initial_environment(alg, InfiniteSquareNetwork(ψ1, T), ℂ^χenv)
        costfun_fixed = CE.generate_bpeps_cost_function(T; boundary_alg = alg)
        E, g_fixed = costfun_fixed((ψ1, env2, env3))                      # also converges env2, env3
        gradient_alg = FixedPointGradient(; solver_alg = KrylovKit.Arnoldi(; tol = 1.0e-8, maxiter = 10, krylovdim = 30, eager = true))
        costfun_full = CE.generate_bpeps_cost_function(T; boundary_alg = alg, gradient_alg, differentiate_environment = true)
        E_full, g_full = costfun_full((ψ1, deepcopy(env2), deepcopy(env3)))
        @test E_full ≈ E atol = 1.0e-6                                    # same cost (same converged environments)
        # the full gradient differs from the one with fixed environments by the environment term (4 % and 42 % of the directional derivative in my sandbox tests, χenv = 8)
        @test norm(g_full - g_fixed) < 1.0 * norm(g_fixed)
    end
end
