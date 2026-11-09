# Check of the gradient of the variational boundary PEPS (src/utility/observables_cubic_bpeps.jl): prints numbers, tests nothing.
#
# For a small 3D Ising PEPO (the one of test/bpeps.jl: g = 2, β = 0.2, D = 2) and a PEPS ψ1 = ψ0 + 0.1 η away from the optimum it compares, for one random direction η2:
#   * <grad, η2>                  the gradient of the cost function with fixed environments (differentiate_environment = false),
#   * FD frozen                   a finite difference of the cost with the environments kept fixed (must agree with <grad, η2>: tests the automatic differentiation),
#   * FD total                    a finite difference of the TOTAL cost, the environments re-converged for every ψ (differs from <grad, η2> by the omitted
#                                 environment term d(cost)/d(env) * d(env)/d(psi); 4 % in my sandbox run with χenv = 8, error source 3: environment error).
# The CTMRG convergence error is printed for every environment: the finite difference of the total cost only means something when it is small (< 1e-8).
# Warning: in this small case the 2D CTMRG converges to different fixed points for different random starting environments (the cost of the same PEPS differed
# by 0.03 between two seeds in my sandbox), so the loop at the end repeats the whole comparison for several seeds.
#
# Run (PowerShell, in the clone):   julia +1.12 -t 4 --project=. development/check_bpeps_gradient.jl
# To compare other environment dimensions, change the tuple of χenv values in the for loop at the end (every value costs several minutes).
using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using LinearAlgebra
using Random
const CE = ClusterExpansions

setprecision(128)
D, χpeps = 2, 3
ce_alg = ising_operators(1.0, 2.0, 0.0; T = BigFloat, symmetry = nothing)
ΔO = evolution_operator_cubic(ce_alg, 0.05; T_conv = Float64)
O = evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)
trunc_alg = NoEnvTruncation(truncrank(D))
for _ in 1:4
    global O
    O, = approximate_state((O, ΔO), trunc_alg; rot_invariance = false)
end

O2 = CE.bpeps_pepo_tensor(O)
T = InfinitePEPO(O2)
Random.seed!(1234)
ψ0 = PEPSKit.peps_normalize(CE.bpeps_initial_state(O2, T, ℂ^χpeps; maxiter_init = 20))
randdir() = (η = InfinitePEPS(map(t -> (x = similar(t); randn!(x); x), ψ0.A)); η / norm(η))
ψ1 = PEPSKit.peps_normalize(ψ0 + 0.1 * randdir())
η2 = randdir()
cost_value(ψ, e2, e3) = -log(abs(network_value(InfiniteSquareNetwork(ψ, T), e3) / network_value(InfiniteSquareNetwork(ψ), e2)))

for χenv in (8,), seed in (1, 2, 3)
    alg = SimultaneousCTMRG(; tol = 1.0e-11, maxiter = 2000, verbosity = 0)
    # converged environments (and the largest CTMRG convergence error) of <ψ|ψ> and <ψ|O|ψ>, started from env2, env3 or from a random environment
    function converge(ψ, env2 = nothing, env3 = nothing)
        n2 = InfiniteSquareNetwork(ψ)
        n3 = InfiniteSquareNetwork(ψ, T)
        e2, i2 = leading_boundary(env2 === nothing ? CE.bpeps_initial_environment(alg, n2, ℂ^χenv) : deepcopy(env2), n2, alg)
        e3, i3 = leading_boundary(env3 === nothing ? CE.bpeps_initial_environment(alg, n3, ℂ^χenv) : deepcopy(env3), n3, alg)
        return e2, e3, max(i2.convergence_error, i3.convergence_error)
    end

    Random.seed!(seed)       # seed of the random starting environment
    env2, env3, err = converge(ψ1)
    println("χpeps = $χpeps, χenv = $χenv, environment seed $seed: CTMRG error at ψ1 = $err, cost = $(cost_value(ψ1, env2, env3))")
    costfun = CE.generate_bpeps_cost_function(T; boundary_alg = alg)
    E, g = costfun((ψ1, env2, env3))      # converges once more (restart from env2, env3) and updates env2, env3 in place
    dd = PEPSKit.real_inner(ψ1, g, η2)
    frozen(ψ) = cost_value(ψ, env2, env3)
    function total(ψ)
        e2, e3, err = converge(ψ, env2, env3)
        err < 1.0e-8 || println("   warning: CTMRG error $err at a perturbed point, the finite difference of the total cost is unreliable")
        return cost_value(ψ, e2, e3)
    end
    for ε in (1.0e-3, 1.0e-4)
        fd_frozen = (frozen(ψ1 + ε * η2) - frozen(ψ1 - ε * η2)) / (2ε)
        fd_total = (total(ψ1 + ε * η2) - total(ψ1 - ε * η2)) / (2ε)
        println("   ε = $ε: <grad, η2> = $dd | FD frozen environments = $fd_frozen | FD total cost = $fd_total | relative difference total vs grad = $(abs(fd_total - dd) / abs(dd))")
    end
end
