# Check of the gradient of the variational boundary PEPS (src/utility/observables_cubic_bpeps.jl): prints numbers, tests nothing.
#
# For a small 3D Ising PEPO (the one of test/bpeps.jl: g = 2, β = 0.2, D = 2) and a PEPS ψ1 = ψ0 + 0.1 η away from the optimum it compares, for one random direction η2:
#   * <grad, η2>                  the gradient of the cost function with fixed environments (differentiate_environment = false),
#   * FD frozen                   a finite difference of the cost with the environments kept fixed (must agree with <grad, η2>: tests the automatic differentiation),
#   * FD total                    a finite difference of the TOTAL cost, the environments re-converged for every ψ (differs from <grad, η2> by the omitted
#                                 environment term d(cost)/d(env) * d(env)/d(psi)).
# The CTMRG convergence error is printed for every environment: the finite difference of the total cost only means something when it is small (< 1e-8).
# Warning: in this small case the 2D CTMRG converges to different fixed points for different random starting environments, so the loop at the end repeats the whole comparison for several seeds.
#
# To compare other environment dimensions, change the tuple of χenv values in the for loop at the end (every value costs several minutes).
using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using LinearAlgebra
using Random
const CE = ClusterExpansions

# CE is built in high precision
setprecision(128)
D, χpeps = 2, 3     # χpeps is bond dimension of boundary peps

# Now we build the pepo
ce_alg = ising_operators(1.0, 2.0, 0.0; T = BigFloat, symmetry = nothing)
ΔO = evolution_operator_cubic(ce_alg, 0.05; T_conv = Float64)
O = evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)

# Define the truncation and i time evolution
trunc_alg = NoEnvTruncation(truncrank(D))
# We perform 4 time steps: β = 4 × 0.05 = 0.2
for _ in 1:4
    global O
    O, = approximate_state((O, ΔO), trunc_alg; rot_invariance = false)
end

# Convert O to complex tensor with physical legs traced out
O2 = CE.bpeps_pepo_tensor(O)
# T is the PEPO on a 1x1x1 unit cell
T = InfinitePEPO(O2)

# Choose starting state and perturbations:

Random.seed!(1234)

# Starting boundary PEPS state ψ0 is found by using a random boundary peps as startig point and then performing
# 20 times power iteration of PEPO O/T on it.
ψ0 = PEPSKit.peps_normalize(CE.bpeps_initial_state(O2, T, ℂ^χpeps; maxiter_init = 20))

# Calculate a random unit-norm tangent direction:
# t -> (x = similar(t); randn!(x); x) is an anonymous function that takes a tensor t and returns a random tensor of the same kind:
randdir() = (η = InfinitePEPS(map(t -> (x = similar(t); randn!(x); x), ψ0.A)); η / norm(η))

# Define the test point (here we will evaulate the gradient)
ψ1 = PEPSKit.peps_normalize(ψ0 + 0.1 * randdir())

# Now we define the bPEPS cost function: −log |⟨ψ|T|ψ⟩ / ⟨ψ|ψ⟩|
cost_value(ψ, e2, e3) = -log(abs(network_value(InfiniteSquareNetwork(ψ, T), e3) / network_value(InfiniteSquareNetwork(ψ), e2)))


# Main loop: run one χenv for 2 random seeds  and 2 random directions
for χenv in (8,12), dir in (1, 2, 3), seed in (1, 2)
        # Define the 2D CTMRG algorithm: tight tolerance and high iteration cap
    alg = SimultaneousCTMRG(; tol = 1.0e-11, maxiter = 2000, verbosity = 0)
    
    # This converges both environments for a given PEPS, if no starting environments were given it uses
    # random one from bpeps_initial_environment 
    function converge(ψ, env2 = nothing, env3 = nothing)
        n2 = InfiniteSquareNetwork(ψ)
        n3 = InfiniteSquareNetwork(ψ, T)
        e2, i2 = leading_boundary(env2 === nothing ? CE.bpeps_initial_environment(alg, n2, ℂ^χenv) : deepcopy(env2), n2, alg)
        e3, i3 = leading_boundary(env3 === nothing ? CE.bpeps_initial_environment(alg, n3, ℂ^χenv) : deepcopy(env3), n3, alg)
        # It returns the environments (and the largest CTMRG convergence error) of <ψ|ψ> and <ψ|O|ψ>
        return e2, e3, max(i2.convergence_error, i3.convergence_error)
    end

    Random.seed!(100 + dir); η2 = randdir()     # direction number dir: same for both seeds, different between directions
    Random.seed!(seed)       # seed of the random starting environment
    # Now we converge the environment of ψ1
    env2, env3, err = converge(ψ1)
    # And print the CTMRG error and cost
    println("direction $dir, χpeps = $χpeps, χenv = $χenv, environment seed $seed: CTMRG error at ψ1 = $err, cost = $(cost_value(ψ1, env2, env3))")    
    # Now we define the cost function that the LBFGS would use, it uses differentiate_environment = false
    costfun = CE.generate_bpeps_cost_function(T; boundary_alg = alg)
    # g = the gradient the optimiser would use
    E, g = costfun((ψ1, env2, env3))      # converges once more (restart from env2, env3) and updates env2, env3 in place
    
    # Calculate the direction derivative from the gradient ⟨g, η2⟩
    # dd ≈ d/dε cost(ψ1 + ε·η2) at ε = 0
    # Calculated by: taking the inner product ⟨g, η2⟩
    dd = PEPSKit.real_inner(ψ1, g, η2)
    
    # function: Evaluate the cost at a new PEPS, but keep env fixed
    # the finite-difference reference for the part of the gradient that comes only from the explicit ψ dependence.
    frozen(ψ) = cost_value(ψ, env2, env3)
    
    # Full cost function: re-converges the environments at the new ψ, starting from the old ones, and then evaluates the cost
    function total(ψ)
        e2, e3, err = converge(ψ, env2, env3)
        err < 1.0e-8 || println("   warning: CTMRG error $err at a perturbed point, the finite difference of the total cost is unreliable")
        return cost_value(ψ, e2, e3)
    end

    # Calculate the central finite differences along η2 at two step sizes
    for ε in (1.0e-3, 1.0e-4)
        fd_frozen = (frozen(ψ1 + ε * η2) - frozen(ψ1 - ε * η2)) / (2ε)
        fd_total = (total(ψ1 + ε * η2) - total(ψ1 - ε * η2)) / (2ε)
        println("   ε = $ε: <grad, η2> = $dd | FD frozen environments = $fd_frozen | FD total cost = $fd_total | relative difference total vs grad = $(abs(fd_total - dd) / abs(dd))")
    end
end
