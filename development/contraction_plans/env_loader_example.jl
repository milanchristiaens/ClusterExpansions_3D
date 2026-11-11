# This is a template that needs changing, the goal is to run the contraction benchmark on a 'real' environment and PEPO O and obs M tensor

# Used by development/contraction_plans/benchmark_contract_onesite.jl: set   const ENV_LOADER = "development/contraction_plans/env_loader_example.jl"   in the SETTINGS block of that file,
# then run   julia +1.12 -O1 -t 8 --project=. development/contraction_plans/benchmark_contract_onesite.jl
# The benchmark then times the plans of the plan file that were made for the bond dimensions of THIS environment on the real tensors,
# and writes the times into the column time_real_s (the plans for other dimensions are skipped).
# It must define `onesite_load_env()` returning `(env, M)`: a converged-or-not CTMRG3D environment and the tensor M (6 legs, (0,6))
# that `contract_onesite(env, M)` contracts it with.  

# We have to fill in: The PEPO source O and the spaces, and the env gets build automatically


using TensorKit, TensorOperations, MPSKit
using MatrixAlgebraKit: truncrank
using ClusterExpansions

# This function builds the env and the tensor M and returns both to the benchmark
function onesite_load_env()
    # Change the following to the correct PEPO: a tensor with 2 physical and 6 virtual legs (AbstractTensorMap{<:Any,<:Any,2,6})
    PEPO = error("EDIT 1 in development/contraction_plans/env_loader_example.jl: build or load the PEPO (e.g. the one used in your 3D CTMRG test)")
    
    # Change/input the spaces of chi_peps and chi_env
    bpepsspace = error("EDIT 2: chi_peps space, e.g. ℂ^5 or a graded space")
    envspace = error("EDIT 2: chi_env space, e.g. ℂ^8 or a graded space")

    # Define the tensor O from from PEPO, contract the physical legs
    @tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]
    # Above all legs end up in codomain, now we move them all to domain, and flip the spaces of DSW to their duals
    # Because env needs all arows to be pointing inwards into O
    O = flip(permute(O, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])

    # Call the constructor for a environment 
    env = CTMRG3D(O; trunc_env = truncrank(dim(envspace)), trunc_bpeps = truncrank(dim(bpepsspace)),
                  Vbp = bpepsspace, Vv = envspace, miniter = 5, maxiter = 150, tol = 1e-8, verbosity = 1)
    # Runs the CTMRG interation
    env, ϵs = MPSKit.leading_boundary(env)

    return env, O          # M = O (the "norm" tensor); an observable-dressed M has the same cost
end