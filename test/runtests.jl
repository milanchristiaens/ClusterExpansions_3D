using SafeTestsets

# disabled because it takes too long
# @time @safetestset "Ground state search" begin 
#     include("ground_state_search.jl")
# end

# Evolve 2D PEPO from β=0.1 to 0.8, using C4v cluster expansion
# Measures ⟨σᶻ⟩ and ⟨σˣ⟩ with VUMPS, this tests if we indeed see phase transition
@time @safetestset "Phase transition - Ising model" begin
    include("phase_transition_ising.jl")
end

# Same as above, but for different model
@time @safetestset "Phase transition - Spinless Fermion model" begin
    include("phase_transition_spinless_fermions.jl")
end

# Tests the 4-loop solver on a fermionic 2x2 plaquette
@time @safetestset "4-loop solver" begin
    include("utility_functions.jl")
end

# Check bookkeeping, gate and benchmarking of Simple update in 3D cubic lattice.
@time @safetestset "Simple update - cubic lattice" begin
    include("simple_update_cubic.jl")
end

# Check density-matrix (vectorize), thermal gate for SU at finite T 
@time @safetestset "Finite-temperature simple update - cubic lattice" begin
    include("thermal_simple_update_cubic.jl")
end

# Performs the exact SU algorithm on classical ising, checks with theoretical result
@time @safetestset "Classical Ising: exact D = 2 simple update - cubic lattice" begin
    include("classical_ising_simple_update_cubic.jl")
end

# Test the different symmetries
@time @safetestset "Symmetries" begin
    # C4v symmetry of 2D CE pepo
    include("spatial_symmetries.jl")
    # Tests the t-J model with Trivial/U1/SU2 particle/spin symmetries
    include("internal_symmetries.jl")
    # This is the main test: for Ising/Heisenberg/Spinless fermions it loops over all combinations of
    # Spatial symmetry, number type, and Cluster size (P= 3 or 4)
    # And then compares the results for a given model with each other
    # THis also means that (p=3 or p=4) fall within 5e-5 of each other.
    include("symmetries.jl")
end

# Compare CE PEPO with exact exp(−βH) for p=2, 3,4. The error must scale as β^p
@time @safetestset "Order of cluster expansion" begin
    include("exact_exponential.jl")
end

# Evolve 3D CE of Bose-Hubbard model to finite temperature
# Then check the tree rotations/flips of O_h group on PEPO
# If "Oh" was imposed it should be equal, otherwise not
@time @safetestset "O_h symmetry of 3D CE" begin
    include("3d_symmetries.jl")
end

# Test the slicing code (cost model, cut selection, plan csv, OMEinsumContractionOrders
# trees, allocators, contract_onesite_auto) on a mock environment
# Also tests for dense and Z2 graded tensors. Tests if every contraction results in same answer.
@time @safetestset "contract_onesite slicing" begin
    include("contract_onesite_slicing.jl")
end

# Variational boundary PEPS (observables_cubic_bpeps.jl) on a small 3D Ising PEPO: 2D
# PEPO tensor, gradient with fixed environments (automatic differentiation
# against finite differences), LBFGS from a perturbed PEPS, observable, nested timers.
# The gradient through the CTMRG is opt-in: BPEPS_TEST_FIXEDPOINT=1
@time @safetestset "Variational boundary PEPS (bPEPS)" begin
    include("bpeps.jl")
end
