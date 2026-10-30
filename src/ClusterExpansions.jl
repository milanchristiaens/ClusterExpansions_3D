module ClusterExpansions

using MatrixAlgebraKit
using TensorKit, TensorKitTensors
using KrylovKit
using BlockTensorKit: ⊕, SumSpace
using MPSKit
using PEPSKit
using Graphs
using Zygote, OptimKit
using TensorOperations
using TensorOperations: ManualAllocator
using GenericLinearAlgebra, GenericSchur
import TimerOutputs     # TimerOutputs is imported WITHOUT its macros: the package defines its own `@timeit` below.
using TimerOutputs: TimerOutput


const to = TimerOutput()

# Largest allocation (in bytes, including sub-sections) of ONE call, per section name.
# Allocation = bytes allocated by the Julia GC; memory from malloc (ManualAllocator in contract_onesite) is not counted.
const MAXALLOC = Dict{String, Int}()
reset_max_alloc!() = empty!(MAXALLOC)
function record_max_alloc!(label, bytes)
    key = String(label)
    MAXALLOC[key] = max(get(MAXALLOC, key, 0), Int(bytes))
    return nothing
end

# Same use as TimerOutputs' `@timeit to "name" expr`: times the section with TimerOutputs
# and ALSO records the allocation of this single call in MAXALLOC (the largest one is kept).
macro timeit(to, label, ex)
    return quote
        b0 = TimerOutputs.gc_bytes()
        try
            TimerOutputs.@timeit($(esc(to)), $(esc(label)), $(esc(ex)))
        finally
            record_max_alloc!($(esc(label)), TimerOutputs.gc_bytes() - b0)
        end
    end
end

include("clusterexpansions/utility/symmetries.jl")

include("utility/canonical_form.jl")
include("utility/observables.jl")
include("utility/observables_cubic_bpeps.jl")
include("utility/ctmrg_cubic.jl")
include("utility/ctmrg3D.jl")

include("time_evolution/utility.jl")
include("time_evolution/global_truncation.jl")
include("time_evolution/local_truncation.jl")
include("time_evolution/local_truncation_cubic.jl")
include("time_evolution/vomps_utility.jl")
include("time_evolution/vomps.jl")
include("time_evolution/simple_update_cubic.jl")

include("models/models.jl")
include("models/models_Trotter.jl")

include("time_evolution/time_evolve.jl")
include("time_evolution/time_evolve_cubic.jl")
include("time_evolution/time_evolve_cubic_ani.jl")
include("models/time_evolve_models.jl")

include("models/data_generation.jl")

include("clusterexpansions/generate_trees.jl")
include("clusterexpansions/generate_loops.jl")
include("clusterexpansions/clusters.jl")
include("clusterexpansions/contractions.jl")
include("clusterexpansions/PEPOs.jl")

include("clusterexpansions/solvers/linearsolvers.jl")
include("clusterexpansions/solvers/nonlinearsolvers_optimkit.jl")
include("clusterexpansions/solve_clusters.jl")

export rotl90_fermionic, rotl180_fermionic
export symmetrize

export Canonicalization, canonicalize
export PEPOObservable, PEPO_observables, localoperator_model, calculate_observables

export fidelity, apply_PEPO_exact
export ApproximateEnvTruncation, NoEnvTruncation
export VOPEPO_CTMRG, VOPEPO_VUMPS
export approximate_state

export ClusterExpansion
export ising_operators
export spinless_fermion_operators, spinless_fermion_model
export heisenberg_operators, heisenberg_XXX_operators
export J1J2_operators
export tJ_operators, hubbard_operators
export bose_hubbard_operators

export TrotterDecomposition, ising_operators_Trotter

export evolution_operator
export StaticTimeEvolution, TimeDependentTimeEvolution
export UniformTimeEvolution, SquaringTimeEvolution
export UniformGroundStateTimeEvolution, SquaringGroundStateTimeEvolution
export time_evolve, get_time_array, time_scan
export time_evolve_model, time_scan_model

export evolution_operator_cubic, time_evolve_cubic
export CTMRG3D

export SUWeight3D, InfiniteWeightPEPS3D, InfinitePEPS3D, product_peps3D
export CubicHamiltonian, SimpleUpdate3D, simpleupdate3D, su_iter3D
export onesite_rdm, bond_rdm, energy_per_site, canonicalize3D

export DensityMatrixSpace, vectorize, unvectorize
export ThermalCubicHamiltonian, thermal_gate, infinite_temperature_peps3D
export cubic_pepo, cubic_pepo_purified

export data_generation_SF_CE, data_generation_ising_CE
export data_generation_SF_SU, data_generation_ising_SU

export clusterexpansion

end # module ClusterExpansions
