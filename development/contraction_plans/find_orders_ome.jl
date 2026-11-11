# Finds contraction orders + cuts ("plans") for the 27-tensor contraction contract_onesite with OMEinsumContractionOrders.jl,
# and appends them to the plan file that contract_onesite_auto reads (default: src/utility/onesite_plans.csv).
# Nothing here is needed while we run a simulation: it only prepares the csv.
#
# WHAT IT DOES
#   For every combination (chi_env, chi_peps, D) of DEFAULT_DIMS it asks
#   OMEinsumContractionOrders for contraction trees. The candidates are always TREE_CURRENT and TREE_FAST (the two fixed trees of the code), the tree with the fewest
#   flops ("ome-flops") and N_TARGETS trees that also have to keep their largest tensor small ("ome-scN"). For every RAM budget in RAM_GIBS every candidate gets its
#   best cut bonds (brute force over all sets of at most KMAX bonds); a candidate that does not fit the budget is dropped, and the same plan for several budgets is
#   kept once (for the smallest budget).
#   Every combination is planned one after the other and its rows are appended to the plan file right away, so a long list that is interrupted keeps what was done.
#   Rows that are already in the file are never changed (so measured times are never lost); if you change the settings, the new plans are added next to the old ones.
#   The search is random (TreeSA): another run can give other trees.
#
# NEXT STEP: development/contraction_plans/benchmark_contract_onesite.jl times every plan that has no time yet and writes the times back into the plan file.
#   The predicted flops are only a PROXY for the wall time (a tree with fewer flops can run slower, we saw ~1.4x for TREE_FAST).
#   contract_onesite_auto only uses a plan of the file if it has a measured time (select = :measured, the default).

using ClusterExpansions
const CE = ClusterExpansions

# ======================================================================================================================
# SETTINGS:
# ======================================================================================================================
# (chi_env, chi_peps , D): chi_env should be idealiter 2-4 times chi_peps, chi_peps should be 1.5 times D, D should be 3-6
# This is at the moment not really possible (chi_peps would go to 36), so only doable dims are used for now
const DEFAULT_DIMS = [
    # D = 3: chi_peps = 4 and 5 (1.5*3 = 4.5), chi_env = 2, 3, 4 times chi_peps
    (8, 4, 3),      # 0.5%
    (12, 4, 3),     # 4.1%
    (16, 4, 3),     # 29.7%
    (10, 5, 3),     # 8.5%
    (12, 5, 3),     # 29.1%
    # D = 4: combinations that are doable
    (6, 4, 4),      # 0.3%
    (10, 5, 4),     # 23.0%
    # D = 5: combinations that are doable
    (7, 5, 5)       # 4.8%
    # Higher dimensions are for now too costly
]
const RAM_GIBS  = [4, 29, 59]              # RAM budgets in GiB for which a plan is made (4 = laptop; HPC: 29 if 1/8 of a node, 59 if 1/4 of a node, 238 if a full node)
const KMAX      = 4                             # at most this many cut bonds (brute force over all cut sets, at most 5 is possible; 4 takes ~4 s per tree at 54 bonds)
const BYTES     = 16                            # bytes per element: 16 = ComplexF64, 8 = Float64
const ALLOCATOR = :reuse                        # the allocator the memory rule is applied for (:reuse or :manual)
const MARGIN    = 0.8                           # a plan counts as fitting if predicted RAM <= MARGIN * budget
const N_TARGETS = 1                             # number of extra trees with a memory target (target = sc_max - SC_STEP * j, j = 1..N_TARGETS)
const SC_STEP   = 6                             # the memory target is sc_max - SC_STEP (log2 of elements); with N_TARGETS = 1 this is the single target
const NTRIALS   = 4                             # TreeSA settings (more = better trees, slower)
const NITERS    = 40
const PLAN_FILE = CE.ONESITE_PLAN_FILE          # plan file that rows are appended to (the same constant is in benchmark_contract_onesite.jl)
# ======================================================================================================================

function main()
    nplans, nnew = 0, 0
    for (i, (χenv, χpeps, D)) in enumerate(DEFAULT_DIMS)
        println("\n===== combination $i of $(length(DEFAULT_DIMS)): (chi_env, chi_peps, D) = ($χenv, $χpeps, $D) =====")
        rows = CE.onesite_find_plans(χenv, χpeps, D;
            ram_gibs = RAM_GIBS,
            bytes = BYTES,
            allocator = ALLOCATOR,
            kmax = KMAX,
            margin = MARGIN,
            n_targets = N_TARGETS,
            sc_step = SC_STEP,
            ntrials = NTRIALS,
            niters = NITERS,
            verbosity = 1)

        # written right away: an interrupted long list keeps the combinations that are done
        n = CE.onesite_write_plans(PLAN_FILE, rows)
        nplans += length(rows); nnew += n
        println("$(length(rows)) plans found for (chi_env, chi_peps, D) = ($χenv, $χpeps, $D), $n of them were new and appended to $PLAN_FILE")
        isempty(rows) && println("  (no plan fits the budgets $RAM_GIBS with at most $KMAX cuts: more RAM_GIBS or KMAX, or other dimensions)")
        flush(stdout)
    end

    println("\n$nplans plans found for $(length(DEFAULT_DIMS)) combinations, $nnew of them were new and appended to $PLAN_FILE")
    println("Next: time all plans that have no time yet (the times are written back into the plan file):")
    println("  julia +1.12 -O1 -t 8 --project=. development/contraction_plans/benchmark_contract_onesite.jl")
    println("  (if you changed PLAN_FILE here, change it in benchmark_contract_onesite.jl too)")
end

main()