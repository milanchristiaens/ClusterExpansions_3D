# Finds contraction orders and cuts ("plans") for the 27-tensor contraction contract_onesite with OMEinsumContractionOrders.jl and appends them to the plan file that contract_onesite_auto reads (default: src/utility/onesite_plans.csv).
# Rows that are already in the file are never changed; benchmark_contract_onesite.jl then times every plan that has no time yet.

using ClusterExpansions
const CE = ClusterExpansions

# ======================================================================================================================
# SETTINGS:
# ======================================================================================================================
# (chi_env, chi_peps , D): chi_env should be idealiter 2-4 times chi_peps, chi_peps should be 1.5 times D, D should be 3-6
# This is at the moment not really possible (chi_peps would go to 36), so only doable dims are used for now
const DEFAULT_DIMS = [
    # D = 3: chi_peps = 4 and 5 (1.5*3 = 4.5), chi_env = 2, 3, 4 times chi_peps
    (8, 4, 3),      # 0.5%    the low set of the validation of the benchmark (profile :validate of benchmark_contract_onesite.jl)
    (12, 4, 3),     # 3.7%    (12,4,3), (16,4,3), (10,5,3), (12,5,3): the plan benchmark on Hortense (development/hpc/job_plans.pbs)
    (16, 4, 3),     # 27.3%
    (10, 5, 3),     # 7.9%
    (11, 5, 3),     # 13.8%   laptop (CTMRG3D compare, profile :laptop)
    (12, 5, 3),     # 23.1%
    # Hortense: the candidates of the CTMRG3D dimension pilot
    (16, 5, 3),
    (16, 6, 3),
    (18, 6, 3),
    # D = 4: combinations that are doable
#    (6, 4, 4),      # 0.3%
#    (10, 5, 4),     # 18.8%
    # D = 5: combinations that are doable
#    (7, 5, 5)       # 4.5%
    # Higher dimensions are for now too costly
]
# only some of the sets: CE_ORDERS_DIMS = "10_5_3,12_4_3" (chienv_chipeps_D, comma separated); otherwise all of DEFAULT_DIMS
const DIMS = haskey(ENV, "CE_ORDERS_DIMS") ? [Tuple(parse.(Int, split(x, "_"))) for x in split(ENV["CE_ORDERS_DIMS"], ",")] : DEFAULT_DIMS
const RAM_GIBS  = [3, 7, 29]                    # RAM budgets in GiB for which a plan is made (3 = laptop and Hortense on 4 cores (the CTMRG3D compare); Hortense: 7 = 6 cores, 29 = 16 cores)
const KMAX      = 5                             # TREE_CURRENT and OPT_TRUE: at most this many cut bonds (brute force over all cut sets, at most 5 is possible; 4 takes ~4 s per tree at 54 bonds)
const ELTYPE    = Float64                       # element type of the tensors (Float64: 8 bytes, ComplexF64: 16 bytes); the CE Ising PEPOs are Float64
const ALLOCATOR = :reuse                        # the allocator the memory rule is applied for (:reuse or :manual)
const MARGIN    = 0.8                           # a plan counts as fitting if predicted RAM <= MARGIN * budget
const N_TARGETS = 1                             # number of extra trees with a memory target (target = sc_max - SC_STEP * j, j = 1..N_TARGETS)
const SC_STEP   = 6                             # the memory target is sc_max - SC_STEP (log2 of elements); with N_TARGETS = 1 this is the single target
const NTRIALS   = 10                             # TreeSA settings (more = better trees, slower)
const NITERS    = 100
const SLICE_NTRIALS = 50                        # settings of the slicer of OMEinsumContractionOrders (TreeSASlicer; its defaults are 10 and 10)
const SLICE_NITERS  = 50
const OME_SLICING = :both                       # how the OME trees are cut: :sliced (by the slicer of OME, source ome-*-sliced), :mincuts (the tree as it is, fewest cuts by brute force, source ome-*-mincuts) or :both
const PLAN_FILE = CE.ONESITE_PLAN_FILE          # plan file that rows are appended to (the same constant is in benchmark_contract_onesite.jl)
# ======================================================================================================================

function main()
    nplans, nnew = 0, 0
    for (i, (χenv, χpeps, D)) in enumerate(DIMS)
        println("\n===== combination $i of $(length(DIMS)): (chi_env, chi_peps, D) = ($χenv, $χpeps, $D) =====")
        rows = CE.onesite_find_plans(χenv, χpeps, D;
            ram_gibs = RAM_GIBS,
            T = ELTYPE,
            allocator = ALLOCATOR,
            kmax = KMAX,
            margin = MARGIN,
            n_targets = N_TARGETS,
            sc_step = SC_STEP,
            ntrials = NTRIALS,
            niters = NITERS,
            slice_ntrials = SLICE_NTRIALS,
            slice_niters = SLICE_NITERS,
            ome_slicing = OME_SLICING,
            verbosity = 1)

        # written right away: an interrupted long list keeps the combinations that are done
        n = CE.onesite_write_plans(PLAN_FILE, rows)
        nplans += length(rows); nnew += n
        println("$(length(rows)) plans found for (chi_env, chi_peps, D) = ($χenv, $χpeps, $D), $n of them were new and appended to $PLAN_FILE")
        isempty(rows) && println("  (no plan fits the budgets $RAM_GIBS with at most $KMAX cuts: more RAM_GIBS or KMAX, or other dimensions)")
        flush(stdout)
    end

    println("\n$nplans plans found for $(length(DIMS)) combinations, $nnew of them were new and appended to $PLAN_FILE")
    println("Next: time all plans that have no time yet (the times are written back into the plan file):")
    println("  julia +1.12 -O1 -t 8 --project=. development/contraction_plans/benchmark_contract_onesite.jl")
    println("  (if you changed PLAN_FILE here, change it in benchmark_contract_onesite.jl too)")
end

main()