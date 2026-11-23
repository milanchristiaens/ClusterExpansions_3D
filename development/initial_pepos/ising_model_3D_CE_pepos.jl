# Stage 1 of the Ising compare runs: the CE PEPOs of the 3D Ising model, evolved ONCE to all β in BETAS for every D_CE in D_CES, and saved to
# data/pepos/ising_3D_J<J>_g<g>_D<D_CE>_dbeta<Δβ>.jld2 (one file per D_CE; see pepo_files.jl in this folder for the content). With EXACT_FILE also the file of the
# EXACT PEPOs (exact_ising_pepo of pepo_files.jl, bond dimension 2): data/pepos/ising_3D_exact_J<J>_g<g>_D2.jld2, the same content (O, M and the
# start states of the vbPEPS with the same seeds), so CE_COMPARE_PEPO = exact can use it in both compare scripts.
# Stage 2 reads these files: ising_model_3D_CE_CTMRG_compare.jl (the 3D PEPOs), ising_model_3D_CE_vbPEPS_compare.jl and
# development/checks/vbpeps/check_vbpeps_decompositions.jl (the 2D tensors O and M of the vbPEPS and its start states). So all algorithms use exactly the same PEPOs,
# and the vbPEPS of BetterPEPSKit runs in its own environment, without ClusterExpansions.
#
# Run it in the ClusterExpansions environment (julia --project=.) before stage 2, and again after a change of the settings below
# (an existing file of the same D_CE, Δβ, J and g is replaced). The compare scripts stop with an error if a β, D_CE or χpeps that they need is not in a file.
#
# Per file:
#   the PEPOs: build_pepos of pepo_files.jl (rot_invariance = true, Float64), at every β of BETAS
#   O and M: bpeps_pepo_tensor(PEPO) and bpeps_pepo_tensor(PEPO, σᶻ) of ClusterExpansions
#   the start states of the vbPEPS for every χpeps of CHIPEPS: bpeps_initial_state (power method, MAXITER_INIT steps) after Random.seed!(psi0_seed(β, Δβ)),
#   exactly as bpeps_optimize makes its start state (so the vbPEPS of PEPSKit and of BetterPEPSKit start from the same state)

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using Random
using LinearAlgebra
using JLD2

#   pepo_files.jl (this folder): build_pepos, exact_ising_pepo, pepo_file, psi0_seed, pack_tensor, PSI0_SEED_BASE
#   compare_tools.jl (development/profiling/tools): only for FlushingLogger (the log of a killed Hortense job stays complete)
include(joinpath(@__DIR__, "pepo_files.jl"))
include(joinpath(@__DIR__, "..", "profiling", "tools", "compare_tools.jl"))

# ======================================================================================================================
# Settings
# ======================================================================================================================
const D_CES = [2, 3]                                    # one file per D_CE (laptop D_CE = 2 for vbPEPS, D_CE = 3 for CTMRG3D and on Hortense)
const BETAS = [0.15, 0.19, 0.20, 0.21, 0.22, 0.225, 0.23, 0.25, 0.35]   # β_ALL of the compare scripts and the β of their pilots (increasing, multiples of Δβ_iter;
                                                                         # 0.20 since 23 Nov, for check_ctmrg3d_exponent)
const EXACT_FILE = true                                 # also the file of the exact PEPOs (bond dimension 2; only g = 0)
const CHIPEPS = [2, 3, 4, 5, 6, 7, 8]                   # χpeps of the start states of the vbPEPS (4: the decompositions benchmark; 5, 6: since 21 Nov,
                                                        # χpeps = 5 as in Sander's CE + vbPEPS figure; 7, 8: since 22 Nov, the vbPEPS pilots (·,7,2), (·,8,2))
const Δβ_iter = 0.005
const J = 1.0
const g = 0.0
const MAXITER_INIT = 20                                 # power-method steps of the start states (base_kwargs.maxiter_init of the vbPEPS compare)

const T = Float64
const spin_symmetry = Trivial

# ======================================================================================================================
# Main
# ======================================================================================================================
function main()
    obs = SpinOperators.σᶻ(T, spin_symmetry)
    mkpath(PEPO_DATA_DIR)
    for D_CE in D_CES
        t0 = time()
        Os = build_pepos(BETAS, Δβ_iter, D_CE, J, g)
        path = pepo_file(D_CE, Δβ_iter, J, g)
        save_pepo_file(path, Os, D_CE, obs)
        @info "D_CE = $D_CE: $(length(BETAS)) PEPOs and start states for χpeps = $CHIPEPS in $(round(time() - t0; digits = 1)) s, saved to $path"
    end
    if EXACT_FILE
        g == 0 || error("EXACT_FILE: the exact PEPO is only the Ising model at g = 0")
        t0 = time()
        path = pepo_file(2, Δβ_iter, J, g; kind = :exact)
        save_pepo_file(path, [exact_ising_pepo(β; J) for β in BETAS], 2, obs)
        @info "exact: $(length(BETAS)) PEPOs and start states for χpeps = $CHIPEPS in $(round(time() - t0; digits = 1)) s, saved to $path"
    end
    return nothing
end

# One file: the 3D PEPOs Os (at BETAS), the 2D tensors O and M of the vbPEPS and its start states
function save_pepo_file(path, Os, D_CE, obs)
    pepo3d, O2, M2 = Any[], Any[], Any[]
    # psi0 is a list of lists, one per χpeps: psi0[k][i] is the start state for CHIPEPS[k] at BETAS[i]
    psi0 = [Any[] for _ in CHIPEPS]
    for (i, β) in enumerate(BETAS)
        PEPO = Os[i]
        # traces the two physical legs and reorders the legs to O[D, U; N, E, S, W]
        O = ClusterExpansions.bpeps_pepo_tensor(PEPO)
        # pack_tensor: stores each tensor as plain data (needed for TensorKit 0.17.3 vs 0.17.1)
        push!(pepo3d, pack_tensor(PEPO))
        push!(O2, pack_tensor(O))
        push!(M2, pack_tensor(ClusterExpansions.bpeps_pepo_tensor(PEPO, obs)))
        
        # Now start states for vbpeps after some power iteration steps
        for (k, χ) in enumerate(CHIPEPS)
            Random.seed!(psi0_seed(β, Δβ_iter))
            push!(psi0[k], pack_tensor(ClusterExpansions.bpeps_initial_state(O, InfinitePEPO(O), ℂ^χ; maxiter_init = MAXITER_INIT)[1, 1]))
        end
    end

    # Saving:
    # e.g. data/pepos/ising_3D_J1.0_g0.0_D3_dbeta0.005.jld2
    # We first write a temporary file and then rename it (23 Nov): a job that reads the file while we rebuild it sees the old or the new file, never half of one
    tmp = replace(path, r"\.jld2$" => "_tmp.jld2")
    jldsave(tmp; beta = collect(Float64, BETAS), D_CE, dbeta = Δβ_iter, J, g, maxiter_init = MAXITER_INIT, psi0_seed_base = PSI0_SEED_BASE,
            chipeps = collect(Int, CHIPEPS), pepo3d, O = O2, M = M2, psi0)
    mv(tmp, path; force = true)
    return nothing
end

main()
