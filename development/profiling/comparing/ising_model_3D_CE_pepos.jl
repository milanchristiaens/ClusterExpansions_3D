# Stage 1 of the Ising compare runs: the CE PEPOs of the 3D Ising model, evolved ONCE to all β in BETAS for every D_CE in D_CES, and saved to
# data/pepos/ising_3D_J<J>_g<g>_D<D_CE>_dbeta<Δβ>.jld2 (one file per D_CE; see ../tools/pepo_files.jl for the content).
# Stage 2 reads these files: ising_model_3D_CE_CTMRG_compare.jl (the 3D PEPOs), ising_model_3D_CE_vbPEPS_compare.jl and
# ising_model_3D_CE_vbPEPS_decompositions.jl (the 2D tensors O and M of the vbPEPS and its start states). So all algorithms use exactly the same PEPOs,
# and the vbPEPS of BetterPEPSKit runs in its own environment, without ClusterExpansions.
#
# Run it in the ClusterExpansions environment (julia --project=.) before stage 2, and again after a change of the settings below
# (an existing file of the same D_CE, Δβ, J and g is replaced). The compare scripts stop with an error if a β, D_CE or χpeps that they need is not in a file.
#
# Per file:
#   the PEPOs: as build_pepos of ../tools/compare_tools.jl (rot_invariance = true, Float64), at every β of BETAS
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

#   compare_tools.jl: build_pepos;  pepo_files.jl: pepo_file, psi0_seed, pack_tensor, PSI0_SEED_BASE
include(joinpath(@__DIR__, "..", "tools", "compare_tools.jl"))
include(joinpath(@__DIR__, "..", "tools", "pepo_files.jl"))

# ======================================================================================================================
# Settings
# ======================================================================================================================
const D_CES = [2, 3]                                    # one file per D_CE (laptop D_CE = 2 for vbPEPS, D_CE = 3 for CTMRG3D and on Hortense)
const BETAS = [0.10, 0.19, 0.21, 0.23, 0.25, 0.35]      # β_ALL of the compare scripts (increasing, multiples of Δβ_iter)
const CHIPEPS = [2, 3, 4]                               # χpeps of the start states of the vbPEPS (4: only the decompositions benchmark)
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
        pepo3d, O2, M2 = Any[], Any[], Any[]
        psi0 = [Any[] for _ in CHIPEPS]
        for (i, β) in enumerate(BETAS)
            PEPO = Os[i]
            O = ClusterExpansions.bpeps_pepo_tensor(PEPO)
            push!(pepo3d, pack_tensor(PEPO))
            push!(O2, pack_tensor(O))
            push!(M2, pack_tensor(ClusterExpansions.bpeps_pepo_tensor(PEPO, obs)))
            for (k, χ) in enumerate(CHIPEPS)
                Random.seed!(psi0_seed(β, Δβ_iter))
                push!(psi0[k], pack_tensor(ClusterExpansions.bpeps_initial_state(O, InfinitePEPO(O), ℂ^χ; maxiter_init = MAXITER_INIT)[1, 1]))
            end
        end
        path = pepo_file(D_CE, Δβ_iter, J, g)
        jldsave(path; beta = collect(Float64, BETAS), D_CE, dbeta = Δβ_iter, J, g, maxiter_init = MAXITER_INIT, psi0_seed_base = PSI0_SEED_BASE,
                chipeps = collect(Int, CHIPEPS), pepo3d, O = O2, M = M2, psi0)
        @info "D_CE = $D_CE: $(length(BETAS)) PEPOs and start states for χpeps = $CHIPEPS in $(round(time() - t0; digits = 1)) s, saved to $path"
    end
    return nothing
end

main()
