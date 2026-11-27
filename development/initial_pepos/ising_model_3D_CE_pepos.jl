# Stage 1 of the Ising compare runs: evolves the CE PEPOs of the 3D Ising model once to all β in BETAS for every D_CE in D_CES (and the exact PEPOs) and saves them in data/pepos/*.jld2.
# Stage 2 (the compare scripts and the decompositions check) reads these files, so all algorithms use exactly the same PEPOs.

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using Random
using LinearAlgebra
using JLD2

# compare_tools.jl is only included for FlushingLogger (the log of a killed Hortense job stays complete)
include(joinpath(@__DIR__, "pepo_files.jl"))
include(joinpath(@__DIR__, "..", "profiling", "tools", "compare_tools.jl"))

# ======================================================================================================================
# Settings
# ======================================================================================================================
const D_CES = [2, 3]                                    # one file per D_CE (laptop D_CE = 2 for vbPEPS, D_CE = 3 for CTMRG3D and on Hortense)
const BETAS = [0.15, 0.19, 0.195, 0.20, 0.205, 0.21, 0.215, 0.22, 0.225, 0.2275, 0.23, 0.24, 0.25, 0.35]   # β_ALL of the compare scripts and the β of their pilots (increasing, multiples of Δβ_iter)
const EXACT_FILE = true                                 # also the file of the exact PEPOs (bond dimension 2; only g = 0)
const CHIPEPS = [2, 3, 4, 5, 6, 7, 8]                   # χpeps of the start states of the vbPEPS
const Δβ_iter = 0.0025
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

    # We first write a temporary file and then rename it, so a job that reads the file never sees half of one
    tmp = replace(path, r"\.jld2$" => "_tmp.jld2")
    jldsave(tmp; beta = collect(Float64, BETAS), D_CE, dbeta = Δβ_iter, J, g, maxiter_init = MAXITER_INIT, psi0_seed_base = PSI0_SEED_BASE,
            chipeps = collect(Int, CHIPEPS), pepo3d, O = O2, M = M2, psi0)
    mv(tmp, path; force = true)
    return nothing
end

main()
