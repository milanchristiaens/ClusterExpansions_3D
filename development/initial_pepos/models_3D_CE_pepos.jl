# Stage 1 of the BP comparison: evolves the CE PEPOs of the models of model_pepos.jl (transverse-field Ising at g ≠ 0, Bose-Hubbard) once to all their β
# and saves them in data/pepos/model_3D_<model>_*.jld2; we also check that every O is real and exactly cube symmetric (needed by CTMRG3DOctant).
# For the models with vbpeps = true we also save the 2D tensors O, M (both observables) and the start states of the vbPEPS, as ising_model_3D_CE_pepos.jl.

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using LinearAlgebra
using JLD2
using Random

include(joinpath(@__DIR__, "pepo_files.jl"))
include(joinpath(@__DIR__, "model_pepos.jl"))

# ======================================================================================================================
# Settings (a Hortense job may pass CE_MODELS: model names separated by +, since the names contain _)
# ======================================================================================================================
const RUN_MODELS = split(get(ENV, "CE_MODELS", "tfim_g2+tfim_g3.5+tfim_g4.5+bh_a+bh_b"), '+')
const CHIPEPS = [3, 4, 5]          # χpeps of the vbPEPS start states
const MAXITER_INIT = 20            # power-method steps of the start states (maxiter_init of models_3D_CE_vbpeps_compare.jl)
# ======================================================================================================================

# the largest relative change of O under the cube permutations of its six legs
function cube_asymmetry(PEPO)
    @tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]
    o = convert(Array, flip(permute(O, ((), (1, 2, 3, 4, 5, 6))), [4 5 6]))
    perms = ((2, 1, 3, 5, 4, 6), (3, 2, 1, 6, 5, 4), (1, 3, 2, 4, 6, 5), (4, 5, 6, 1, 2, 3), (4, 2, 3, 1, 5, 6))
    return maximum(norm(permutedims(o, p) - o) / norm(o) for p in perms)
end

# the 2D tensors of the vbPEPS: O, M (a list of the two observables) and the start states psi0[k][i] for CHIPEPS[k] at β i
function bpeps_tensors(m, Os)
    obss = model_observables(m)
    O2 = [pack_tensor(ClusterExpansions.bpeps_pepo_tensor(O)) for O in Os]
    M2 = [[pack_tensor(ClusterExpansions.bpeps_pepo_tensor(O, obs)) for obs in obss] for O in Os]
    psi0 = map(CHIPEPS) do χ
        map(zip(m.betas, Os)) do (β, O)
            O2d = ClusterExpansions.bpeps_pepo_tensor(O)
            Random.seed!(psi0_seed(β, m.dbeta))
            pack_tensor(ClusterExpansions.bpeps_initial_state(O2d, InfinitePEPO(O2d), ℂ^χ; maxiter_init = MAXITER_INIT)[1, 1])
        end
    end
    return (; O = O2, M = M2, psi0, chipeps = collect(Int, CHIPEPS), maxiter_init = MAXITER_INIT)
end

function main()
    mkpath(MODEL_PEPO_DIR)
    for name in RUN_MODELS
        m = model_settings(name)
        t0 = time()
        Os = build_model_pepos(m)
        asym = maximum(cube_asymmetry, Os)
        eltypes = unique(string(scalartype(O)) for O in Os)
        path = model_pepo_file(name)
        tmp = replace(path, r"\.jld2$" => "_tmp.jld2")
        data = (; model = name, beta = collect(Float64, m.betas), D_CE = m.D_CE, dbeta = m.dbeta, pepo3d = [pack_tensor(O) for O in Os])
        m.vbpeps && (data = (; data..., bpeps_tensors(m, Os)...))
        jldsave(tmp; data...)
        mv(tmp, path; force = true)
        @info "$name: $(length(Os)) PEPOs in $(round(time() - t0; digits = 1)) s, element type $(join(eltypes, ", ")), largest cube asymmetry $asym, saved to $path"
        asym < 1e-10 || @warn "$name: O is not cube symmetric ($asym): CTMRG3DOctant will refuse it"
    end
    return nothing
end

main()
