# pepo_files.jl -- the PEPO files of the Ising compare runs: the CE PEPOs are evolved ONCE to all β (ising_model_3D_CE_pepos.jl, stage 1) and saved,
# and every measurement (CTMRG3D compare, vbPEPS compare, dense against block-Krylov) reads them (stage 2). So all algorithms use exactly the same
# PEPOs, and the vbPEPS of BetterPEPSKit can run in its own environment without ClusterExpansions.
#
# One JLD2 file per (D_CE, Δβ, J, g): data/pepos/ising_3D_J<J>_g<g>_D<D_CE>_dbeta<Δβ>.jld2 (see pepo_file), with
#   beta            the β values (increasing, multiples of dbeta)
#   D_CE, dbeta, J, g, maxiter_init, psi0_seed_base, chipeps
#   pepo3d[i]       the 3D PEPO at beta[i] (legs (P ⊗ P') ← (U, N, E, D, S, W)), for CTMRG3D and the vbPEPS of ClusterExpansions
#   O[i], M[i]      the 2D tensors of the vbPEPS at beta[i]: bpeps_pepo_tensor(PEPO) and bpeps_pepo_tensor(PEPO, σᶻ)
#   psi0[k][i]      the start state of the vbPEPS for chipeps[k] at beta[i]: bpeps_initial_state(O, ...) after Random.seed!(psi0_seed(beta[i], dbeta))
# Every tensor is stored as plain data (pack_tensor): [(dimension, isdual)] of the codomain and domain legs and the dense array, not as a TensorMap,
# because the BetterPEPSKit environment has another TensorKit version than ClusterExpansions. Only for spaces without symmetry (ComplexSpace).
#
# Used in both environments (ClusterExpansions and development/fast_vbpeps); the script that includes this file needs `using TensorKit` and `using JLD2`.

const PEPO_DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "pepos"))     # <repo>/data/pepos
const PSI0_SEED_BASE = 2000

# The file of these settings
pepo_file(D_CE, Δβ, J, g) = joinpath(PEPO_DATA_DIR, "ising_3D_J$(Float64(J))_g$(Float64(g))_D$(D_CE)_dbeta$(Float64(Δβ)).jld2")

# The seed of the start state of the vbPEPS at β (the same for every profile, so the same β always gets the same start state)
psi0_seed(β, Δβ) = PSI0_SEED_BASE + round(Int, β / Δβ)

# A tensor without symmetry as plain data, and back
pack_tensor(t) = (; cod = [(dim(v), isdual(v)) for v in codomain(t)], dom = [(dim(v), isdual(v)) for v in domain(t)], data = convert(Array, t))
_pf_space(v) = v[2] ? ComplexSpace(v[1])' : ComplexSpace(v[1])
_pf_prodspace(vs) = isempty(vs) ? one(ComplexSpace) : ⊗(map(_pf_space, vs)...)
unpack_tensor(p) = TensorMap(p.data, _pf_prodspace(p.cod) ← _pf_prodspace(p.dom))

# The content of the file of these settings (a Dict); an error that says how to make it if it is not there
function read_pepo_file(D_CE, Δβ, J, g)
    path = pepo_file(D_CE, Δβ, J, g)
    isfile(path) || error("No PEPO file $path: run development/profiling/comparing/ising_model_3D_CE_pepos.jl first (in the ClusterExpansions environment), " *
                          "with D_CE = $D_CE, Δβ = $Δβ, J = $J, g = $g in its settings")
    return JLD2.load(path)
end

# The index of β in the file (error if it is not there)
function pepo_index(f, β)
    i = findfirst(b -> isapprox(b, β; atol = 1.0e-10), f["beta"])
    i === nothing && error("β = $β is not in the PEPO file (D_CE = $(f["D_CE"]), β = $(f["beta"])): add it to BETAS of ising_model_3D_CE_pepos.jl and run it again")
    return i
end

# The 3D PEPOs at the β of β_obs (as TensorMaps; in the ClusterExpansions environment)
function load_pepos(β_obs, Δβ, D_CE, J, g)
    f = read_pepo_file(D_CE, Δβ, J, g)
    return [unpack_tensor(f["pepo3d"][pepo_index(f, β)]) for β in β_obs]
end

# The 2D tensors of the vbPEPS at the β of β_obs, packed: [(; O, M, psi0)] with the start state of this chipeps.
# maxiter_init must be the number of power-method steps of the file (the start state depends on it).
function load_bpeps_tensors(β_obs, Δβ, D_CE, J, g, chipeps; maxiter_init)
    f = read_pepo_file(D_CE, Δβ, J, g)
    k = findfirst(==(chipeps), f["chipeps"])
    k === nothing && error("chipeps = $chipeps has no start states in the PEPO file (chipeps = $(f["chipeps"])): add it to CHIPEPS of ising_model_3D_CE_pepos.jl and run it again")
    f["maxiter_init"] == maxiter_init || error("the start states of the PEPO file have maxiter_init = $(f["maxiter_init"]), not $maxiter_init: run ising_model_3D_CE_pepos.jl again")
    return [(; O = f["O"][i], M = f["M"][i], psi0 = f["psi0"][k][i]) for i in (pepo_index(f, β) for β in β_obs)]
end
