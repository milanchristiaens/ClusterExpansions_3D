# pepo_files.jl -- the PEPOs of the Ising compare runs and their files: the CE PEPOs are evolved ONCE to all β (build_pepos below, called by
# ising_model_3D_CE_pepos.jl in this folder, stage 1) and saved,
# and every measurement (CTMRG3D compare, vbPEPS compare, dense against block-Krylov) reads them (stage 2). So all algorithms use exactly the same
# PEPOs, and the vbPEPS of BetterPEPSKit can run in its own environment without ClusterExpansions.
#
# One JLD2 file per (D_CE, Δβ, J, g): data/pepos/ising_3D_J<J>_g<g>_D<D_CE>_dbeta<Δβ>.jld2 (see pepo_file), and one file of the EXACT PEPOs
# (exact_ising_pepo below, bond dimension 2): data/pepos/ising_3D_exact_J<J>_g<g>_D2.jld2 (kind = :exact; same content, D_CE = 2, and dbeta only sets
# the seeds of the start states, the same seeds as the CE file). Both with
#   beta            the β values (increasing, multiples of dbeta)
#   D_CE, dbeta, J, g, maxiter_init, psi0_seed_base, chipeps
#   pepo3d[i]       the 3D PEPO at beta[i] (legs (P ⊗ P') ← (U, N, E, D, S, W)), for CTMRG3D and the vbPEPS of ClusterExpansions
#   O[i], M[i]      the 2D tensors of the vbPEPS at beta[i]: bpeps_pepo_tensor(PEPO) and bpeps_pepo_tensor(PEPO, σᶻ)
#   psi0[k][i]      the start state of the vbPEPS for chipeps[k] at beta[i]: bpeps_initial_state(O, ...) after Random.seed!(psi0_seed(beta[i], dbeta))
# Every tensor is stored as plain data (pack_tensor): [(dimension, isdual)] of the codomain and domain legs and the dense array, not as a TensorMap,
# because the BetterPEPSKit environment has another TensorKit version than ClusterExpansions. Only for spaces without symmetry (ComplexSpace).
#
# Used in both environments (ClusterExpansions and development/fast_vbpeps); the script that includes this file needs `using TensorKit` and `using JLD2`
# (build_pepos also needs `using ClusterExpansions`: only stage 1 calls it).

const PEPO_DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "pepos"))     # <repo>/data/pepos
const PSI0_SEED_BASE = 2000

# The file of these settings (kind = :ce, the CE PEPOs) or of the exact PEPOs (kind = :exact; D_CE and Δβ do not enter the name)
function pepo_file(D_CE, Δβ, J, g; kind = :ce)
    kind === :exact && return joinpath(PEPO_DATA_DIR, "ising_3D_exact_J$(Float64(J))_g$(Float64(g))_D2.jld2")
    kind === :ce || error("pepo_file: kind = :$kind, use :ce or :exact")
    return joinpath(PEPO_DATA_DIR, "ising_3D_J$(Float64(J))_g$(Float64(g))_D$(D_CE)_dbeta$(Float64(Δβ)).jld2")
end

# The seed of the start state of the vbPEPS at β (the same for every profile, so the same β always gets the same start state)
psi0_seed(β, Δβ) = PSI0_SEED_BASE + round(Int, β / Δβ)

# A tensor without symmetry as plain data, and back
pack_tensor(t) = (; cod = [(dim(v), isdual(v)) for v in codomain(t)], dom = [(dim(v), isdual(v)) for v in domain(t)], data = convert(Array, t))
_pf_space(v) = v[2] ? ComplexSpace(v[1])' : ComplexSpace(v[1])
_pf_prodspace(vs) = isempty(vs) ? one(ComplexSpace) : ⊗(map(_pf_space, vs)...)
unpack_tensor(p) = TensorMap(p.data, _pf_prodspace(p.cod) ← _pf_prodspace(p.dom))

# The content of the file of these settings (a Dict); an error that says how to make it if it is not there
function read_pepo_file(D_CE, Δβ, J, g; kind = :ce)
    kind === :exact && D_CE != 2 && error("the exact PEPOs have bond dimension 2: choose dims with D_CE = 2, not $D_CE")
    path = pepo_file(D_CE, Δβ, J, g; kind)
    isfile(path) || error("No PEPO file $path: run development/initial_pepos/ising_model_3D_CE_pepos.jl first (in the ClusterExpansions environment), " *
                          "with D_CE = $D_CE, Δβ = $Δβ, J = $J, g = $g in its settings" * (kind === :exact ? " and EXACT_FILE = true" : ""))
    return JLD2.load(path)
end

# The index of β in the file (error if it is not there)
function pepo_index(f, β)
    i = findfirst(b -> isapprox(b, β; atol = 1.0e-10), f["beta"])
    i === nothing && error("β = $β is not in the PEPO file (D_CE = $(f["D_CE"]), β = $(f["beta"])): add it to BETAS of development/initial_pepos/ising_model_3D_CE_pepos.jl and run it again")
    return i
end

# The 3D PEPOs at the β of β_obs (as TensorMaps; in the ClusterExpansions environment)
function load_pepos(β_obs, Δβ, D_CE, J, g; kind = :ce)
    f = read_pepo_file(D_CE, Δβ, J, g; kind)
    return [unpack_tensor(f["pepo3d"][pepo_index(f, β)]) for β in β_obs]
end

# The EXACT 3D PEPO of the classical Ising model at g = 0 (added 21 Nov, to separate the CE error from the contraction error): all terms commute, so
# e^{-βH} = Π_bonds e^{βJ sᵢsⱼ} exactly. We split every bond weight W[s, t] = e^{βJ s t} as W = Q·Qᵀ with Q = [√cosh(βJ) √sinh(βJ); √cosh(βJ) −√sinh(βJ)]
# (rows s = +1, −1) and put one Q on each of the six legs of a site:
#     O[p, p'; U, N, E, D, S, W] = δ_{p p'} Q[p, U] Q[p, N] Q[p, E] Q[p, D] Q[p, S] Q[p, W].
# Bond dimension 2, real, cube-symmetric; no Δβ, no CE order, no truncation. The leg spaces are those of the CE PEPOs ((P ⊗ P') ← (U, N, E, D, S, W),
# U/N/E non-dual, D/S/W dual, P non-dual, P' dual; checked against the D_CE = 2 file), so it replaces a CE PEPO of D_CE = 2 everywhere.
function exact_ising_pepo(β; J = 1.0)
    β * J >= 0 || error("exact_ising_pepo: only βJ ≥ 0 (ferromagnet; for βJ < 0, √sinh(βJ) is not real)")
    c, s = sqrt(cosh(β * J)), sqrt(sinh(β * J))
    Q = [c s; c -s]
    a = zeros(Float64, 2, 2, 2, 2, 2, 2, 2, 2)
    for p in 1:2, u in 1:2, n in 1:2, e in 1:2, d in 1:2, so in 1:2, w in 1:2
        a[p, p, u, n, e, d, so, w] = Q[p, u] * Q[p, n] * Q[p, e] * Q[p, d] * Q[p, so] * Q[p, w]
    end
    V = ComplexSpace(2)
    return TensorMap(a, (V ⊗ V') ← (V ⊗ V ⊗ V ⊗ V' ⊗ V' ⊗ V'))
end

# ======================================================================================================================
# CE: the PEPOs for all β, built once (stage 1; moved here from development/profiling/tools/compare_tools.jl on 22 Nov)
# ======================================================================================================================
# CE: the PEPOs at all β of the list β_obs (increasing, multiples of Δβ_iter), built once; stage 1 saves them, so every compare script uses exactly these.
function build_pepos(β_obs, Δβ_iter, D, J, g)
    @assert issorted(β_obs) "β_obs must be increasing"
    trunc_alg = NoEnvTruncation(truncrank(D))
    ce_alg = ising_operators(J, g, 0.0; T = BigFloat, symmetry = nothing)
    ΔO = ClusterExpansions.evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64)
    O = ClusterExpansions.evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)
    Os = Any[]
    β_now = 0.0
    for β in β_obs
        # number of time steps from the previous β to this one
        nsteps = round(Int, (β - β_now) / Δβ_iter)
        @assert abs(nsteps - (β - β_now) / Δβ_iter) < 1.0e-8 "β = $β is not a multiple of Δβ_iter = $Δβ_iter"
        for _ in 1:nsteps
            # rot_invariance = true: one Takagi projector pair for all six legs, so every O is exactly symmetric under the cube rotations and
            # reflections (needed by the C4v CTMRG of vbPEPS); it stays Float64 (see takagi_trunc)
            O, = approximate_state((O, ΔO), trunc_alg; rot_invariance = true)
        end
        β_now = β
        push!(Os, deepcopy(O))   # copy, so that later steps cannot change an earlier O
    end
    return Os
end

# The 2D tensors of the vbPEPS at the β of β_obs, packed: [(; O, M, psi0)] with the start state of this chipeps.
# maxiter_init must be the number of power-method steps of the file (the start state depends on it).
function load_bpeps_tensors(β_obs, Δβ, D_CE, J, g, chipeps; maxiter_init, kind = :ce)
    f = read_pepo_file(D_CE, Δβ, J, g; kind)
    k = findfirst(==(chipeps), f["chipeps"])
    k === nothing && error("chipeps = $chipeps has no start states in the PEPO file (chipeps = $(f["chipeps"])): add it to CHIPEPS of development/initial_pepos/ising_model_3D_CE_pepos.jl and run it again")
    f["maxiter_init"] == maxiter_init || error("the start states of the PEPO file have maxiter_init = $(f["maxiter_init"]), not $maxiter_init: run development/initial_pepos/ising_model_3D_CE_pepos.jl again")
    return [(; O = f["O"][i], M = f["M"][i], psi0 = f["psi0"][k][i]) for i in (pepo_index(f, β) for β in β_obs)]
end
