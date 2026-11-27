# The models of the BP comparison (the transverse-field Ising model at g ≠ 0 and the Bose-Hubbard model) and their PEPO files.
# Every model has its cluster expansion, its two observables and its β list; the CE PEPOs are evolved once to all β (build_model_pepos) and saved
# by models_3D_CE_pepos.jl, and the compare runs read them (load_model_pepos).

const MODEL_PEPO_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "pepos"))

# kind :ising: H = -J Σ σᶻσᶻ - g J Σ σˣ (ising_operators); observables σᶻ and σˣ.
# kind :bose_hubbard: H = -t Σ (b†b + h.c.) + U/2 Σ n(n-1) - μ Σ n + δ Σ (b† + b) with at most `cutoff` bosons per site (bose_hubbard_operators); observables n and b
# (ρ = ⟨n⟩, ρs = |⟨b⟩|²). Its O is real but not cube symmetric (the two legs of a hopping bond carry b† and b), so CTMRG3DOctant cannot run it.
# kind :xy: hard-core bosons at half filling (μ = 0) written as spins: b† = S⁺, so H = -2t Σ (SˣSˣ + SʸSʸ) + 2δ Σ Sˣ; we rotate the axes (x, y, z) -> (z, x, y),
# which gives H = -2t Σ (SᶻSᶻ + SˣSˣ) + 2δ Σ Sᶻ (heisenberg_operators): real and cube symmetric. Observables Sᶻ = Re⟨b⟩ and Sˣ = -Im⟨b⟩ (ρs = ⟨Sᶻ⟩² + ⟨Sˣ⟩²; ρ = 1/2).
# beta_c: our estimate of the true βc (the warm rule starts cold near it): for the Ising models a crossing of the Binder cumulant of tfim_3d_qmc.jl at L = 8 and 16,
# for bh_b the QMC value Tc = 2.0169 t (Carrasquilla and Rigol, PRA 86, 043629 (2012)), for bh_a that value for the effective hard-core bosons between
# n = 1 and n = 2 (hopping 2t at μ = U, corrections of order t/U); betas: the β of the observations (multiples of dbeta), from just below the BP βc to above beta_c;
# vbpeps: we also save the 2D tensors and start states of the vbPEPS (its C4v CTMRG needs a cube-symmetric O, so not for bh_a)
const MODELS = Dict(
    "tfim_g2"   => (; kind = :ising, J = 1.0, g = 2.0, D_CE = 3, dbeta = 0.005, beta_c = 0.234, vbpeps = true,
                    betas = [0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.25, 0.27]),
    "tfim_g3.5" => (; kind = :ising, J = 1.0, g = 3.5, D_CE = 3, dbeta = 0.005, beta_c = 0.285, vbpeps = true,
                    betas = [0.24, 0.25, 0.26, 0.27, 0.275, 0.28, 0.285, 0.29, 0.30, 0.33]),
    "tfim_g4.5" => (; kind = :ising, J = 1.0, g = 4.5, D_CE = 3, dbeta = 0.005, beta_c = 0.378, vbpeps = true,
                    betas = [0.30, 0.31, 0.32, 0.34, 0.36, 0.37, 0.38, 0.39, 0.41, 0.45]),
    "bh_a"      => (; kind = :bose_hubbard, t = 1.0, U = 40.0, V = 0.0, μ = 40.0, δ = 0.002, cutoff = 2, D_CE = 4, dbeta = 0.005, beta_c = 0.248, vbpeps = false,
                    betas = [0.19, 0.195, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.30]),
    "bh_b"      => (; kind = :xy, t = 1.0, δ = 0.002, D_CE = 3, dbeta = 0.005, beta_c = 0.496, vbpeps = true,
                    betas = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.53]),
)

model_settings(name) = haskey(MODELS, name) ? MODELS[name] : error("unknown model $name: the models are $(sort(collect(keys(MODELS))))")

function model_ce(m)
    m.kind === :ising && return ising_operators(m.J, m.g, 0.0; T = BigFloat, symmetry = nothing)
    m.kind === :bose_hubbard && return bose_hubbard_operators(m.t, m.U, m.V, m.μ; δ = m.δ, cutoff = m.cutoff, T = BigFloat, symmetry = nothing)
    m.kind === :xy && return heisenberg_operators(-2 * m.t, 0.0, -2 * m.t, 2 * m.δ; T = BigFloat, symmetry = nothing)
    error("unknown model kind $(m.kind)")
end

model_observables(m) = m.kind === :ising ? [SpinOperators.σᶻ(Float64, Trivial), SpinOperators.σˣ(Float64, Trivial)] :
                       m.kind === :xy ? [SpinOperators.S_z(Float64, Trivial), SpinOperators.S_x(Float64, Trivial)] :
                       [BosonOperators.n(Float64, Trivial; cutoff = m.cutoff), BosonOperators.b⁻(Float64, Trivial; cutoff = m.cutoff)]
model_observable_names(m) = m.kind === :ising ? ("sz", "sx") : m.kind === :xy ? ("Sz", "Sx") : ("n", "b")

model_pepo_file(name) = (m = model_settings(name); joinpath(MODEL_PEPO_DIR, "model_3D_$(name)_D$(m.D_CE)_dbeta$(m.dbeta).jld2"))

# The CE PEPOs of the model at all its β, as build_pepos (rot_invariance = true: exactly cube symmetric)
function build_model_pepos(m)
    trunc_alg = NoEnvTruncation(truncrank(m.D_CE))
    ce_alg = model_ce(m)
    ΔO = ClusterExpansions.evolution_operator_cubic(ce_alg, m.dbeta; T_conv = Float64)
    O = ClusterExpansions.evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)
    Os, β_now = Any[], 0.0
    for β in m.betas
        nsteps = round(Int, (β - β_now) / m.dbeta)
        @assert abs(nsteps - (β - β_now) / m.dbeta) < 1.0e-8 "β = $β is not a multiple of dbeta = $(m.dbeta)"
        for _ in 1:nsteps
            O, = approximate_state((O, ΔO), trunc_alg; rot_invariance = true)
        end
        β_now = β
        push!(Os, deepcopy(O))
    end
    return Os
end

# The PEPOs of the model at the β of β_obs (as TensorMaps)
function load_model_pepos(name, β_obs)
    path = model_pepo_file(name)
    isfile(path) || error("No PEPO file $path: run development/initial_pepos/models_3D_CE_pepos.jl first (with CE_MODELS = $name)")
    f = JLD2.load(path)
    idx(β) = (i = findfirst(b -> isapprox(b, β; atol = 1.0e-10), f["beta"]); i === nothing ? error("β = $β is not in $path ($(f["beta"]))") : i)
    return [unpack_tensor(f["pepo3d"][idx(β)]) for β in β_obs]
end

# The 2D tensors of the vbPEPS at the β of β_obs, packed: [(; O, M = [M1, M2], psi0)] with the start state of this chipeps
function load_model_bpeps_tensors(name, β_obs, chipeps; maxiter_init)
    path = model_pepo_file(name)
    f = isfile(path) ? JLD2.load(path) : error("No PEPO file $path: run development/initial_pepos/models_3D_CE_pepos.jl first (with CE_MODELS = $name)")
    haskey(f, "O") || error("$path has no vbPEPS tensors (vbpeps = false for $name, or the file is older than this script)")
    k = findfirst(==(chipeps), f["chipeps"])
    k === nothing && error("chipeps = $chipeps has no start states in $path (chipeps = $(f["chipeps"])): add it to CHIPEPS of models_3D_CE_pepos.jl and run it again")
    f["maxiter_init"] == maxiter_init || error("the start states of $path have maxiter_init = $(f["maxiter_init"]), not $maxiter_init")
    idx(β) = (i = findfirst(b -> isapprox(b, β; atol = 1.0e-10), f["beta"]); i === nothing ? error("β = $β is not in $path ($(f["beta"]))") : i)
    return [(; O = f["O"][i], M = f["M"][i], psi0 = f["psi0"][k][i]) for i in map(idx, β_obs)]
end
