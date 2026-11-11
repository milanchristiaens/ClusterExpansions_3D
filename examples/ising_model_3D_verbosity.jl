using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using TimerOutputs
using Random
using LinearAlgebra

Random.seed!(123456789)
setprecision(128)

T = Float64
spin_symmetry = Trivial
J = 1.0
g = 0.0

D = 4          # PEPO bond dimension (truncation in approximate_state)
chipeps = 4    # CTMRG3D bond dimension towards the faces
chienv = 6     # CTMRG3D bond dimension of corners/edges
pepsspace = ℂ^chipeps
envspace = ℂ^chienv

obs = SpinOperators.σᶻ(T, spin_symmetry)

Δβ_iter = 0.005
# Around the critical point βc ≈ 0.2217 (g = 0) CTMRG should converge slowly.
# Each value must be a multiple of Δβ_iter.
β_obs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

ce_verbosity = 2         # 2D cluster expansion
ctm_verbosity = 3        # 1: step lines, 2: + one line per side, 3: + one line per West update (very long)
trunc_verbosity = 0      # Not needed (info already printed, see line '        k % 10 == 0 && @info "  time step $k/$nsteps: truncation error = $ϵ, bond dimension = $(dim(domain(O)[1]))"')
ram_gib = 4              # nothing = old unsliced contract_onesite; a small number makes the planner cut

ctm_miniter = 3
ctm_maxiter = 150
ctm_tol = 1e-7  

trunc_alg = NoEnvTruncation(truncrank(D); verbosity = trunc_verbosity)

ce_alg = ising_operators(J, g, 0.0; T = BigFloat, symmetry = "Oh", verbosity = ce_verbosity)

reset_timer!(ClusterExpansions.to)

@info "Cluster expansion for ΔO (2D CE, embedded in 3D)"
ΔO = evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64)
O = evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)   # β = 0: identity, no CE solve

βs_all = [0.0; β_obs]    # start point + observation points
ms = []
for i in eachindex(β_obs)
    Δ = βs_all[i+1] - βs_all[i]
    nsteps = round(Int, Δ / Δβ_iter)
    @assert abs(nsteps - Δ / Δβ_iter) < 1e-10 "β_obs values must be multiples of Δβ_iter"

    @info "Observation $i / $(length(β_obs)) at β = $(β_obs[i]): $nsteps time steps of Δβ = $Δβ_iter"
    for k in 1:nsteps
        global O
        O, ϵ = approximate_state((O, ΔO), trunc_alg; rot_invariance = false)
        k % 10 == 0 && @info "  time step $k/$nsteps: truncation error = $ϵ, bond dimension = $(dim(domain(O)[1]))"
    end
    m, = MPSKit.expectation_value(O, [obs], pepsspace, envspace;
                                  miniter = ctm_miniter, maxiter = ctm_maxiter, tol = ctm_tol,
                                  verbosity = ctm_verbosity, ram_gib, west_updates = :auto)
    push!(ms, m)
    @info "β = $(β_obs[i]): magnetization = $m"
end

print_timer(ClusterExpansions.to; sortby = :time)