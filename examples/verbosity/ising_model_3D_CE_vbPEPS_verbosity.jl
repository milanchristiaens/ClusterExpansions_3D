# Verbose run of the 3D Ising model (g = 0) with CE + VARIATIONAL BOUNDARY PEPS (vbPEPS): prints the cluster expansion, the truncation,
# the LBFGS iterations, the 2D CTMRG runs of the boundary PEPS and the timer tree.
# Use this to watch how the optimization of the boundary PEPS converges.
# The same run with CTMRG3D: ising_model_3D_CE_CTMRG_verbosity.jl (this folder).

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using TimerOutputs
using Random
using LinearAlgebra
BLAS.set_num_threads(1)

Random.seed!(123456789)

## Settings (the same names and the same β grid in all 3D examples)
T = Float64
spin_symmetry = Trivial
J = 1.0
g = 0.0

Δβ_iter = 0.005
# Each value must be a multiple of Δβ_iter.
β_obs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

D_CE = 2       # PEPO bond dimension (truncation in approximate_state)
chipeps = 2    # bond dimension of the boundary PEPS
chienv = 8     # bond dimension of the 2D CTMRG environment of the boundary PEPS
pepsspace = ℂ^chipeps
envspace = ℂ^chienv

obs = SpinOperators.σᶻ(T, spin_symmetry)

## Settings of the output
ce_verbosity = 2         # 2D cluster expansion
trunc_verbosity = 0      # Not needed (info already printed, see line '        k % 10 == 0 && @info "  time step $k/$nsteps: truncation error = $ϵ, bond dimension = $(dim(domain(O)[1]))"')
# boundary_verbosity (2D CTMRG of PEPSKit) and optimizer_verbosity (LBFGS of OptimKit) are in bpeps_kwargs below

## Settings of the variational boundary PEPS (the same as in ../profiling/ising_model_3D_CE_vbPEPS_profiling.jl)
# warm_start: false = every observation starts from the power-method PEPS; true = it starts from the optimized PEPS and environments of the previous β
warm_start = false
bpeps_kwargs = (;
    boundary_alg_type = SimultaneousCTMRG, symmetrization = nothing,
    boundary_tol = 1.0e-8, boundary_maxiter = 400, boundary_verbosity = 2,      # PEPSKit: 1 = warnings, 2 = + convergence info, 3 = + every iteration (not checked)
    gradient_tol = 1.0e-6, gradient_maxiter = 5, gradient_krylovdim = 20,
    optimizer_memory = 64, optimizer_tol = 1.0e-5, optimizer_maxiter = 10,
    maxiter_init = 20, differentiate_environment = true,    # gradient through the CTMRG (hook_pullback); false = fixed environments, much cheaper to compile
    optimizer_verbosity = 2,
)

trunc_alg = NoEnvTruncation(truncrank(D_CE); verbosity = trunc_verbosity)

ce_alg = ising_operators(J, g, 0.0; T = BigFloat, symmetry = nothing, verbosity = ce_verbosity)

reset_timer!(ClusterExpansions.to)

@info "Threads.nthreads() = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())"

@info "Cluster expansion for ΔO (2D CE, embedded in 3D)"
ΔO = evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64)
O = evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)   # β = 0: identity, no CE solve

βs_all = [0.0; β_obs]    # start point + observation points
ms = []
state = nothing          # (psi, env2, env3) of the previous observation, only used if warm_start
for i in eachindex(β_obs)
    global state
    Δ = βs_all[i+1] - βs_all[i]
    nsteps = round(Int, Δ / Δβ_iter)
    @assert abs(nsteps - Δ / Δβ_iter) < 1e-10 "β_obs values must be multiples of Δβ_iter"

    @info "Observation $i / $(length(β_obs)) at β = $(β_obs[i]): $nsteps time steps of Δβ = $Δβ_iter"
    for k in 1:nsteps
        global O
        O, ϵ = approximate_state((O, ΔO), trunc_alg; rot_invariance = false)
        k % 10 == 0 && @info "  time step $k/$nsteps: truncation error = $ϵ, bond dimension = $(dim(domain(O)[1]))"
    end

    # Optimize the boundary PEPS, then measure the observable on it
    t_obs = time()
    result = ClusterExpansions.bpeps_optimize(O, pepsspace, envspace; state0 = warm_start ? state : nothing, bpeps_kwargs...)
    m = ClusterExpansions.bpeps_observables(O, [obs], result)[1]
    warm_start && (state = (result.psi, result.env2, result.env3))
    push!(ms, m)
    @info "β = $(β_obs[i]): magnetization = $m, f = $(result.cost), cost-function calls = $(result.numfg), time = $(round(time() - t_obs; digits = 1)) s"
end

print_timer(ClusterExpansions.to; sortby = :time)
