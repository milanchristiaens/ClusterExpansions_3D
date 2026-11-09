# Profiling of the 3D Ising simulation with CE + VARIATIONAL BOUNDARY PEPS (vbPEPS), the counterpart of ising_model_3D_timing.jl (CE + CTMRG3D).
#
# Same physics, same default parameters, same cluster-expansion (CE) time evolution and the same timers/plots as ising_model_3D_timing.jl;
# the only difference is the observable: instead of MPSKit.expectation_value (CTMRG3D, 27 tensors) every observation is
# bpeps_expectation_value (observables_cubic_bpeps.jl): LBFGS optimization of a boundary PEPS with a 2D CTMRG of PEPSKit, then the observable.
# The nested timers of the vbPEPS part are called "bPEPS: ..." (initial state, LBFGS > cost function > n2 / n3 leading_boundary (the two 2D CTMRG contractions), forward pass, backward pass, ...).
#
# Settings to decide:
#   * warm_start: false = every observation starts from the power-method PEPS (like the run script of Sander); true = it starts from the optimized
#     PEPS and environments of the previous β (faster, but the timing is then not the cost of one independent observation).
#   * boundary_alg_type / symmetrization: SimultaneousCTMRG + nothing (works for any PEPO) or C4vCTMRG + RotateReflect() (needs a C4v symmetric PEPO,
#     for example from the CE with symmetry = "Oh"; the CE here uses symmetry = nothing, as in ising_model_3D_timing.jl).
#   * the LBFGS settings (bpeps_kwargs):
#   * differentiate_environment: false (default) = gradient with fixed environments (cheap,);
#     true = gradient through the CTMRG with hook_pullback (more expensive).

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using TimerOutputs
using Plots
using Random
using LinearAlgebra
BLAS.set_num_threads(1)

Random.seed!(123456789)

const to = ClusterExpansions.to # the timer that the package code writes to

# This function performs the evolution and the observations of CE+Vbeps
function evolve_CE_vbpeps_timed(obs, Δβ_iter, Δβ_obs, maxiter_obs, D, chipeps, chienv; J = 1.0, g = 0.0, spin_symmetry = Trivial, warm_start = false, bpeps_kwargs = (;))
    # We first build the virtual spaces
    if spin_symmetry == Trivial
        pepsspace = ℂ^chipeps
        envspace = ℂ^chienv
    else
        pepsspace = Vect[Z2Irrep](0 => chipeps - div(chipeps, 2), 1 => div(chipeps, 2))
        envspace = Vect[Z2Irrep](0 => chienv - div(chienv, 2), 1 => div(chienv, 2))
    end

    # time steps per observation
    nsteps = round(Int, Δβ_obs / Δβ_iter)
    @assert abs(nsteps - Δβ_obs / Δβ_iter) < 1e-10 "Δβ_obs and Δβ_iter are incomptatible"

    trunc_alg = NoEnvTruncation(truncrank(D))

    # The cluster expansion part
    ce_alg = ClusterExpansions.@timeit to "CE: ising_operators (setup)" ising_operators(J, g, 0.0; T = BigFloat, symmetry = nothing)
    ΔO = ClusterExpansions.@timeit to "CE: evolution_operator_cubic for ΔO" ClusterExpansions.evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64)
    O = ClusterExpansions.@timeit to "CE: evolution_operator_cubic for O at beta=0" ClusterExpansions.evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)

    # Collect the observables
    ms = []
    # holds the warm start
    state = nothing
    # Loop over all observations
    for i = 1:maxiter_obs
        # First perform the time evolution
        ClusterExpansions.@timeit to "CE: time evolution (nsteps x approximate_state)" begin
            for _ = 1:nsteps
                O, = ClusterExpansions.@timeit to "approximate_state (stack and truncate)" approximate_state((O, ΔO), trunc_alg; rot_invariance = false)
            end
        end

        # Then perform the observation
        # if warm_start is true, it stores the tuple (ψ, env2, env3) of this result in state
        m = ClusterExpansions.@timeit to "expectation_value (vbPEPS and observable)" begin
            result = ClusterExpansions.bpeps_optimize(O, pepsspace, envspace; state0 = warm_start ? state : nothing, bpeps_kwargs...)
            warm_start && (state = (result.psi, result.env2, result.env3))
            ClusterExpansions.bpeps_observables(O, [obs], result)[1]
        end
        push!(ms, m)
    end
    βs = [Δβ_obs * i for i in 1:maxiter_obs]
    return βs, ms
end

# timer_rows, print_overview and write_timer_csv (read the TimerOutput back, print an overview, write the CSV)
include(joinpath(@__DIR__, "timing_tools.jl"))
# the plot functions
include(joinpath(@__DIR__, "timing_plots.jl"))
outdir = normpath(joinpath(@__DIR__, "..", "..", "data", "profiling"))
mkpath(outdir)

# Parameters
T = Float64
spin_symmetry = Trivial
J = 1.0
g = 0.0

D_CE = 2
chipeps = 2
chienv = 8

obs = SpinOperators.σᶻ(T, spin_symmetry)

Δβ_iter = 0.005
Δβ_obs = 0.05
maxiter_obs = 1

# Settings of the variational boundary PEPS: (boundary_alg_type = SimultaneousCTMRG, symmetrization = nothing).
# Change e.g. to boundary_alg_type = C4vCTMRG, symmetrization = RotateReflect() for a C4v symmetric PEPO.
warm_start = false
bpeps_kwargs = (;
    boundary_alg_type = SimultaneousCTMRG, symmetrization = nothing,
    boundary_tol = 1.0e-8, boundary_maxiter = 300, boundary_verbosity = 1,
    gradient_tol = 1.0e-6, gradient_maxiter = 5, gradient_krylovdim = 20,
    optimizer_memory = 16, optimizer_tol = 1.0e-8, optimizer_maxiter = 30,
    maxiter_init = 10, differentiate_environment = true, optimizer_verbosity = 3,
)

@info "Threads.nthreads() = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())"


# Warm-up: compile everything with tiny settings (not measured)
@info "Warm-up run (compilation), not timed"
tiny = (; bpeps_kwargs..., optimizer_maxiter = 1, maxiter_init = 2)
evolve_CE_vbpeps_timed(obs, Δβ_iter, Δβ_iter, 1, D_CE, chipeps, chienv; J, g, spin_symmetry, warm_start, bpeps_kwargs = tiny)


# Measured run: CE + vbPEPS
reset_timer!(to)
ClusterExpansions.reset_max_alloc!()
βs, ms = ClusterExpansions.@timeit to "TOTAL CE + vbPEPS run" evolve_CE_vbpeps_timed(obs, Δβ_iter, Δβ_obs, maxiter_obs, D_CE, chipeps, chienv; J, g, spin_symmetry, warm_start, bpeps_kwargs)
rows = timer_rows(to)
println("\n===== CE + vbPEPS: overview (depth <= 3) =====")
print_overview(rows; maxdepth = 3)
println("\n===== CE + vbPEPS: full tree (TimerOutputs) =====")
print_timer(to; sortby = :time)
write_timer_csv(joinpath(outdir, "timing_CE_vbpeps.csv"), rows)
@info "Peak memory of the Julia process so far (includes memory outside the GC): $(round(Sys.maxrss() / 2^20; digits = 1)) MiB"
plt_phases = plot_phases(rows, "CE + vbPEPS")
plt_self = plot_self_time(rows, "CE + vbPEPS")
plt_bpeps = plot_self_time(rows, "CE + vbPEPS"; under = "bPEPS", top = 25)      # only the sections of the variational boundary PEPS
display(plt_phases)
display(plt_self)
display(plt_bpeps)
savefig(plt_phases, joinpath(outdir, "timing_CE_vbpeps_phases.png"))
savefig(plt_self, joinpath(outdir, "timing_CE_vbpeps_self.png"))
savefig(plt_bpeps, joinpath(outdir, "timing_CE_vbpeps_bpeps.png"))
