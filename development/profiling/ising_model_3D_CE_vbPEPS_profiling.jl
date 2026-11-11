# Profiling of the 3D Ising simulation with CE + VARIATIONAL BOUNDARY PEPS (vbPEPS) (g = 0), the counterpart of ising_model_3D_CE_CTMRG_profiling.jl (CE + CTMRG3D).
# Comparing several settings or algorithms: comparing/.
#
# Same physics, same β grid, same cluster-expansion (CE) time evolution and the same timers/plots as ising_model_3D_CE_CTMRG_profiling.jl;
# the only difference is the observable: instead of MPSKit.expectation_value (CTMRG3D, 27 tensors) every observation is
# bpeps_expectation_value (observables_cubic_bpeps.jl): LBFGS optimization of a boundary PEPS with a 2D CTMRG of PEPSKit, then the observable.
# The nested timers of the vbPEPS part are called "bPEPS: ..." (initial state, LBFGS > cost function > n2 / n3 leading_boundary (the two 2D CTMRG contractions), forward pass, backward pass, ...).
#
# Where the data goes: a new run folder data/profiling/ising_model_3D_CE_vbPEPS_profiling/<date>_<time>/ (see tools/timing_io.jl).
# With REPLOT_DIR (see the settings) the plots of an earlier run are made again from its CSV file, without running anything.
#
# Settings to decide:
#   * warm_start: false = every observation starts from the power-method PEPS (like the run script of Sander); true = it starts from the optimized
#     PEPS and environments of the previous β (faster, but the timing is then not the cost of one independent observation).
#   * boundary_alg_type / symmetrization: SimultaneousCTMRG + nothing (works for any PEPO) or C4vCTMRG + RotateReflect() (needs a C4v symmetric PEPO,
#     for example from the CE with symmetry = "Oh"; the CE here uses symmetry = nothing, as in ising_model_3D_CE_CTMRG_profiling.jl).
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

## Settings (the same names and the same β grid in all 3D examples)
T = Float64
spin_symmetry = Trivial
J = 1.0
g = 0.0

Δβ_iter = 0.005
# Each value must be a multiple of Δβ_iter.
β_obs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

D_CE = 2
chipeps = 2
chienv = 8

obs = SpinOperators.σᶻ(T, spin_symmetry)

# Settings of the variational boundary PEPS: (boundary_alg_type = SimultaneousCTMRG, symmetrization = nothing).
# Change e.g. to boundary_alg_type = C4vCTMRG, symmetrization = RotateReflect() for a C4v symmetric PEPO.
warm_start = false          # every observation is independent, so the timing is the cost of one observation
bpeps_kwargs = (;
    boundary_alg_type = SimultaneousCTMRG, symmetrization = nothing,
    boundary_tol = 1.0e-8, boundary_maxiter = 400, boundary_verbosity = 1,
    gradient_tol = 1.0e-6, gradient_maxiter = 5, gradient_krylovdim = 20,
    optimizer_memory = 64, optimizer_tol = 1.0e-5, optimizer_maxiter = 10,
    maxiter_init = 20, differentiate_environment = true,    # gradient through the CTMRG (hook_pullback); the first run (false) is in data/profiling/ising_model_3D_CE_vbPEPS_profiling/false_cold
    optimizer_verbosity = 2,
)

# If REPLOT_DIR is the NAME of a run folder (for example "false_cold" or "20261111_1248") in data/profiling/ising_model_3D_CE_vbPEPS_profiling/, we only make the plots again
# from its timing_CE_vbpeps.csv and nothing else is run (no warm-up, no calculation, no compilation of the pullback). Leave it "" for a normal run.
REPLOT_DIR = ""


# This function performs the evolution and the observations of CE+Vbeps
function evolve_CE_vbpeps_timed(obs, β_obs, Δβ_iter, D, chipeps, chienv; J = 1.0, g = 0.0, spin_symmetry = Trivial, warm_start = false, bpeps_kwargs = (;))
    # We first build the virtual spaces
    if spin_symmetry == Trivial
        pepsspace = ℂ^chipeps
        envspace = ℂ^chienv
    else
        pepsspace = Vect[Z2Irrep](0 => chipeps - div(chipeps, 2), 1 => div(chipeps, 2))
        envspace = Vect[Z2Irrep](0 => chienv - div(chienv, 2), 1 => div(chienv, 2))
    end

    trunc_alg = NoEnvTruncation(truncrank(D))

    # The cluster expansion part
    ce_alg = ClusterExpansions.@timeit to "CE: ising_operators (setup)" ising_operators(J, g, 0.0; T = BigFloat, symmetry = nothing)
    ΔO = ClusterExpansions.@timeit to "CE: evolution_operator_cubic for ΔO" ClusterExpansions.evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64)
    O = ClusterExpansions.@timeit to "CE: evolution_operator_cubic for O at beta=0" ClusterExpansions.evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)

    # Collect the observables
    βs_all = [0.0; β_obs]    # start point + observation points
    ms = []
    # holds the warm start
    state = nothing
    # Loop over all observations
    for i in eachindex(β_obs)
        # time steps from the previous β to this one
        Δ = βs_all[i+1] - βs_all[i]
        nsteps = round(Int, Δ / Δβ_iter)
        @assert abs(nsteps - Δ / Δβ_iter) < 1e-10 "β_obs values must be multiples of Δβ_iter"
        # First perform the time evolution
        ClusterExpansions.@timeit to "CE: time evolution (nsteps x approximate_state)" begin
            for _ = 1:nsteps
                O, = ClusterExpansions.@timeit to "approximate_state (stack and truncate)" approximate_state((O, ΔO), trunc_alg; rot_invariance = false)
            end
        end

        # Then perform the observation
        # if warm_start is true, it stores the tuple (ψ, env2, env3) of this result in state
        t_obs = time()
        m = ClusterExpansions.@timeit to "expectation_value (vbPEPS and observable)" begin
            result = ClusterExpansions.bpeps_optimize(O, pepsspace, envspace; state0 = warm_start ? state : nothing, bpeps_kwargs...)
            # one line per observation: β, final cost, number of cost-function calls of LBFGS, time of the optimization and the sanity check
            println("OBS β = $(round(β_obs[i]; digits = 3)): f = $(result.cost), cost-function calls = $(result.numfg), time = $(round(time() - t_obs; digits = 1)) s, ", check_cost(β_obs[i], result.cost; reference = true))
            warm_start && (state = (result.psi, result.env2, result.env3))
            ClusterExpansions.bpeps_observables(O, [obs], result)[1]
        end
        push!(ms, m)
    end
    return β_obs, ms
end


# timer_rows, print_overview, write_timer_csv, read_timer_csv and the data folder functions (new_run_dir, existing_run_dir)
include(joinpath(@__DIR__, "tools", "timing_io.jl"))
# the plot functions (make_timing_plots, ...)
include(joinpath(@__DIR__, "tools", "timing_plots.jl"))
# check_cost (sanity check of the final cost of an observation)
include(joinpath(@__DIR__, "tools", "compare_tools.jl"))


if !isempty(REPLOT_DIR)
    ## Only plot: read the CSV file of an earlier run
    outdir = existing_run_dir(@__FILE__, REPLOT_DIR)
    @info "Replot only: reading the CSV file in $outdir"
    rows = read_timer_csv(joinpath(outdir, "timing_CE_vbpeps.csv"))
    make_timing_plots(rows, "CE + vbPEPS", outdir, "CE_vbpeps"; under = "bPEPS", under_name = "bpeps")
else
    outdir = new_run_dir(@__FILE__)
    @info "Output folder: $outdir"

    @info "Threads.nthreads() = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())"


    # Warm-up: compile everything with tiny settings (not measured)
    @info "Warm-up run (compilation), not timed"
    tiny = (; bpeps_kwargs..., optimizer_maxiter = 1, maxiter_init = 2)
    evolve_CE_vbpeps_timed(obs, [Δβ_iter], Δβ_iter, D_CE, chipeps, chienv; J, g, spin_symmetry, warm_start, bpeps_kwargs = tiny)


    # Measured run: CE + vbPEPS
    reset_timer!(to)
    ClusterExpansions.reset_max_alloc!()
    βs, ms = ClusterExpansions.@timeit to "TOTAL CE + vbPEPS run" evolve_CE_vbpeps_timed(obs, β_obs, Δβ_iter, D_CE, chipeps, chienv; J, g, spin_symmetry, warm_start, bpeps_kwargs)
    rows = timer_rows(to)
    println("\n===== CE + vbPEPS: overview (depth <= 3) =====")
    print_overview(rows; maxdepth = 3)
    println("\n===== CE + vbPEPS: full tree (TimerOutputs) =====")
    print_timer(to; sortby = :time)
    write_timer_csv(joinpath(outdir, "timing_CE_vbpeps.csv"), rows)
    @info "Peak memory of the Julia process so far (includes memory outside the GC): $(round(Sys.maxrss() / 2^20; digits = 1)) MiB"
    make_timing_plots(rows, "CE + vbPEPS", outdir, "CE_vbpeps"; under = "bPEPS", under_name = "bpeps")      # under: only the sections of the variational boundary PEPS
end
