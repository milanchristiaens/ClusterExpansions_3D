# Comparison of the vbPEPS cases for the 3D Ising model (CE + variational boundary PEPS), g = 0:
#
#     start (cold / warm), with the fixed environment (differentiate_environment = false)
#
# The extra term of the gradient (differentiate_environment = true) is not part of the compare any more: development/checks/check_vbpeps_diff_env.jl
# checks that it gives the same result. (The table CASES below can still hold true cases for a test.)
# warm = every β starts from the optimized state of the previous β, the first β is always cold, and a β closer than WARM_WINDOW to BETA_C also starts cold
#        (warm_start_at in ../tools/compare_tools.jl); the check column of the results file says "cold start near βc" there.
#
# Everything uses the same CE PEPOs (read from the PEPO file of ising_model_3D_CE_pepos.jl, which must be run first; the CTMRG3D compare reads the same
# file) and the same random seed per observation (Random.seed!(SEED_BASE + i) before observation i),
# so the cases differ only in the two settings above. For every case and every β we store: final cost f, number of cost-function calls, time, magnetization m.
# Each observation is appended (and the file closed) immediately to the results file of all Ising compare runs,
# data/profiling/comparing/ising_model_3D_CE_results.csv (algorithm = "vbPEPS", run = name of the run folder; see ISING_CSV_HEADER in ../tools/compare_tools.jl),
# so a crash halfway keeps all earlier data.
# The other files go to a new run folder data/profiling/comparing/ising_model_3D_CE_vbPEPS_compare/<date>_<time>/ (or <CE_COMPARE_RUN>/; see tools/timing_io.jl) :
#     settings.txt, timing_<case>.csv (TimerOutputs rows of each case),
#     compare_vbpeps_magnetization.png (left: |m| against β; right: |m| of every case minus the mean of the cases),
#     compare_vbpeps_cost_time.png (left: time per observation against β; right: cost f against β),
#     compare_vbpeps_timing_phases.png and compare_vbpeps_timing_self.png (timings per section, 4 bars per section)
#
# With REPLOT_DIR (see the settings) we only redo all plots from the results of an earlier run, without running anything.
# The helper functions (check_cost, results file, time limit, spread plot, timing plots, PEPO file) are in ../tools/ (compare_tools.jl, timing_io.jl,
# timing_plots.jl, pepo_files.jl).
#
# Order: the order of CASES (a true case would need the compilation of its pullback first, ~1h40 with :pepskit in an earlier run).
# Compilation happens once per process (warm-up with tiny settings, not measured).
#
# Time limit: every case gets RUN_BUDGET_S seconds in total and every observation at most MAX_SECONDS_OBS seconds (see the settings, and obs_seconds_allowed
# in ../tools/compare_tools.jl; the CTMRG3D compare script uses the same rule).
# An observation that runs out of time stops after its current LBFGS iteration and is marked timed_out = true (a cross in the plot):
# its f and m are then the values of the last iterate, not of a converged optimization.
#
# Note on m: at g = 0 the exact <σᶻ> is 0 below βc (Z2 symmetry) and has an arbitrary sign above βc, so we plot |m|. A nonzero m below βc is a contraction/optimization artefact.
#
# Two implementations of vbPEPS (VBPEPS_IMPL in the settings; the column setting of the results file ends with it, for example "fixed_env_betterpepskit"),
# each in its own environment (BetterPEPSKit cannot be loaded together with ClusterExpansions: it needs MPSKit 0.14 and OptimKit 0.5):
#   :pepskit        ClusterExpansions.bpeps_optimize / bpeps_observables (PEPSKit 0.8.1): dense eigh/SVD in the CTMRG, FixedPointGradient with Arnoldi.
#                   Run in the ClusterExpansions environment:          julia -t 4 --project=. <this file>
#   :betterpepskit  the same algorithm on BetterPEPSKit (Lander Burgelman e.a.): block-Krylov decompositions, gap cut of the spectrum, ImplicitGradient
#                   with GMRES, a start environment grown from a product state (see the top of development/fast_vbpeps/fast_bpeps.jl).
#                   Run in the BetterPEPSKit environment:              julia -t 4 --project=development/fast_vbpeps <this file>
#                   (needs the BetterPEPSKit clone next to this clone, see development/fast_vbpeps/README.md). This environment has no Plots: at the end
#                   we print the summary, and the plots are made afterwards with REPLOT_DIR in the ClusterExpansions environment.
# Both start from the same state: the start states in the PEPO file are made exactly as bpeps_optimize makes its own (same seed psi0_seed(β, Δβ)).
# The environment decides which packages are loaded (IN_FAST_ENV below); VBPEPS_IMPL must fit it (otherwise an error at the start). REPLOT_DIR needs the
# ClusterExpansions environment (Plots).

ENV["GKSwstype"] = "nul"     # no plot window: the plots are only saved to a file

# Which environment: development/fast_vbpeps (BetterPEPSKit) or the ClusterExpansions environment of the clone
const IN_FAST_ENV = basename(dirname(Base.active_project())) == "fast_vbpeps"

using TensorKit
using TimerOutputs
using Random
using LinearAlgebra
using JLD2
if IN_FAST_ENV
    # the vbPEPS of BetterPEPSKit (module FastBpeps); C4vCTMRG and RotateReflect below come from BetterPEPSKit
    include(joinpath(@__DIR__, "..", "..", "fast_vbpeps", "fast_bpeps.jl"))
    using BetterPEPSKit
    const to = FastBpeps.to           # the timer of FastBpeps
else
    using TensorKitTensors
    using MPSKit
    using PEPSKit
    using ClusterExpansions
    using Plots
    const to = ClusterExpansions.to   # the timer that the package code writes to
end

# The helper functions of this script, in ../tools/:
#   timing_io.jl:    timer_rows, write_timer_csv, read_timer_csv, new_run_dir, existing_run_dir
#   timing_plots.jl: make_timing_phases_plot, make_timing_self_plot, ...
#   compare_tools.jl: check_cost, obs_seconds_allowed, ising_results_file, append_ising_row, read_ising_results, plot_spread, ...
#   pepo_files.jl:    load_pepos, load_bpeps_tensors, psi0_seed (the PEPO file of stage 1, ising_model_3D_CE_pepos.jl)
# (timing_plots.jl loads Plots: only in the ClusterExpansions environment)
include(joinpath(@__DIR__, "..", "tools", "timing_io.jl"))
IN_FAST_ENV || include(joinpath(@__DIR__, "..", "tools", "timing_plots.jl"))
include(joinpath(@__DIR__, "..", "tools", "compare_tools.jl"))
include(joinpath(@__DIR__, "..", "tools", "pepo_files.jl"))
# Talapov-Blöte, Bethe and the MC points of development/reference/ising_3d_mc.jl in the magnetization plot (plots only in the ClusterExpansions environment)
IN_FAST_ENV || include(joinpath(@__DIR__, "..", "..", "reference", "ising_3d_reference.jl"))

# ======================================================================================================================
# Settings
# ======================================================================================================================
# Profile: which machine and which run (the Hortense job scripts in development/hpc/ set CE_COMPARE_PROFILE; without it we use :laptop)
#   :laptop  Milan's laptop, julia -t 4: 1 h per case (4 core hours)
#   :pilot   Hortense, dimension pilot (:betterpepskit): 1 β near βc, the two cold cases, 1 h each on 3 cores; one job per candidate (CE_COMPARE_DIMS)
#   :timing  Hortense, fast against slow (round 1, next to the pilots): (27, 3, 3), the smallest candidate, 1 β near βc, the cold case, 30 min on 4 cores,
#            one job per implementation (:pepskit needs the 4 cores for its memory; both get them, so the times can be compared)
#   :hpc     Hortense, the full run: all β, 21.3 h per case on 3 cores (64 core hours); one job per case (CE_COMPARE_CASES), the 2 jobs in parallel
# The Hortense job scripts can also choose the dimensions (CE_COMPARE_DIMS), the cases (CE_COMPARE_CASES) and the run (CE_COMPARE_RUN) of one job
# (see compare_dims in ../tools/compare_tools.jl).
# The same CORE budget per case as the CTMRG3D compare (cores × time): on Hortense vbPEPS runs on 3 cores and CTMRG3D on 4, so vbPEPS gets 4/3 of
# the time (21.3 h against 16 h per case = 64 core hours each). On the laptop both have 4 cores and 1 h.
# Memory: 2 cores (about 3.7 GiB) ran out of memory in the pilots of round 1 (20 Nov); on 3 cores (about 5.6 GiB) the pilots peaked at 5.3-5.6 GiB
# (Slurm; 4.6 GiB by the script). The cores are also in development/hpc/submit_compare.sh, submit_round1.sh and job_vbpeps_compare.pbs.
# dims = (chienv, chipeps, D_CE). HPC_DIMS (the dims of :timing and :hpc) is the choice of the dimension pilot (the candidates are in
# development/hpc/submit_compare.sh).
# Threads: vbPEPS works with small tensors, so every core is one Julia thread with 1 BLAS thread.
const PROFILE = Symbol(get(ENV, "CE_COMPARE_PROFILE", "laptop"))
const β_ALL = [0.15, 0.21, 0.22, 0.225, 0.23, 0.25, 0.35]   # one point deep in each phase and five around βc ≈ 0.2217 (21 Nov: also 0.22 and 0.225;
                                                      # the finite-χ transition is shifted and smeared)
const β_PILOT = [0.19, 0.22, 0.25]                    # the β of the pilots: paramagnet, just below βc (the slowest point), ordered
const CASES_ALL = ["false_cold", "false_warm"]
const HPC_DIMS = (32, 3, 3)                          # dims of :hpc: set to the choice of the pilot
const PROFILES = Dict(
    :laptop => (; dims = (14, 2, 2), β_obs = β_ALL, cases = CASES_ALL, run_budget_s = 3600.0, max_seconds_obs = 1200.0, min_seconds_obs = 60.0,
                julia_threads = 4, blas_threads = 1),
    :pilot  => (; dims = (32, 3, 3), β_obs = β_PILOT, cases = ["false_cold"], run_budget_s = 3600.0, max_seconds_obs = 3600.0, min_seconds_obs = 60.0,
                julia_threads = 3, blas_threads = 1),
    :timing => (; dims = (27, 3, 3), β_obs = [0.21], cases = ["false_cold"], run_budget_s = 1800.0, max_seconds_obs = 1800.0, min_seconds_obs = 60.0,
                julia_threads = 4, blas_threads = 1),
    :hpc    => (; dims = HPC_DIMS, β_obs = β_ALL, cases = CASES_ALL, run_budget_s = 76800.0, max_seconds_obs = 25600.0, min_seconds_obs = 60.0,
                julia_threads = 3, blas_threads = 1),
)
haskey(PROFILES, PROFILE) || error("unknown profile :$PROFILE (CE_COMPARE_PROFILE): use laptop, pilot, timing or hpc")
const P = PROFILES[PROFILE]
# Which PEPO (environment variable CE_COMPARE_PEPO, as in the CTMRG3D compare): ce (default, the CE PEPOs) or exact (the file of the exact PEPOs of stage 1,
# bond dimension 2, so D_CE of the dims must be 2; the setting column of the results file then ends in _exactpepo)
const PEPO_KIND = Symbol(lowercase(get(ENV, "CE_COMPARE_PEPO", "ce")))
PEPO_KIND in (:ce, :exact) || error("CE_COMPARE_PEPO = $PEPO_KIND: use ce or exact")
# Optional changes of the profile for one job, as in the CTMRG3D compare (Hortense: qsub -v CE_COMPARE_...=...; added 21 Nov for the pilot at (16,5,3)):
#   CE_COMPARE_BETAS      the β of the observations, separated by _ (for example 0.21_0.225)
#   CE_COMPARE_BUDGET_S   seconds for the case, and also the most one observation may use
#   CE_COMPARE_MAXITER    the most LBFGS iterations per observation (optimizer_maxiter of base_kwargs; default 100)
const OVERRIDES = [n for n in ("CE_COMPARE_BETAS", "CE_COMPARE_BUDGET_S", "CE_COMPARE_MAXITER", "CE_COMPARE_PEPO") if haskey(ENV, n)]
BLAS.set_num_threads(P.blas_threads)
Threads.nthreads() == P.julia_threads || @warn "Profile :$PROFILE expects julia -t $(P.julia_threads), but Threads.nthreads() = $(Threads.nthreads())"

const T = Float64
const spin_symmetry = Trivial
const J = 1.0
const g = 0.0
# chienv above the bond of the three-layer network (chipeps * D_CE * chipeps); the laptop keeps D_CE = 2 (with D_CE = 3 vbPEPS needs chipeps >= 3 and then
# chienv is unaffordable on the laptop, tests of 16-17 Nov); on Hortense D_CE = 3, the same as CTMRG3D
const chienv, chipeps, D_CE = compare_dims(P.dims)
PEPO_KIND === :exact && D_CE != 2 && error("CE_COMPARE_PEPO = exact has bond dimension 2: choose dims with D_CE = 2 (for example CE_COMPARE_DIMS = 27_3_2), not $D_CE")
const Δβ_iter = 0.005
# The β values of the observations (increasing; every value must be a multiple of Δβ_iter): from the profile
const β_obs = env_override("CE_COMPARE_BETAS", parse_betas, P.β_obs)

const SEED_BASE = 1000              # observation i uses Random.seed!(SEED_BASE + i), the same in all cases
const BETA_C = 0.2217               # warm cases: a β closer than WARM_WINDOW to BETA_C starts cold (nothing = always warm)
const WARM_WINDOW = 0.02

# Time limits (seconds). Total of the run is about: compile of `true` (~6000 s) + compile of `false` (a few minutes) + 4 * RUN_BUDGET_S (+ at most one LBFGS iteration overshoot per observation).
# For a comparison with CTMRG3D at the same core budget, (cores × RUN_BUDGET_S) must be the same in both compare scripts (see the profiles).
const RUN_BUDGET_S = env_override("CE_COMPARE_BUDGET_S", s -> parse(Float64, s), P.run_budget_s)           # per case (laptop 1 h on 4 cores; Hortense 21.3 h on 3 cores = the core budget of CTMRG3D)
# The most one observation may use, also: MAX_OBS_FACTOR × the fair share RUN_BUDGET_S / length(β_obs) (21 Nov: β near βc that do not converge would
# otherwise take twice the fair share each and leave the last β, which converge fast, almost nothing)
const MAX_OBS_FACTOR = 1.2
const MAX_SECONDS_OBS = min(env_override("CE_COMPARE_BUDGET_S", s -> parse(Float64, s), P.max_seconds_obs), MAX_OBS_FACTOR * RUN_BUDGET_S / length(β_obs))   # per observation (lowered further when the case budget is running out)
const MIN_SECONDS_OBS = P.min_seconds_obs     # but every observation gets at least this


# Implementation of vbPEPS (see the top of this file); the Hortense job scripts can set it with the environment variable CE_COMPARE_IMPL
#   :pepskit (the reference, PEPSKit 0.8.1, environment of the clone) or :betterpepskit (fast, BetterPEPSKit, environment development/fast_vbpeps/)
const VBPEPS_IMPL = Symbol(get(ENV, "CE_COMPARE_IMPL", "betterpepskit"))
VBPEPS_IMPL in (:pepskit, :betterpepskit) || error("VBPEPS_IMPL (CE_COMPARE_IMPL) must be pepskit or betterpepskit, got $VBPEPS_IMPL")
# Settings that only :betterpepskit uses (the values of Lander's example.jl):
#   sparse: block-Krylov decompositions (true) or dense eigh/SVD (false); truncation_window, truncation_rtol: the gap cut (truncgap);
#   gradient_maxiter, gradient_krylovdim: GMRES of ImplicitGradient (these two replace those of base_kwargs, which are for Arnoldi)
#   (ising_model_3D_CE_vbPEPS_decompositions.jl times sparse = true against sparse = false for several bond dimensions)
# sparse = true (block-Krylov): the decompositions benchmark of round 1 (20 Nov) was faster with block-Krylov at every size of <psi|O|psi> (×1.1-4.5)
# (needs the one-line fix in BetterPEPSKit's projector_contraction_metrics, see the guide of 21 Nov)
const FAST_KWARGS = (; sparse = true, truncation_window = 4, truncation_rtol = 1.0e-3, gradient_maxiter = 20, gradient_krylovdim = 30)

# Settings of the vbPEPS (the same for all cases, except warm_start and differentiate_environment which are the 4 cases)
# C4v symmetric CTMRG: the CE PEPOs are symmetric under the cube rotations and reflections (rot_invariance = true in build_pepos, stage 1), and RotateReflect()
# keeps the boundary PEPS symmetric. Everything is Float64 (the element type of the PEPOs).
const base_kwargs = (;
    boundary_alg_type = C4vCTMRG, symmetrization = RotateReflect(),
    boundary_tol = 1.0e-10, boundary_maxiter = 1500, boundary_verbosity = 1,
    gradient_tol = 1.0e-6, gradient_maxiter = 5, gradient_krylovdim = 20,
    optimizer_memory = 64, optimizer_tol = 1.0e-5,
    optimizer_maxiter = env_override("CE_COMPARE_MAXITER", s -> parse(Int, s), 100),   # 100 is super high, but the time limit stops it before that
    maxiter_init = 20,            # must be MAXITER_INIT of ising_model_3D_CE_pepos.jl (the start states of the PEPO file)
    optimizer_verbosity = 2,
)

# (name, warm_start, differentiate_environment); the order matters: the `true` cases come last, because their compilation takes very long.
# The profile (or CE_COMPARE_CASES) chooses which of them we run.
const RUN_CASES = compare_cases(P.cases, CASES_ALL)
const CASES = filter(c -> c[1] in RUN_CASES, [
    ("false_cold", false, false),
    ("false_warm", true, false),
])


# If REPLOT_DIR is the NAME of a run folder (for example "20261116_1500") in data/profiling/comparing/ising_model_3D_CE_vbPEPS_compare/,
# we only redo all the plots from the rows of that run in the results file and nothing else is run (no calculation, no compilation of the pullback).
# Leave it "" for a normal run.
const REPLOT_DIR = ""


# Settings of the spread plot (|m| of every case minus the mean |m| of the cases, per β)
const SPREAD_EXCLUDE_CASES = String[]       # cases that we leave out completely, for example ["true_warm"]
const SPREAD_EXCLUDE_FLAGGED = false         # leave out observations that failed or whose check is not "ok" (unphysical f)
const SPREAD_EXCLUDE_TIMED_OUT = false      # leave out observations that were stopped by the time limit
const SPREAD_OUTLIER_TOL = 0.1             # leave out a point whose |m| differs more than this from the median |m| of the cases at the same β

# Settings of the timing plots (4 bars per section, one per case)
const TIMING_TOP = 4                     # number of sections in the self-time plot
const CASE_COLORS = Dict("false_cold" => :steelblue, "false_warm" => :lightskyblue, "true_cold" => :firebrick, "true_warm" => :lightsalmon)   # blue = fixed env, red = diff env, dark = cold, light = warm

# ======================================================================================================================
# One case
# ======================================================================================================================
function run_case(name, warm, diff, Os, pepsspace, envspace, obs, resultspath, outdir, run)
    # Os[i] = the input of observation i: the 3D PEPO (:pepskit) or the packed tensors (; O, M, psi0) of the PEPO file (:betterpepskit)
    # name = e.g. "true_warm"

    @info "===== Case $name: warm_start = $warm, differentiate_environment = $diff, budget = $(RUN_BUDGET_S) s ====="
    reset_timer!(to)
    IN_FAST_ENV ? FastBpeps.reset_timer() : ClusterExpansions.reset_max_alloc!()
    kwargs = (; base_kwargs..., differentiate_environment = diff)
    state = nothing           # holds (psi, env2, env3) for the warm start
    # the columns of the results file that are the same for every observation of this case
    row_settings = (; algorithm = "vbPEPS", run, case = name, setting = (diff ? "diff_env_" : "fixed_env_") * String(VBPEPS_IMPL) * (PEPO_KIND === :exact ? "_exactpepo" : ""), warm, D_CE, chipeps, chienv,
                    dβ = Δβ_iter, budget_s= RUN_BUDGET_S)
    t_case = time()

    # Loop over all observations (at different β, using different O's)
    for i in eachindex(Os)
        β = β_obs[i]
        # Use same random seed in the different cases
        Random.seed!(SEED_BASE + i)
        
        # time we allow this observation: twice the fair share of the case budget that is left, between MIN_SECONDS_OBS and MAX_SECONDS_OBS
        # (length(Os) - i + 1 = observations left, including this one)
        allowed = obs_seconds_allowed(RUN_BUDGET_S, t_case, length(Os) - i + 1; min_s = MIN_SECONDS_OBS, max_s = MAX_SECONDS_OBS)

        # Start of observation
        t_obs = time()
        try
            # The actual optimisation (may be with warm start) and ⟨σᶻ⟩ on the converged state, with the implementation VBPEPS_IMPL
            # warm case: warm from the previous β, but cold near βc (warm_start_at in ../tools/compare_tools.jl)
            use_warm = warm && warm_start_at(β, BETA_C, WARM_WINDOW)
            res, state = observe(name, i, Os[i], pepsspace, envspace, obs, use_warm, state, allowed, kwargs)
            # Total time of this observation (optimisation + observation)
            dt = time() - t_obs
            # See if f is realistic
            check = check_cost(β, res.cost) * (warm && !use_warm ? "; cold start near βc" : "")
            GC.gc()             # the garbage of the optimization (see memory_note in ../tools/compare_tools.jl); after dt, so not in the timings

            # Print/save result
            println("OBS $name β = $(round(β; digits = 3)): f = $(res.cost), calls = $(res.numfg), time = $(round(dt; digits = 1)) s, timed out = $(res.timed_out), m = $(res.m), ", check, "; ", memory_note())
            flush(stdout)       # the line is in the job log at once (also when the job is killed later)

            append_ising_row(resultspath; row_settings..., β, m = ComplexF64(res.m), f = res.cost, iters = res.numfg, timed_out = res.timed_out,
                             allowed_s = allowed, time_s = dt, check)
        # Runs if anything inside the try block threw an error
        catch err
            # Time untill failure
            dt = time() - t_obs
            # Print/Save the result
            @error "OBS $name β = $(round(β; digits = 3)) failed" exception = (err, catch_backtrace())
            append_ising_row(resultspath; row_settings..., β, m = complex(NaN, NaN), allowed_s = allowed, time_s = dt, 
                             check = "ERROR: " * first(sprint(showerror, err), 150))

            # If warm, don't update state (reuse latest succesfull one)
        end
        # Force the log output to be writte now
        flush(stdout); flush(stderr)
    end

    @info "Case $name finished: $(round(time() - t_case; digits = 1)) s, peak memory of the process so far $(round(Sys.maxrss() / 2^30; digits = 2)) GiB"
    try
        write_timer_csv(joinpath(outdir, "timing_$(name).csv"), timer_rows(to))
    catch err
        @error "Writing the timer csv of $name failed" exception = (err, catch_backtrace())
    end

    # Force garbage collection between cases (releases memory)
    GC.gc()
    return nothing
end

# Warm-up: compile everything with tiny settings (not measured, nothing is saved)
function warmup(diff, Os, pepsspace, envspace, obs)
    @info "Warm-up (compilation) for differentiate_environment = $diff ($VBPEPS_IMPL): not measured"
    t0 = time()
    tiny = (; base_kwargs..., differentiate_environment = diff, optimizer_maxiter = 1, maxiter_init = 2, boundary_maxiter = 100)
    Random.seed!(SEED_BASE)
    # first call with a cold start, second call with a warm start, so that the warm-start path is compiled as well
    # (Os[min(2, end)]: also with a single β, as in the pilot)
    _, state = observe("warmup", 1, Os[1], pepsspace, envspace, obs, false, nothing, Inf, tiny)
    observe("warmup", min(2, length(Os)), Os[min(2, end)], pepsspace, envspace, obs, true, state, Inf, tiny)
    @info "Warm-up for differentiate_environment = $diff took $(round(time() - t0; digits = 1)) s"
    GC.gc()
    @info "After the warm-up: " * memory_note()
    return nothing
end

# ======================================================================================================================
# One observation with the implementation VBPEPS_IMPL: returns ((; cost, numfg, timed_out, m), state for the next warm start)
# input = Os[i] (see run_case), β = β_obs[i]
# ======================================================================================================================
function observe(name, i, input, pepsspace, envspace, obs, warm, state, allowed, kwargs)
    β = β_obs[i]
    cold = !warm || state === nothing
    if VBPEPS_IMPL === :pepskit
        # cold: the seed of the start state of the PEPO file, so bpeps_optimize makes the same start state as the one :betterpepskit gets
        cold && Random.seed!(psi0_seed(β, Δβ_iter))
        res = TimerOutputs.@timeit to "expectation_value (vbPEPS and observable)" begin
            result = ClusterExpansions.bpeps_optimize(input, pepsspace, envspace; state0 = cold ? nothing : state, max_seconds = allowed, kwargs...)
            (; result, m = ClusterExpansions.bpeps_observables(input, [obs], result)[1])
        end
        r = res.result
        # (psi, env2, env3) for the warm start of the next observation (also after a timeout)
        return (; cost = r.cost, numfg = r.numfg, timed_out = r.timed_out, m = res.m), (r.psi, r.env2, r.env3)
    end

    # :betterpepskit: the 2D tensors O and M and (cold start) the start state come from the PEPO file; FastBpeps keeps the state of the warm start
    # itself (state = true once there is one)
    fast_kw = (; boundary_c4v = kwargs.boundary_alg_type === C4vCTMRG, symmetric = kwargs.symmetrization !== nothing,
               boundary_tol = kwargs.boundary_tol, boundary_maxiter = kwargs.boundary_maxiter, boundary_verbosity = kwargs.boundary_verbosity,
               gradient_tol = kwargs.gradient_tol, optimizer_memory = kwargs.optimizer_memory, optimizer_tol = kwargs.optimizer_tol,
               optimizer_maxiter = kwargs.optimizer_maxiter, optimizer_verbosity = get(kwargs, :optimizer_verbosity, 0),
               differentiate_environment = kwargs.differentiate_environment, max_seconds = Float64(allowed), FAST_KWARGS...)
    res = FastBpeps.run_observation(name, input.O, input.M, cold ? input.psi0 : nothing, dim(pepsspace), dim(envspace);
                                    warm = !cold, seed = rand(1:typemax(Int32)), fast_kw...)
    return res, true
end

# ======================================================================================================================
# Summary table and plots (from the rows of this run in the results file)
# ======================================================================================================================
function print_summary(rows)
    println("\n===== Summary per case =====")
    println(rpad("case", 12), rpad("n obs", 7), rpad("total time (s)", 16), rpad("calls", 8), rpad("timed out", 11), "failed")
    for (name, _, _) in CASES
        sub = filter(r -> r.case == name, rows)
        isempty(sub) && continue
        println(rpad(name, 12), rpad(length(sub), 7), rpad(round(sum(r.time_s for r in sub); digits = 1), 16), rpad(sum(r.iters for r in sub), 8),
                rpad(count(r -> r.timed_out, sub), 11), count(r -> !isfinite(r.f), sub))
    end
    println("\n===== m (real part) per β =====")
    βs = sort(unique(r.β for r in rows))
    println(rpad("β", 8), join([rpad(name, 14) for (name, _, _) in CASES]))
    for β in βs
        line = rpad(round(β; digits = 3), 8)
        for (name, _, _) in CASES
            r = filter(r -> r.case == name && r.β == β, rows)
            line *= rpad(isempty(r) ? "-" : string(round(r[1].m_re; sigdigits = 4)), 14)
        end
        println(line)
    end
    return nothing
end

# One line per case in the panel p: y(r) of every observation r of the case, sorted by β, and a cross where the observation timed out.
# Colour: differentiate_environment, line style: warm start (dashed) or cold start (solid). labelled = true puts the cases in the legend of this panel.
colour(diff) = diff ? :crimson : :royalblue
style(warm) = warm ? :dash : :solid
function plot_cases!(p, rows, y; labelled = false)
    for (name, warm, diff) in CASES
        sub = sort(filter(r -> r.case == name, rows); by = r -> r.β)
        isempty(sub) && continue
        βs = [r.β for r in sub]
        ys = [y(r) for r in sub]
        plot!(p, βs, ys; color = colour(diff), linestyle = style(warm), marker = :circle, markersize = 4, label = labelled ? case_label(warm, diff) : "")
        idx = findall(r -> r.timed_out, sub)
        isempty(idx) || scatter!(p, βs[idx], ys[idx]; color = colour(diff), markershape = :xcross, markersize = 9, markerstrokewidth = 2, label = "")
    end
    return p
end

# The three panels. The difference panel (|m| of every case minus the mean of the cases) is plot_spread in compare_tools.jl.
function plot_magnetization(rows)
    p = plot(; xlabel = "β", ylabel = "|⟨σᶻ⟩|", title = "magnetization (g = 0: 0 below βc)", legend = :topleft)
    plot_cases!(p, rows, r -> hypot(r.m_re, r.m_im); labelled = true)
    plot_ising_references!(p, minimum(r.β for r in rows), maximum(r.β for r in rows))
    vline!(p, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    return p
end

function plot_time(rows)
    p = plot(; xlabel = "β", ylabel = "time per observation (s)", title = "time", yscale = :log10, legend = :outertopright)
    plot_cases!(p, rows, r -> max(r.time_s, 1.0e-1); labelled = true)
    return p
end

function plot_cost(rows)
    p = plot(; xlabel = "β", ylabel = "f = -log(λ)", title = "cost (bounds: -(ln 2 + 3β) ≤ f ≤ -3β)", legend = false)
    plot_cases!(p, rows, r -> r.f)
    βline = sort(unique(r.β for r in rows))
    plot!(p, βline, -3 .* βline; color = :black, linestyle = :dot, label = "")
        plot!(p, βline, -(log(2) .+ 3 .* βline); color = :black, linestyle = :dot, label = "")
    vline!(p, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    return p
end

# The two figures (each with two panels next to each other)
# (the dimensions of the rows themselves: a replot can be made with another profile)
figure_title(rows) = "vbPEPS ($(split(replace(rows[1].setting, "_exactpepo" => ""), '_')[end])): χpeps = $(rows[1].chipeps), χenv = $(rows[1].chienv), D_CE = $(rows[1].D_CE)" *
                     "$(endswith(rows[1].setting, "_exactpepo") ? " (exact PEPO)" : ""), g = $g, C4v CTMRG\n(cross = timed out; dotted gray = βc)"

# Figure 1: magnetization (left) and the difference of |m| of every case to the mean of the cases (right)
function make_magnetization_figure(rows, outdir, spread_specs)
    p_dev = plot_spread(rows, spread_specs; exclude_cases = SPREAD_EXCLUDE_CASES, exclude_flagged = SPREAD_EXCLUDE_FLAGGED, exclude_timed_out = SPREAD_EXCLUDE_TIMED_OUT,
        outlier_tol = SPREAD_OUTLIER_TOL)
    plt = compare_figure([plot_magnetization(rows), p_dev], figure_title(rows))   # without p_dev (nothing) when no β has two cases
    savefig(plt, joinpath(outdir, "compare_vbpeps_magnetization.png"))
    return nothing
end

# Figure 2: time per observation (left) and cost f (right)
function make_cost_time_figure(rows, outdir)
    plt = compare_figure([plot_time(rows), plot_cost(rows)], figure_title(rows))
    savefig(plt, joinpath(outdir, "compare_vbpeps_cost_time.png"))
    return nothing
end

# Label, colour and line style of a case in all plots (colour: differentiate_environment, line style: warm start)
case_label(warm, diff) = "$(diff ? "diff env" : "fixed env"), $(warm ? "warm" : "cold")"

# All plots; each one in its own try, so that a failing plot does not stop the others
function make_all_plots(rows, outdir)
    spread_specs = [(; name, label = case_label(warm, diff), color = diff ? :crimson : :royalblue, linestyle = warm ? :dash : :solid) for (name, warm, diff) in CASES]
    timing_specs = [(; name, label = case_label(warm, diff), color = CASE_COLORS[name]) for (name, warm, diff) in CASES]
    plots = [
        "compare_vbpeps_magnetization.png" => () -> make_magnetization_figure(rows, outdir, spread_specs),
        "compare_vbpeps_cost_time.png" => () -> make_cost_time_figure(rows, outdir),
        "compare_vbpeps_timing_phases.png" => () -> make_timing_phases_plot(outdir, timing_specs; label = "vbPEPS, 4 cases", file_prefix = "compare_vbpeps"),
        "compare_vbpeps_timing_self.png" => () -> make_timing_self_plot(outdir, timing_specs; label = "vbPEPS, 4 cases", file_prefix = "compare_vbpeps", top = TIMING_TOP),
    ]
    for (file, makeplot) in plots
        try
            makeplot()
            @info "Plot saved: $(joinpath(outdir, file))"
        catch err
            @error "Plot $file failed, the data is in $ISING_RESULTS_PATH (set REPLOT_DIR to retry)" exception = (err, catch_backtrace())
        end
    end
    return nothing
end

# ======================================================================================================================
# Main
# ======================================================================================================================
function main()
    # Replot mode if run was already done, then it just reprints/replots the stored results
    if !isempty(REPLOT_DIR)
        IN_FAST_ENV && error("REPLOT_DIR needs Plots: run this script in the ClusterExpansions environment (julia --project=. <this file>)")
        plotdir = existing_run_dir(@__FILE__, REPLOT_DIR)
        rows = read_ising_results(; algorithm = "vbPEPS", run = REPLOT_DIR)
        isempty(rows) && error("no vbPEPS rows of run $REPLOT_DIR in $ISING_RESULTS_PATH")
        print_summary(rows)
        make_all_plots(rows, plotdir)
        return nothing
    end

    # The implementation must fit the environment (see the top of this file)
    (VBPEPS_IMPL === :betterpepskit) == IN_FAST_ENV || error("VBPEPS_IMPL = :$VBPEPS_IMPL needs " *
        (IN_FAST_ENV ? "the ClusterExpansions environment: julia --project=. <this file>" : "the BetterPEPSKit environment: julia --project=development/fast_vbpeps <this file>"))

    # Create the output folder (named by the current date/time; this name is the run in the results file) and the results file if it is not there yet
    t_run = time()
    outdir = compare_run_dir(@__FILE__)
    run = basename(outdir)
    resultspath = ising_results_file()

    # Write Julia version, threads and all settings to the file (appended: the parallel jobs of one run write into the same folder)
    open(joinpath(outdir, "settings.txt"), "a") do io
        println(io, "PROFILE = $PROFILE, Julia $(VERSION), threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads()), host = $(gethostname()), " *
                    "environment = $(Base.active_project())")
        println(io, "J = $J, g = $g, D_CE = $D_CE, chipeps = $chipeps, chienv = $chienv, Δβ_iter = $Δβ_iter, β_obs = $β_obs")
        isempty(OVERRIDES) || println(io, "changed for this job: ", join(["$n = $(ENV[n])" for n in OVERRIDES], ", "))
        println(io, "PEPO = $PEPO_KIND", PEPO_KIND === :exact ? " (the exact file of stage 1, bond dimension 2)" : " (CE, PEPO file)")
        println(io, "SEED_BASE = $SEED_BASE, RUN_BUDGET_S = $RUN_BUDGET_S, MAX_SECONDS_OBS = $MAX_SECONDS_OBS (MAX_OBS_FACTOR = $MAX_OBS_FACTOR), MIN_SECONDS_OBS = $MIN_SECONDS_OBS")
        println(io, "VBPEPS_IMPL = $VBPEPS_IMPL", VBPEPS_IMPL === :betterpepskit ? ", FAST_KWARGS = $FAST_KWARGS" : "")
        println(io, "base_kwargs = $base_kwargs")
        println(io, "cases = $CASES")
    end
    @info "Output directory: $outdir, results file: $resultspath (run = $run)"
    @info "PROFILE = $PROFILE, Threads.nthreads() = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads()), VBPEPS_IMPL = $VBPEPS_IMPL"
    @info memory_info()
    isempty(OVERRIDES) || @info "Changed for this job: β_obs = $β_obs, RUN_BUDGET_S = $RUN_BUDGET_S, MAX_SECONDS_OBS = $MAX_SECONDS_OBS, optimizer_maxiter = $(base_kwargs.optimizer_maxiter), PEPO = $PEPO_KIND"
    # Define the virtual spaces and the observable (:pepskit; :betterpepskit has M = the tensor of σᶻ in the PEPO file)
    pepsspace = ℂ^chipeps
    envspace = ℂ^chienv
    obs = IN_FAST_ENV ? nothing : SpinOperators.σᶻ(T, spin_symmetry)

    # The PEPOs of stage 1 (ising_model_3D_CE_pepos.jl): the 3D PEPOs (:pepskit) or the packed 2D tensors and start states (:betterpepskit)
    Os = IN_FAST_ENV ? load_bpeps_tensors(β_obs, Δβ_iter, D_CE, J, g, chipeps; maxiter_init = base_kwargs.maxiter_init, kind = PEPO_KIND) :
                       load_pepos(β_obs, Δβ_iter, D_CE, J, g; kind = PEPO_KIND)
    @info "$(PEPO_KIND === :exact ? "EXACT" : "CE") PEPOs read from $(pepo_file(D_CE, Δβ_iter, J, g; kind = PEPO_KIND))"

    # compile and run, `false` first and `true` last
    # Compiled remembers wether warm-up has been done (for both false and true env)
    compiled = Dict(false => false, true => false)
    for (name, warm, diff) in CASES
        if !compiled[diff]
            try
                warmup(diff, Os, pepsspace, envspace, obs)
            catch err
                @error "Warm-up for differentiate_environment = $diff failed (we continue)" exception = (err, catch_backtrace())
            end
            # Update compiled to show that the diff= has been compiled
            compiled[diff] = true
        end
        try
            run_case(name, warm, diff, Os, pepsspace, envspace, obs, resultspath, outdir, run)
        catch err
            @error "Case $name failed" exception = (err, catch_backtrace())
        end
    end

    rows = read_ising_results(resultspath; algorithm = "vbPEPS", run)
    print_summary(rows)
    if IN_FAST_ENV
        @info "No plots in this environment: set REPLOT_DIR = \"$run\" and run this script in the ClusterExpansions environment (julia --project=. <this file>)"
    else
        make_all_plots(rows, outdir)
    end
    @info "Total time of the run: $(round((time() - t_run) / 3600; digits = 2)) h"
    return nothing
end

main()