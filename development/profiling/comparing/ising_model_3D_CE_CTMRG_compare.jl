# Comparison of 8 CTMRG3D cases for the 3D Ising model (CE + CTMRG3D), g = 0:
#
#     projectors (:svd = original U * sqrt(S), :oblique = Proposal 1, one projector pair per bond)  x  start (cold / warm)  x  west_updates (:auto / 1)
#
# and 10 extra cases that only run when chosen with CE_COMPARE_CASES (see CASES_EXTRA): the same 8 with class_projectors = true (gauge fix),
# and Proposal 2, the cube-symmetric CTMRG3DOctant (cold / warm). The magnetization plot also shows the Ising references of
# development/reference/ising_3d_reference.jl (Talapov-Blöte fit of MC at L = ∞, Bethe = BP of the exact network, and MC points if
# data/reference/ising3d_mc.csv exists; run development/reference/ising_3d_mc.jl for them).
#
# cold = every β starts from a new environment (INIT: :ones = all ones; :random = random tensors; :nishino_okunishi; seed INIT_SEED);
# warm = every β starts from the converged environment of the previous β (env0 of MPSKit.expectation_value, see ctmrg3D_main.jl), the first β is always cold,
#        and a β closer than WARM_WINDOW to BETA_C also starts cold (warm_start_at in ../tools/compare_tools.jl: near βc a warm start stays in the
#        symmetric state of the previous β); the check column of the results file says "cold start near βc" there.
# west_updates: the number of West updates per side and iteration (:auto = until ϵ < west_tol, at most 4; 1 = one).
# Everything uses the same CE PEPOs (read from the PEPO file of ising_model_3D_CE_pepos.jl, which must be run first; the vbPEPS compare reads the same
# file), so the cases differ only in the two settings above.
# The only random numbers are those of the cold start (INIT_SEED): a run is reproducible.
# For every case and every β we store: m, number of CTMRG3D iterations, converged or not, last ϵ, time of the CTMRG3D iterations, time of the two contractions, total time.
# Each observation is appended (and the file closed) immediately to the results file of all Ising compare runs,
# data/profiling/comparing/ising_model_3D_CE_results.csv (algorithm = "CTMRG3D", run = name of the run folder; see ISING_CSV_HEADER in ../tools/compare_tools.jl),
# so a crash halfway keeps all earlier data. ising_model_3D_CE_CTMRG_vs_vbPEPS.jl compares cases of this script with cases of the vbPEPS one.
# The other files go to a new run folder data/profiling/comparing/ising_model_3D_CE_CTMRG_compare/<date>_<time>/ (or <CE_COMPARE_RUN>/; see tools/timing_io.jl) :
#     settings.txt, timing_<case>.csv (TimerOutputs rows of each case),
#     compare_ctmrg_magnetization.png (left: |m| against β; right: |m| of every case minus the mean of the cases),
#     compare_ctmrg_time_iterations.png (left: time of the CTMRG3D iterations per observation against β; right: number of iterations against β),
#     compare_ctmrg_timing_phases.png and compare_ctmrg_timing_self.png (timings per section, 4 bars per section)
#
# With REPLOT_DIR (see the settings) we only redo all plots from the results of an earlier run, without running anything.
# The helper functions (results file, time limit, spread plot, timing plots, PEPO file) are in ../tools/ (compare_tools.jl, timing_io.jl, timing_plots.jl,
# pepo_files.jl).
# This is the CTMRG3D counterpart of ising_model_3D_CE_vbPEPS_compare.jl.
#
# Compilation happens once per process (warm-up with tiny settings, not measured).
# Time limit, the same rule as in the vbPEPS compare script: every case gets RUN_BUDGET_S seconds in total and every observation at most MAX_SECONDS_OBS
# seconds for its CTMRG3D iterations (obs_seconds_allowed in ../tools/compare_tools.jl, max_seconds of MPSKit.expectation_value). The limit is checked
# after every full CTMRG3D iteration, so it can be overshot by one iteration (the summary prints the largest overshoot); the contraction of the
# observable comes on top of it, but counts for the case budget. An observation stops after ctm_maxiter iterations or when its time is up
# (then converged = false, a cross in the plots; timed_out = true if time was indeed up (timed_out = false if not converged after ctm_maxiter iterations)).
# Machine and run: PROFILE (see the settings) chooses the bond dimensions, β_obs, the cases, the time limits, the RAM budget, the machine of the plan times
# and the BLAS threads: :laptop (default), :pilot or :hpc (Hortense, 4 cores and about 7 GiB). The Hortense job scripts in development/hpc/ set it with the
# environment variable CE_COMPARE_PROFILE, and can choose the dimensions (CE_COMPARE_DIMS), the cases (CE_COMPARE_CASES) and the run (CE_COMPARE_RUN) of one job
# (see compare_dims in ../tools/compare_tools.jl): the cases of a full run are parallel jobs that write into the same run.
# CTMRG3D gets its parallelism from BLAS, so it uses as many BLAS threads as cores.
# Note on m: at g = 0 the exact <σᶻ> is 0 below βc (Z2 symmetry) and has an arbitrary sign above βc, so we plot |m|.

ENV["GKSwstype"] = "nul"     # no plot window: the plots are only saved to a file

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using TimerOutputs
using Plots
using LinearAlgebra
using JLD2

const to = ClusterExpansions.to # the timer that the package code writes to

# The helper functions of this script, in ../tools/:
#   timing_io.jl:    timer_rows, write_timer_csv, new_run_dir, existing_run_dir
#   timing_plots.jl: make_timing_phases_plot, make_timing_self_plot
#   compare_tools.jl: obs_seconds_allowed, ising_results_file, append_ising_row, read_ising_results, plot_spread, βc_ising
#   pepo_files.jl:    load_pepos (the PEPOs of stage 1, ising_model_3D_CE_pepos.jl)
include(joinpath(@__DIR__, "..", "tools", "timing_io.jl"))
include(joinpath(@__DIR__, "..", "tools", "timing_plots.jl"))
include(joinpath(@__DIR__, "..", "tools", "compare_tools.jl"))
include(joinpath(@__DIR__, "..", "tools", "pepo_files.jl"))
include(joinpath(@__DIR__, "..", "..", "reference", "ising_3d_reference.jl"))   # Talapov-Blöte, Bethe, MC points in the magnetization plot

# ======================================================================================================================
# Settings
# ======================================================================================================================
# Profile: which machine and which run (the Hortense job scripts set CE_COMPARE_PROFILE; without it we use :laptop)
#   :laptop  Milan's laptop, julia -t 4: 1 h per case (4 core hours), plans for 3 GiB
#   :pilot   Hortense, dimension pilot: 1 β near βc, the two cold cases, 80 min each on 4 cores; one job per candidate (CE_COMPARE_DIMS)
#   :hpc     Hortense, the full run: all β, 16 h per case on 4 cores (64 core hours); one job per case (CE_COMPARE_CASES), the 8 jobs in parallel
# The pilot does not use the plan file: it slices the current tree itself (plans = nothing in onesite_kwargs), because the Hortense plans are timed
# in the same round.
# The same CORE budget per case as the vbPEPS compare (cores × time): on Hortense CTMRG3D runs on 4 cores and vbPEPS on 2, so CTMRG3D gets half
# of the time (16 h against 32 h per case = 64 core hours each). On the laptop both have 4 cores and 1 h.
# 4 cores come with about 7.4 GiB: the CTMRG3D iterations themselves need little memory (sandbox, 17 Nov: the process peaks at 3.4-3.8 GiB, compilation
# included, at (16-18, 5-6, 3)), so the RAM budget of the sliced 27-tensor contraction is 3 GiB (the pilot shows the real peak).
# dims = (chienv, chipeps, D_CE). The dims of :hpc are the choice of the dimension pilot (the candidates are in development/hpc/submit_compare.sh).
# machine: contract_onesite_auto only uses the plan times measured on this machine (column machine of the plan file; Hortense: development/hpc/job_benchmark_plans.pbs).
const PROFILE = Symbol(get(ENV, "CE_COMPARE_PROFILE", "laptop"))
const β_ALL = [0.10, 0.19, 0.21, 0.23, 0.25, 0.35]   # one point in each phase and four around βc ≈ 0.2217 (the finite-χ transition is shifted and smeared)
const CASES_ALL = ["svd_cold_auto", "svd_warm_auto", "oblique_cold_auto", "oblique_warm_auto", "svd_cold_w1", "svd_warm_w1", "oblique_cold_w1", "oblique_warm_w1"]
# Extra cases, not in the default profiles; choose them with CE_COMPARE_CASES (for example svdclass_cold_auto+octant_cold):
#   svdclass_* / obliqueclass_*: the 8 cases above with class_projectors = true (the gauge fix of ctmrg3D_helpers.jl)
#   octant_*: Proposal 2 (CTMRG3DOctant, ctmrg3D_octant.jl; west_updates has no meaning there)
const CASES_EXTRA = ["svdclass_cold_auto", "svdclass_warm_auto", "obliqueclass_cold_auto", "obliqueclass_warm_auto",
                     "svdclass_cold_w1", "svdclass_warm_w1", "obliqueclass_cold_w1", "obliqueclass_warm_w1", "octant_cold", "octant_warm"]
# Hortense: the number of cores of the job (job_ctmrg_compare.pbs passes CORES as CE_COMPARE_CORES; 4 GiB-cores were too few for (16,6,3), 21 Nov).
# Julia and BLAS threads = cores; the full run keeps 64 core hours per case (16 h on 4 cores, 12.8 h on 5, 10.7 h on 6), at most a third per observation.
const HPC_CORES = parse(Int, get(ENV, "CE_COMPARE_CORES", "5"))
const HPC_BUDGET_S = 64 * 3600.0 / HPC_CORES
const PROFILES = Dict(
    :laptop => (; dims = (11, 5, 3), β_obs = β_ALL, cases = CASES_ALL, run_budget_s = 3600.0, max_seconds_obs = 1200.0, min_seconds_obs = 60.0,
                ram_gib = 3, machine = "laptop", julia_threads = 4, blas_threads = 4, plan_file = true),
    :pilot  => (; dims = (16, 6, 3), β_obs = [0.21], cases = ["svd_cold_auto", "oblique_cold_auto"], run_budget_s = 4800.0, max_seconds_obs = 4800.0,
                min_seconds_obs = 60.0, ram_gib = 3, machine = "hortense", julia_threads = HPC_CORES, blas_threads = HPC_CORES, plan_file = false),
    :hpc    => (; dims = (16, 6, 3), β_obs = β_ALL, cases = CASES_ALL, run_budget_s = HPC_BUDGET_S, max_seconds_obs = HPC_BUDGET_S / 3, min_seconds_obs = 60.0,
                ram_gib = 3, machine = "hortense", julia_threads = HPC_CORES, blas_threads = HPC_CORES, plan_file = true),   # dims: set to the choice of the pilot
)
haskey(PROFILES, PROFILE) || error("unknown profile :$PROFILE (CE_COMPARE_PROFILE): use laptop, pilot or hpc")
const P = PROFILES[PROFILE]
BLAS.set_num_threads(P.blas_threads)
Threads.nthreads() == P.julia_threads || @warn "Profile :$PROFILE expects julia -t $(P.julia_threads), but Threads.nthreads() = $(Threads.nthreads())"

const T = Float64
const spin_symmetry = Trivial
const J = 1.0
const g = 0.0
const chienv, chipeps, D_CE = compare_dims(P.dims)   # D_CE must be the D_CE of the vbPEPS run that we compare with (the same PEPO)
const Δβ_iter = 0.005
# The β values of the observations (increasing; every value must be a multiple of Δβ_iter): from the profile
const β_obs = P.β_obs

# Settings of CTMRG3D (the same for all cases, except projectors and the start, which are the 4 cases)
const ctm_miniter = 3
const ctm_maxiter = 80
const ctm_tol = 1e-7
const RAM_GIB = P.ram_gib         # nothing = contract_onesite (unsliced); a number = contract_onesite_auto with this RAM budget in GiB (uses the plan file)
const MACHINE = P.machine         # contract_onesite_auto only uses the plan times measured on this machine
const INIT = :random              # cold start: :ones (all ones), :random (random tensors) or :nishino_okunishi, the same for all cases
const BETA_C = 0.2217             # warm cases: a β closer than WARM_WINDOW to BETA_C starts cold (nothing = always warm)
const WARM_WINDOW = 0.02
const INIT_SEED = 1               # seed of the random numbers of the cold start

# Time limits (seconds), the same rule as in ising_model_3D_CE_vbPEPS_compare.jl; for a comparison at the same core budget (cores × RUN_BUDGET_S) is the same in both
const RUN_BUDGET_S = P.run_budget_s           # per case (laptop 1 h on 4 cores; Hortense 64 core hours / HPC_CORES = the core budget of vbPEPS; Inf = no budget)
const MAX_SECONDS_OBS = P.max_seconds_obs     # per observation, twice the fair share over 6 β (lowered further when the case budget is running out; Inf = no limit)
const MIN_SECONDS_OBS = P.min_seconds_obs     # but every observation gets at least this

# (name, projectors, warm start, west_updates); the profile (or CE_COMPARE_CASES) chooses which of them we run
const RUN_CASES = compare_cases(P.cases, [CASES_ALL; CASES_EXTRA])
const CASES = filter(c -> c[1] in RUN_CASES, [
    ("svd_cold_auto", :svd, false, :auto),
    ("svd_warm_auto", :svd, true, :auto),
    ("oblique_cold_auto", :oblique, false, :auto),
    ("oblique_warm_auto", :oblique, true, :auto),
    ("svd_cold_w1", :svd, false, 1),
    ("svd_warm_w1", :svd, true, 1),
    ("oblique_cold_w1", :oblique, false, 1),
    ("oblique_warm_w1", :oblique, true, 1),
    # :svdclass / :obliqueclass = :svd / :oblique with class_projectors = true; :octant = ctm_alg CTMRG3DOctant (see ctm_kwargs)
    ("svdclass_cold_auto", :svdclass, false, :auto),
    ("svdclass_warm_auto", :svdclass, true, :auto),
    ("obliqueclass_cold_auto", :obliqueclass, false, :auto),
    ("obliqueclass_warm_auto", :obliqueclass, true, :auto),
    ("svdclass_cold_w1", :svdclass, false, 1),
    ("svdclass_warm_w1", :svdclass, true, 1),
    ("obliqueclass_cold_w1", :obliqueclass, false, 1),
    ("obliqueclass_warm_w1", :obliqueclass, true, 1),
    ("octant_cold", :octant, false, :auto),
    ("octant_warm", :octant, true, :auto),
])

# If REPLOT_DIR is the NAME of a run folder (for example "20261116_1500") in data/profiling/comparing/ising_model_3D_CE_CTMRG_compare/,
# we only redo all the plots from the rows of that run in the results file and nothing else is run. Leave it "" for a normal run.
const REPLOT_DIR = ""

# Settings of the spread plot (|m| of every case minus the mean |m| of the cases, per β)
const SPREAD_EXCLUDE_CASES = String[]       # cases that we leave out completely, for example ["oblique_warm_w1"]
const SPREAD_EXCLUDE_FLAGGED = false        # leave out observations that failed or did not converge (check is not "ok")
const SPREAD_EXCLUDE_TIMED_OUT = false      # leave out observations that were stopped by the time limit
const SPREAD_OUTLIER_TOL = 0.1              # leave out a point whose |m| differs more than this from the median |m| of the cases at the same β

# Settings of the timing plots (one bar per case and section)
const TIMING_TOP = 6                     # number of sections in the self-time plot
# blue = svd, red = oblique; dark = cold, light = warm; west_updates 1 = the deeper shade
const CASE_COLORS = Dict("svd_cold_auto" => :steelblue, "svd_warm_auto" => :lightskyblue, "oblique_cold_auto" => :firebrick, "oblique_warm_auto" => :lightsalmon,
                         "svd_cold_w1" => :navy, "svd_warm_w1" => :deepskyblue, "oblique_cold_w1" => :darkred, "oblique_warm_w1" => :orange,
                         "svdclass_cold_auto" => :purple, "svdclass_warm_auto" => :plum, "obliqueclass_cold_auto" => :darkorange, "obliqueclass_warm_auto" => :gold,
                         "svdclass_cold_w1" => :indigo, "svdclass_warm_w1" => :orchid, "obliqueclass_cold_w1" => :chocolate, "obliqueclass_warm_w1" => :khaki,
                         "octant_cold" => :darkgreen, "octant_warm" => :lightgreen)

# The timer sections of one observation (the inner names are those of MPSKit.expectation_value in ctmrg3D_main.jl)
const SEC_OBS = "expectation_value (CTMRG3D and observable)"
const SEC_CTMRG = "EV: leading_boundary (CTMRG3D iterations)"
const SEC_CONTRACT = ("EV: contract_onesite numerator (M)", "EV: contract_onesite denominator (O)")

# ======================================================================================================================
# One case
# ======================================================================================================================
# Total time (s) of the timer section at this path in t (0 if it is not there yet)
function section_s(t::TimerOutput, path...)
    for p in path
        haskey(t.inner_timers, p) || return 0.0
        t = t.inner_timers[p]
    end
    return TimerOutputs.time(t) / 1.0e9
end
contract_s(t) = sum(section_s(t, SEC_OBS, s) for s in SEC_CONTRACT)

# (the pilot slices the current tree itself: plans = nothing)
# projectors of a case: :svd, :oblique, :svdclass, :obliqueclass (class_projectors = true) or :octant (ctm_alg = CTMRG3DOctant)
ctm_kwargs(projectors, west_updates) = (; miniter = ctm_miniter, maxiter = ctm_maxiter, tol = ctm_tol, verbosity = 0, west_updates,
                           projectors = projectors in (:oblique, :obliqueclass) ? :oblique : :svd, class_projectors = projectors in (:svdclass, :obliqueclass),
                           ctm_alg = projectors === :octant ? CTMRG3DOctant : CTMRG3D,
                           ram_gib = RAM_GIB, init = INIT, init_seed = INIT_SEED, onesite_kwargs = P.plan_file ? (; machine = MACHINE) : (; machine = MACHINE, plans = nothing))

function run_case(name, projectors, warm, west_updates, Os, pepsspace, envspace, obs, resultspath, outdir, run)
    @info "===== Case $name: projectors = $projectors, warm start = $warm, west_updates = $west_updates, budget = $(RUN_BUDGET_S) s ====="
    reset_timer!(to)
    ClusterExpansions.reset_max_alloc!()
    env_prev = nothing          # the converged environment of the previous β (warm start)
    # the columns of the results file that are the same for every observation of this case 
    row_settings = (; algorithm = "CTMRG3D", run, case = name, setting = "$(projectors)_west$(west_updates)", warm, D_CE, chipeps, chienv, dβ = Δβ_iter, budget_s = RUN_BUDGET_S)
    t_case = time()

    for i in eachindex(Os)
        β = β_obs[i]
        # time we allow the CTMRG3D iterations of this observation (length(Os) - i + 1 = observations left, including this one)
        allowed = obs_seconds_allowed(RUN_BUDGET_S, t_case, length(Os) - i + 1; min_s = MIN_SECONDS_OBS, max_s = MAX_SECONDS_OBS)
        # the timer sections before this observation (the timer adds up over the observations of the case)
        tc0, tk0 = section_s(to, SEC_OBS, SEC_CTMRG), contract_s(to)
        # warm case: warm from the previous β, but cold near βc (warm_start_at in ../tools/compare_tools.jl)
        use_warm = warm && warm_start_at(β, BETA_C, WARM_WINDOW)
        t_obs = time()
        try
            expvals, env, ϵs, timed_out = ClusterExpansions.@timeit to SEC_OBS MPSKit.expectation_value(Os[i], [obs], pepsspace, envspace;
                                                                                    env0 = use_warm ? env_prev : nothing, return_env = true, 
                                                                                    max_seconds = allowed, ctm_kwargs(projectors, west_updates)...)
            dt = time() - t_obs
            t_ctm, t_con = section_s(to, SEC_OBS, SEC_CTMRG) - tc0, contract_s(to) - tk0
            m = expvals[1]
            conv = ϵs[end] < ctm_tol
            check = (conv ? "ok" : timed_out ? "timed out" : "not converged") * (warm && !use_warm ? "; cold start near βc" : "")

            println("OBS $name β = $(round(β; digits = 3)): m = $m, $(length(ϵs)) iterations, ϵ = $(ϵs[end]), CTMRG $(round(t_ctm; digits = 1)) s ",
                    "(allowed $(round(allowed; digits = 1)) s), contraction $(round(t_con; digits = 1)) s, total $(round(dt; digits = 1)) s, $check")
            
            append_ising_row(resultspath; row_settings..., β, m = ComplexF64(m), iters = length(ϵs), converged = conv, eps = ϵs[end], timed_out,
                             allowed_s = allowed, time_s = dt, t_ctmrg = t_ctm, t_contract = t_con, check)
            # warm: the next β starts from this environment (also if it did not converge)
            warm && (env_prev = env)
        catch err
            dt = time() - t_obs
            @error "OBS $name β = $(round(β; digits = 3)) failed" exception = (err, catch_backtrace())
            append_ising_row(resultspath; row_settings..., β, m = complex(NaN, NaN), converged = false, allowed_s = allowed, time_s = dt,
                             check = "ERROR: " * first(sprint(showerror, err), 150))            
            # warm: we keep the last environment that worked
        end
        flush(stdout); flush(stderr)
    end

    @info "Case $name finished: $(round(time() - t_case; digits = 1)) s, peak memory of the process so far $(round(Sys.maxrss() / 2^30; digits = 2)) GiB"
    try
        write_timer_csv(joinpath(outdir, "timing_$(name).csv"), timer_rows(to))
    catch err
        @error "Writing the timer csv of $name failed" exception = (err, catch_backtrace())
    end
    GC.gc()
    return nothing
end

# Warm-up: compile everything with tiny settings (not measured, nothing is saved): both projector kinds, cold and warm start
function warmup(Os, pepsspace, envspace, obs)
    @info "Warm-up (compilation): not measured"
    t0 = time()
    for projectors in unique(c[2] for c in CASES)
        tiny = (; ctm_kwargs(projectors, :auto)..., miniter = 0, maxiter = 1)
        _, env, _, _ = MPSKit.expectation_value(Os[1], [obs], pepsspace, envspace; return_env = true, tiny...)
        MPSKit.expectation_value(Os[min(2, end)], [obs], pepsspace, envspace; env0 = env, tiny...)
    end
    @info "Warm-up took $(round(time() - t0; digits = 1)) s"
    return nothing
end

# ======================================================================================================================
# Summary table and plots (from the rows of this run in the results file)
# ======================================================================================================================
function print_summary(rows)
    println("\n===== Summary per case =====")
    println(rpad("case", 14), rpad("n obs", 7), rpad("CTMRG time (s)", 16), rpad("total time (s)", 16), rpad("iterations", 12), rpad("not converged", 15),
            rpad("timed out", 11), rpad("max overshoot (s)", 19), "failed")
    # overshoot = time of the CTMRG3D iterations beyond the time that was allowed (only for observations that were stopped by the limit)
    
    for (name, _, _) in CASES
        sub = filter(r -> r.case == name, rows)
        isempty(sub) && continue
        ok = filter(r -> isfinite(r.m_re), sub)
        println(rpad(name, 14), rpad(length(sub), 7), rpad(round(sum((r.t_ctmrg for r in ok); init = 0.0); digits = 1), 16), 
                rpad(round(sum(r.time_s for r in sub); digits = 1), 16), rpad(sum(r.iters for r in sub), 12), 
                rpad(count(r -> isfinite(r.m_re) && r.converged === false, sub), 15), rpad(count(r -> r.timed_out, sub), 11),
                rpad(round(maximum((r.t_ctmrg - r.allowed_s for r in sub if r.timed_out); init = 0.0); digits = 1), 19), count(r -> !isfinite(r.m_re), sub))
    end
    println("\n===== m (real part) per β =====")
    βs = sort(unique(r.β for r in rows))
    println(rpad("β", 8), join([rpad(name, 16) for (name, _, _) in CASES]))
    for β in βs
        line = rpad(round(β; digits = 3), 8)
        for (name, _, _) in CASES
            r = filter(r -> r.case == name && r.β == β, rows)
            line *= rpad(isempty(r) ? "-" : string(round(r[1].m_re; sigdigits = 6)), 16)
        end
        println(line)
    end
    return nothing
end

# Label, colour and line style of a case in all plots (colour: projectors, line style: warm start)
case_label(projectors, warm, west = :auto) = projectors === :octant ? "octant (proposal 2), $(warm ? "warm" : "cold")" : "$projectors, $(warm ? "warm" : "cold"), west $west"
colour(projectors) = Dict(:svd => :royalblue, :oblique => :crimson, :svdclass => :purple, :obliqueclass => :darkorange, :octant => :forestgreen)[projectors]
style(warm, west = :auto) = west === :auto ? (warm ? :dash : :solid) : (warm ? :dashdot : :dot)

# One line per case in the panel p: y(r) of every finished observation r of the case, sorted by β, and a cross where the observation did not converge.
function plot_cases!(p, rows, y; labelled = false)
    for (name, projectors, warm, west) in CASES
        sub = sort(filter(r -> r.case == name && isfinite(r.m_re), rows); by = r -> r.β)
        isempty(sub) && continue
        βs = [r.β for r in sub]
        ys = [y(r) for r in sub]
        plot!(p, βs, ys; color = colour(projectors), linestyle = style(warm, west), marker = :circle, markersize = 4, label = labelled ? case_label(projectors, warm, west) : "")
        idx = findall(r -> r.converged === false, sub)
        isempty(idx) || scatter!(p, βs[idx], ys[idx]; color = colour(projectors), markershape = :xcross, markersize = 9, markerstrokewidth = 2, label = "")
    end
    return p
end

function plot_magnetization(rows)
    p = plot(; xlabel = "β", ylabel = "|⟨σᶻ⟩|", title = "magnetization (g = 0: 0 below βc)", legend = :topleft)
    plot_cases!(p, rows, r -> hypot(r.m_re, r.m_im); labelled = true)
    plot_ising_references!(p, minimum(r.β for r in rows), maximum(r.β for r in rows))
    vline!(p, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    return p
end

function plot_time(rows)
    p = plot(; xlabel = "β", ylabel = "time of the CTMRG3D iterations (s)", title = "time of leading_boundary per observation", yscale = :log10, legend = :outertopright)
    plot_cases!(p, rows, r -> max(r.t_ctmrg, 1.0e-2); labelled = true)
    vline!(p, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    return p
end

function plot_iterations(rows)
    p = plot(; xlabel = "β", ylabel = "CTMRG3D iterations", title = "iterations until ϵ < $ctm_tol (at most $ctm_maxiter)", legend = false)
    plot_cases!(p, rows, r -> r.iters)
    vline!(p, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    return p
end

# (the dimensions and the budget of the rows themselves: a replot can be made with another profile)
figure_title(rows) = "CTMRG3D: χpeps = $(rows[1].chipeps), χenv = $(rows[1].chienv), D_CE = $(rows[1].D_CE), g = $g, 
                  cold start $INIT, budget $(rows[1].budget_s) s per case (cross = not converged or timed out; dotted gray = βc)"

# Figure 1: magnetization (left) and the difference of |m| of every case to the mean of the cases (right)
function make_magnetization_figure(rows, outdir, spread_specs)
    p_dev = plot_spread(rows, spread_specs; exclude_cases = SPREAD_EXCLUDE_CASES, exclude_flagged = SPREAD_EXCLUDE_FLAGGED, 
                        exclude_timed_out = SPREAD_EXCLUDE_TIMED_OUT, outlier_tol = SPREAD_OUTLIER_TOL)
    plt = plot(plot_magnetization(rows), p_dev; layout = (1, 2), size = (1600, 560), margin = 6Plots.mm, bottom_margin = 8Plots.mm, left_margin = 8Plots.mm,
        plot_title = figure_title(rows))
    savefig(plt, joinpath(outdir, "compare_ctmrg_magnetization.png"))
    return nothing
end

# Figure 2: time of the CTMRG3D iterations (left) and number of iterations (right)
function make_time_iterations_figure(rows, outdir)
    plt = plot(plot_time(rows), plot_iterations(rows); layout = (1, 2), size = (1600, 560), margin = 6Plots.mm, bottom_margin = 8Plots.mm, left_margin = 8Plots.mm,
        plot_title = figure_title(rows))
    savefig(plt, joinpath(outdir, "compare_ctmrg_time_iterations.png"))
    return nothing
end

# All plots; each one in its own try, so that a failing plot does not stop the others
function make_all_plots(rows, outdir)
    spread_specs = [(; name, label = case_label(projectors, warm, west), color = colour(projectors), linestyle = style(warm, west)) for (name, projectors, warm, west) in CASES]
    timing_specs = [(; name, label = case_label(projectors, warm, west), color = CASE_COLORS[name]) for (name, projectors, warm, west) in CASES]
    plots = [
        "compare_ctmrg_magnetization.png" => () -> make_magnetization_figure(rows, outdir, spread_specs),
        "compare_ctmrg_time_iterations.png" => () -> make_time_iterations_figure(rows, outdir),
        "compare_ctmrg_timing_phases.png" => () -> make_timing_phases_plot(outdir, timing_specs; label = "CTMRG3D, $(length(CASES)) cases", file_prefix = "compare_ctmrg"),
        "compare_ctmrg_timing_self.png" => () -> make_timing_self_plot(outdir, timing_specs; label = "CTMRG3D, $(length(CASES)) cases", file_prefix = "compare_ctmrg", top = TIMING_TOP),
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
    # Replot mode: only reprint/replot the stored results of an earlier run
    if !isempty(REPLOT_DIR)
        plotdir = existing_run_dir(@__FILE__, REPLOT_DIR)
        rows = read_ising_results(; algorithm = "CTMRG3D", run = REPLOT_DIR)
        isempty(rows) && error("no CTMRG3D rows of run $REPLOT_DIR in $ISING_RESULTS_PATH")
        print_summary(rows)
        make_all_plots(rows, plotdir)
        return nothing
    end

    t_run = time()
    outdir = compare_run_dir(@__FILE__)

    run = basename(outdir)              # the name of the run in the results file
    resultspath = ising_results_file()  # created with the header if it is not there yet

    # appended: the parallel jobs of one run write into the same folder
    open(joinpath(outdir, "settings.txt"), "a") do io
        println(io, "PROFILE = $PROFILE, Julia $(VERSION), threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads()), host = $(gethostname())")
        println(io, "J = $J, g = $g, D_CE = $D_CE, chipeps = $chipeps, chienv = $chienv, Δβ_iter = $Δβ_iter, β_obs = $β_obs")
        println(io, "ctm_miniter = $ctm_miniter, ctm_maxiter = $ctm_maxiter, ctm_tol = $ctm_tol, BETA_C = $BETA_C, WARM_WINDOW = $WARM_WINDOW, RAM_GIB = $RAM_GIB, MACHINE = $MACHINE, INIT = $INIT, INIT_SEED = $INIT_SEED")
        println(io, "RUN_BUDGET_S = $RUN_BUDGET_S, MAX_SECONDS_OBS = $MAX_SECONDS_OBS, MIN_SECONDS_OBS = $MIN_SECONDS_OBS")
        println(io, "cases = $CASES")
    end
    @info "Output directory: $outdir, results file: $resultspath (run = $run)"  
    @info "PROFILE = $PROFILE, Threads.nthreads() = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())"

    pepsspace = ℂ^chipeps
    envspace = ℂ^chienv
    obs = SpinOperators.σᶻ(T, spin_symmetry)

    # the PEPOs of stage 1 (ising_model_3D_CE_pepos.jl)
    Os = load_pepos(β_obs, Δβ_iter, D_CE, J, g)
    @info "CE PEPOs read from $(pepo_file(D_CE, Δβ_iter, J, g))"

    try
        warmup(Os, pepsspace, envspace, obs)
    catch err
        @error "Warm-up failed (we continue)" exception = (err, catch_backtrace())
    end
    for (name, projectors, warm, west) in CASES
        try
            run_case(name, projectors, warm, west, Os, pepsspace, envspace, obs, resultspath, outdir, run)
        catch err
            @error "Case $name failed" exception = (err, catch_backtrace())
        end
    end

    rows = read_ising_results(resultspath; algorithm = "CTMRG3D", run)
    print_summary(rows)
    make_all_plots(rows, outdir)
    @info "Total time of the run: $(round((time() - t_run) / 3600; digits = 2)) h"
    return nothing
end

main()
