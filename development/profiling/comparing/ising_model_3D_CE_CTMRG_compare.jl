# Comparison of 4 CTMRG3D cases for the 3D Ising model (CE + CTMRG3D), g = 0:
#
#     projectors (:svd = original U * sqrt(S), :oblique = Proposal 1, one projector pair per bond)  x  start (cold / warm)
#
# cold = every β starts from a new environment (INIT: :ones = all ones; :random = random tensors, seed INIT_SEED);
# warm = every β starts from the converged environment of the previous β (env0 of MPSKit.expectation_value, see ctmrg3D_main.jl), the first β is always cold.
# Everything uses the same CE PEPOs (built once, the same for all cases), so the cases differ only in the two settings above.
# The only random numbers are those of the cold start (INIT_SEED): a run is reproducible.
# For every case and every β we store: m, number of CTMRG3D iterations, converged or not, last ϵ, time of the CTMRG3D iterations, time of the two contractions, total time.
# Each observation is appended (and the file closed) immediately to the results file of all Ising compare runs,
# data/profiling/comparing/ising_model_3D_CE_results.csv (algorithm = "CTMRG3D", run = name of the run folder; see ISING_CSV_HEADER in ../tools/compare_tools.jl),
# so a crash halfway keeps all earlier data. ising_model_3D_CE_CTMRG_vs_vbPEPS.jl compares cases of this script with cases of the vbPEPS one.
# The other files go to a new run folder data/profiling/comparing/ising_model_3D_CE_CTMRG_compare/<date>_<time>/ (see tools/timing_io.jl) :
#     settings.txt, timing_<case>.csv (TimerOutputs rows of each case),
#     compare_ctmrg_magnetization.png (left: |m| against β; right: |m| of every case minus the mean of the cases),
#     compare_ctmrg_time_iterations.png (left: time of the CTMRG3D iterations per observation against β; right: number of iterations against β),
#     compare_ctmrg_timing_phases.png and compare_ctmrg_timing_self.png (timings per section, 4 bars per section)
#
# With REPLOT_DIR (see the settings) we only redo all plots from the results of an earlier run, without running anything.
# The helper functions (build_pepos, results file, time limit, spread plot, timing plots) are in ../tools/ (compare_tools.jl, timing_io.jl, timing_plots.jl).
# This is the CTMRG3D counterpart of ising_model_3D_CE_vbPEPS_compare.jl.
#
# Compilation happens once per process (warm-up with tiny settings, not measured).
# Time limit, the same rule as in the vbPEPS compare script: every case gets RUN_BUDGET_S seconds in total and every observation at most MAX_SECONDS_OBS
# seconds for its CTMRG3D iterations (obs_seconds_allowed in ../tools/compare_tools.jl, max_seconds of MPSKit.expectation_value). The limit is checked
# after every full CTMRG3D iteration, so it can be overshot by one iteration (the summary prints the largest overshoot); the contraction of the
# observable comes on top of it, but counts for the case budget. An observation stops after ctm_maxiter iterations or when its time is up
# (then converged = false, a cross in the plots; timed_out = true if time was indeed up (timed_out = false if not converged after ctm_maxiter iterations)).
# BLAS uses as many threads as Julia (-t), because CTMRG3D gets its parallelism from BLAS; for a comparison at the same budget we run both compare scripts with the same -t (the same number of cores).
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
BLAS.set_num_threads(Threads.nthreads())   # CTMRG3D gets its parallelism from BLAS; the same number of cores as the Julia threads of the vbPEPS run (-t)

const to = ClusterExpansions.to # the timer that the package code writes to

# The helper functions of this script, in ../tools/:
#   timing_io.jl:    timer_rows, write_timer_csv, new_run_dir, existing_run_dir
#   timing_plots.jl: make_timing_phases_plot, make_timing_self_plot
#   compare_tools.jl: build_pepos, obs_seconds_allowed, ising_results_file, append_ising_row, read_ising_results, plot_spread, βc_ising
include(joinpath(@__DIR__, "..", "tools", "timing_io.jl"))
include(joinpath(@__DIR__, "..", "tools", "timing_plots.jl"))
include(joinpath(@__DIR__, "..", "tools", "compare_tools.jl"))

# ======================================================================================================================
# Settings
# ======================================================================================================================
const T = Float64
const spin_symmetry = Trivial
const J = 1.0
const g = 0.0
const D_CE = 3 # must be the D_CE of the vbPEPS run that we compare with (the same PEPO)
const chipeps = 5
const chienv = 11
const Δβ_iter = 0.005
# The β values of the observations (increasing; every value must be a multiple of Δβ_iter).
# One point in each phase and four around βc ≈ 0.2217 (the transition of the finite-χ results is shifted and smeared, so we cover 0.19-0.25).
#const β_obs = [0.10, 0.19, 0.21, 0.23, 0.25, 0.35]
const β_obs = [0.21]

# Settings of CTMRG3D (the same for all cases, except projectors and the start, which are the 4 cases)
const ctm_miniter = 3
const ctm_maxiter = 80
const ctm_tol = 1e-7
const WEST_UPDATES = :auto        # West updates per side: :auto or an integer ≥ 1
const RAM_GIB = 4                 # nothing = contract_onesite (unsliced); a number = contract_onesite_auto with this RAM budget in GiB (uses the plan file)
const INIT = :ones                # cold start: :ones (all ones) or :random (random tensors), the same for all cases
const INIT_SEED = 1               # seed of the random numbers of the cold start

# Time limits (seconds), the same rule and (for a comparison at the same budget) the same values as in ising_model_3D_CE_vbPEPS_compare.jl
const RUN_BUDGET_S = 3600.0             # per case: 1 hour (Inf = no budget)
const MAX_SECONDS_OBS = 1200.0          # per observation, twice the fair share of 3600 s over 6 β (lowered further when the case budget is running out; Inf = nolimit)
const MIN_SECONDS_OBS = 60.0            # but every observation gets at least this

# (name, projectors, warm start)
const CASES = [
    ("svd_cold", :svd, false),
#    ("svd_warm", :svd, true),
#    ("oblique_cold", :oblique, false),
#    ("oblique_warm", :oblique, true),
]

# If REPLOT_DIR is the NAME of a run folder (for example "20261116_1500") in data/profiling/comparing/ising_model_3D_CE_CTMRG_compare/,
# we only redo all the plots from the rows of that run in the results file and nothing else is run. Leave it "" for a normal run.
const REPLOT_DIR = ""

# Settings of the spread plot (|m| of every case minus the mean |m| of the cases, per β)
const SPREAD_EXCLUDE_CASES = String[]       # cases that we leave out completely, for example ["oblique_warm"]
const SPREAD_EXCLUDE_FLAGGED = false        # leave out observations that failed or did not converge (check is not "ok")
const SPREAD_EXCLUDE_TIMED_OUT = false      # leave out observations that were stopped by the time limit
const SPREAD_OUTLIER_TOL = 0.1              # leave out a point whose |m| differs more than this from the median |m| of the cases at the same β

# Settings of the timing plots (4 bars per section, one per case)
const TIMING_TOP = 6                     # number of sections in the self-time plot
const CASE_COLORS = Dict("svd_cold" => :steelblue, "svd_warm" => :lightskyblue, "oblique_cold" => :firebrick, "oblique_warm" => :lightsalmon)   # blue = svd, red = oblique, dark = cold, light = warm

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

ctm_kwargs(projectors) = (; miniter = ctm_miniter, maxiter = ctm_maxiter, tol = ctm_tol, verbosity = 0, west_updates = WEST_UPDATES, projectors, ram_gib = RAM_GIB,
                           init = INIT, init_seed = INIT_SEED)

function run_case(name, projectors, warm, Os, pepsspace, envspace, obs, resultspath, outdir, run)
    @info "===== Case $name: projectors = $projectors, warm start = $warm, budget = $(RUN_BUDGET_S) s ====="
    reset_timer!(to)
    ClusterExpansions.reset_max_alloc!()
    env_prev = nothing          # the converged environment of the previous β (warm start)
    # the columns of the results file that are the same for every observation of this case 
    row_settings = (; algorithm = "CTMRG3D", run, case = name, setting = String(projectors), warm, D_CE, chipeps, chienv, dβ = Δβ_iter, budget_s = RUN_BUDGET_S)
    t_case = time()

    for i in eachindex(Os)
        β = β_obs[i]
        # time we allow the CTMRG3D iterations of this observation (length(Os) - i + 1 = observations left, including this one)
        allowed = obs_seconds_allowed(RUN_BUDGET_S, t_case, length(Os) - i + 1; min_s = MIN_SECONDS_OBS, max_s = MAX_SECONDS_OBS)
        # the timer sections before this observation (the timer adds up over the observations of the case)
        tc0, tk0 = section_s(to, SEC_OBS, SEC_CTMRG), contract_s(to)
        t_obs = time()
        try
            expvals, env, ϵs, timed_out = ClusterExpansions.@timeit to SEC_OBS MPSKit.expectation_value(Os[i], [obs], pepsspace, envspace;
                                                                                    env0 = warm ? env_prev : nothing, return_env = true, 
                                                                                    max_seconds = allowed, ctm_kwargs(projectors)...)
            dt = time() - t_obs
            t_ctm, t_con = section_s(to, SEC_OBS, SEC_CTMRG) - tc0, contract_s(to) - tk0
            m = expvals[1]
            conv = ϵs[end] < ctm_tol
            check = conv ? "ok" : timed_out ? "timed out" : "not converged"

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
        tiny = (; ctm_kwargs(projectors)..., miniter = 0, maxiter = 1)
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
case_label(projectors, warm) = "$projectors, $(warm ? "warm" : "cold")"
colour(projectors) = projectors == :oblique ? :crimson : :royalblue
style(warm) = warm ? :dash : :solid

# One line per case in the panel p: y(r) of every finished observation r of the case, sorted by β, and a cross where the observation did not converge.
function plot_cases!(p, rows, y; labelled = false)
    for (name, projectors, warm) in CASES
        sub = sort(filter(r -> r.case == name && isfinite(r.m_re), rows); by = r -> r.β)
        isempty(sub) && continue
        βs = [r.β for r in sub]
        ys = [y(r) for r in sub]
        plot!(p, βs, ys; color = colour(projectors), linestyle = style(warm), marker = :circle, markersize = 4, label = labelled ? case_label(projectors, warm) : "")
        idx = findall(r -> r.converged === false, sub)
        isempty(idx) || scatter!(p, βs[idx], ys[idx]; color = colour(projectors), markershape = :xcross, markersize = 9, markerstrokewidth = 2, label = "")
    end
    return p
end

function plot_magnetization(rows)
    p = plot(; xlabel = "β", ylabel = "|⟨σᶻ⟩|", title = "magnetization (g = 0: 0 below βc)", legend = :topleft)
    plot_cases!(p, rows, r -> hypot(r.m_re, r.m_im); labelled = true)
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

figure_title() = "CTMRG3D: χpeps = $chipeps, χenv = $chienv, D_CE = $D_CE, g = $g, west_updates = $WEST_UPDATES, 
                  cold start $INIT, budget $(RUN_BUDGET_S) s per case (cross = not converged or timed out; dotted gray = βc)"

# Figure 1: magnetization (left) and the difference of |m| of every case to the mean of the cases (right)
function make_magnetization_figure(rows, outdir, spread_specs)
    p_dev = plot_spread(rows, spread_specs; exclude_cases = SPREAD_EXCLUDE_CASES, exclude_flagged = SPREAD_EXCLUDE_FLAGGED, 
                        exclude_timed_out = SPREAD_EXCLUDE_TIMED_OUT, outlier_tol = SPREAD_OUTLIER_TOL)
    plt = plot(plot_magnetization(rows), p_dev; layout = (1, 2), size = (1600, 560), margin = 6Plots.mm, bottom_margin = 8Plots.mm, left_margin = 8Plots.mm,
        plot_title = figure_title())
    savefig(plt, joinpath(outdir, "compare_ctmrg_magnetization.png"))
    return nothing
end

# Figure 2: time of the CTMRG3D iterations (left) and number of iterations (right)
function make_time_iterations_figure(rows, outdir)
    plt = plot(plot_time(rows), plot_iterations(rows); layout = (1, 2), size = (1600, 560), margin = 6Plots.mm, bottom_margin = 8Plots.mm, left_margin = 8Plots.mm,
        plot_title = figure_title())
    savefig(plt, joinpath(outdir, "compare_ctmrg_time_iterations.png"))
    return nothing
end

# All plots; each one in its own try, so that a failing plot does not stop the others
function make_all_plots(rows, outdir)
    spread_specs = [(; name, label = case_label(projectors, warm), color = colour(projectors), linestyle = style(warm)) for (name, projectors, warm) in CASES]
    timing_specs = [(; name, label = case_label(projectors, warm), color = CASE_COLORS[name]) for (name, projectors, warm) in CASES]
    plots = [
        "compare_ctmrg_magnetization.png" => () -> make_magnetization_figure(rows, outdir, spread_specs),
        "compare_ctmrg_time_iterations.png" => () -> make_time_iterations_figure(rows, outdir),
        "compare_ctmrg_timing_phases.png" => () -> make_timing_phases_plot(outdir, timing_specs; label = "CTMRG3D, 4 cases", file_prefix = "compare_ctmrg"),
        "compare_ctmrg_timing_self.png" => () -> make_timing_self_plot(outdir, timing_specs; label = "CTMRG3D, 4 cases", file_prefix = "compare_ctmrg", top = TIMING_TOP),
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
    outdir = new_run_dir(@__FILE__)

    run = basename(outdir)              # the name of the run in the results file
    resultspath = ising_results_file()  # created with the header if it is not there yet

    open(joinpath(outdir, "settings.txt"), "w") do io
        println(io, "Julia $(VERSION), threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())")
        println(io, "J = $J, g = $g, D_CE = $D_CE, chipeps = $chipeps, chienv = $chienv, Δβ_iter = $Δβ_iter, β_obs = $β_obs")
        println(io, "ctm_miniter = $ctm_miniter, ctm_maxiter = $ctm_maxiter, ctm_tol = $ctm_tol, WEST_UPDATES = $WEST_UPDATES, RAM_GIB = $RAM_GIB, INIT = $INIT, INIT_SEED = $INIT_SEED")
        println(io, "RUN_BUDGET_S = $RUN_BUDGET_S, MAX_SECONDS_OBS = $MAX_SECONDS_OBS, MIN_SECONDS_OBS = $MIN_SECONDS_OBS")
        println(io, "cases = $CASES")
    end
    @info "Output directory: $outdir, results file: $resultspath (run = $run)"  
    @info "Threads.nthreads() = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())"

    pepsspace = ℂ^chipeps
    envspace = ℂ^chienv
    obs = SpinOperators.σᶻ(T, spin_symmetry)

    t0 = time()
    Os = build_pepos(β_obs, Δβ_iter, D_CE, J, g)
    @info "CE PEPOs built in $(round(time() - t0; digits = 1)) s"

    try
        warmup(Os, pepsspace, envspace, obs)
    catch err
        @error "Warm-up failed (we continue)" exception = (err, catch_backtrace())
    end
    for (name, projectors, warm) in CASES
        try
            run_case(name, projectors, warm, Os, pepsspace, envspace, obs, resultspath, outdir, run)
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
