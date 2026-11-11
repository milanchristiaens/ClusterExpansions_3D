# Comparison of 4 vbPEPS cases for the 3D Ising model (CE + variational boundary PEPS), g = 0:
#
#     warm start (false / true)  x  differentiate_environment (false / true)
#
# Everything uses the same CE PEPOs (built once, same for all cases) and the same random seed per observation (Random.seed!(SEED_BASE + i) before observation i),
# so the cases differ only in the two settings above. For every case and every β we store: final cost f, number of cost-function calls, time, magnetization m.
# Each observation is appended to results.csv (and closed) immediately, so a crash halfway keeps all earlier data.
# Everything goes to a new run folder  data/profiling/comparing/ising_model_3D_CE_vbPEPS_compare/<date>_<time>/  (see tools/timing_io.jl)  :
#     settings.txt, results.csv, timing_<case>.csv (TimerOutputs rows of each case),
#     compare_vbpeps.png (|m|, f and time against β), compare_vbpeps_spread.png (|m| of the cases around their mean), compare_vbpeps_timing_phases.png and compare_vbpeps_timing_self.png (timings per section, 4 bars per section)
#
# With REPLOT_DIR (see the settings) we only redo all plots from the results of an earlier run, without running anything.
# The helper functions (check_cost, build_pepos, results.csv, spread plot, timing plots) are in ../tools/ (compare_tools.jl, timing_io.jl, timing_plots.jl).
#
# Order: false/cold, false/warm, [compile of the `true` pullback, this was ~1h40 in an earlier run], true/cold, true/warm.
# Compilation happens once per process (warm-up with tiny settings, not measured).
#
# Time limit: every case gets RUN_BUDGET_S seconds in total and every observation at most MAX_SECONDS_OBS seconds (see the settings).
# An observation that runs out of time stops after its current LBFGS iteration and is marked timed_out = true (a cross in the plot):
# its f and m are then the values of the last iterate, not of a converged optimization.
#
# Note on m: at g = 0 the exact <σᶻ> is 0 below βc (Z2 symmetry) and has an arbitrary sign above βc, so we plot |m|. A nonzero m below βc is a contraction/optimization artefact.

ENV["GKSwstype"] = "nul"     # no plot window: the plots are only saved to a file

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

const to = ClusterExpansions.to # the timer that the package code writes to

# The helper functions of this script, in ../tools/:
#   timing_io.jl:    timer_rows, write_timer_csv, read_timer_csv, new_run_dir, existing_run_dir
#   timing_plots.jl: make_timing_phases_plot, make_timing_self_plot, ...
#   compare_tools.jl: check_cost, build_pepos, append_row, read_results, make_spread_plot, ...
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
const D_CE = 2
const chipeps = 2
const chienv = 8
const Δβ_iter = 0.005
# The β values of the observations (increasing; every value must be a multiple of Δβ_iter).
const β_obs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

const SEED_BASE = 1000              # observation i uses Random.seed!(SEED_BASE + i), the same in all 4 cases

# Time limits (seconds). Total of the run is about: compile of `true` (~6000 s) + compile of `false` (a few minutes) + 4 * RUN_BUDGET_S (+ at most one LBFGS iteration overshoot per observation).
const RUN_BUDGET_S = 2400.0         # per case: 40 minutes
const MAX_SECONDS_OBS = 420.0       # per observation (the per-observation limit is lowered further when the case budget is running out)
const MIN_SECONDS_OBS = 30.0        # but every observation gets at least this

# Settings of the vbPEPS (the same for all cases, except warm_start and differentiate_environment which are the 4 cases)
const base_kwargs = (;
    boundary_alg_type = SimultaneousCTMRG, symmetrization = nothing,
    boundary_tol = 1.0e-10, boundary_maxiter = 1500, boundary_verbosity = 1,
    gradient_tol = 1.0e-6, gradient_maxiter = 5, gradient_krylovdim = 20,
    optimizer_memory = 64, optimizer_tol = 1.0e-5, optimizer_maxiter = 10,
    maxiter_init = 20,
    optimizer_verbosity = 2,
)

# (name, warm_start, differentiate_environment); the order matters: the `true` cases come last, because their compilation takes very long
const CASES = [
    ("false_cold", false, false),
    ("false_warm", true, false),
    ("true_cold", false, true),
    ("true_warm", true, true),
]



# If REPLOT_DIR is the NAME of a run folder (for example "20261111_1248" or "true_cold") in data/profiling/comparing/ising_model_3D_CE_vbPEPS_compare/ with a results.csv,
# we only redo all the plots from it and nothing else is run (no calculation, no compilation of the pullback). Leave it "" for a normal run.
const REPLOT_DIR = "20261111_1248"

# Settings of the spread plot (|m| of every case minus the mean |m| of the cases, per β)
const SPREAD_EXCLUDE_CASES = String[]       # cases that we leave out completely, for example ["true_warm"]
const SPREAD_EXCLUDE_FLAGGED = true         # leave out observations that failed or whose check is not "ok" (unphysical f)
const SPREAD_EXCLUDE_TIMED_OUT = true      # leave out observations that were stopped by the time limit
const SPREAD_OUTLIER_TOL = 0.1             # leave out a point whose |m| differs more than this from the median |m| of the cases at the same β

# Settings of the timing plots (4 bars per section, one per case)
const TIMING_TOP = 4                     # number of sections in the self-time plot
const CASE_COLORS = Dict("false_cold" => :steelblue, "false_warm" => :lightskyblue, "true_cold" => :firebrick, "true_warm" => :lightsalmon)   # blue = fixed env, red = diff env, dark = cold, light = warm

# ======================================================================================================================
# One case
# ======================================================================================================================
function run_case(name, warm, diff, Os, pepsspace, envspace, obs, resultspath, outdir)
    # name = e.g. "true_warm"

    @info "===== Case $name: warm_start = $warm, differentiate_environment = $diff, budget = $(RUN_BUDGET_S) s ====="
    reset_timer!(to)
    ClusterExpansions.reset_max_alloc!()
    kwargs = (; base_kwargs..., differentiate_environment = diff)
    state = nothing           # holds (psi, env2, env3) for the warm start
    t_case = time()

    # Loop over all observations (at different β, using different O's)
    for i in eachindex(Os)
        β = β_obs[i]
        # Use same random seed in the different cases
        Random.seed!(SEED_BASE + i)
        # time we allow this observation: at most MAX_SECONDS_OBS, and lower if the case budget runs out
        # RUN_BUDGET_S - (time() - t_case) = the budget left
        # length(Os) - i + 1 = observations left (including this one)
        # We allow twice the fair share (*2)
        # We keep this budget between 30 s and 420 s.
        allowed = clamp(2 * (RUN_BUDGET_S - (time() - t_case)) / (length(Os) - i + 1), MIN_SECONDS_OBS, MAX_SECONDS_OBS)
        
        # Start of observation
        t_obs = time()
        try
            res = ClusterExpansions.@timeit to "expectation_value (vbPEPS and observable)" begin
                # The actual optimisation (may be with warm start)
                result = ClusterExpansions.bpeps_optimize(Os[i], pepsspace, envspace; state0 = warm ? state : nothing, max_seconds = allowed, kwargs...)
                # Measures ⟨σᶻ⟩ on the converged state
                # Returns a named tuple as res
                (; result, m = ClusterExpansions.bpeps_observables(Os[i], [obs], result)[1])
            end
            # Total time of this observation (optimisation + observation)
            dt = time() - t_obs
            # See if f is realistic
            check = check_cost(β, res.result.cost)
            
            # Print/save result
            println("OBS $name β = $(round(β; digits = 3)): f = $(res.result.cost), calls = $(res.result.numfg), time = $(round(dt; digits = 1)) s, timed out = $(res.result.timed_out), m = $(res.m), ", check)
            append_row(resultspath, name, warm, diff, β, res.result.cost, res.result.numfg, dt, res.result.timed_out, ComplexF64(res.m), check)
            
            # If warm, then update state to the current psi and env's (even if not enough time)
            warm && (state = (res.result.psi, res.result.env2, res.result.env3))
        # Runs if anything inside the try block threw an error
        catch err
            # Time untill failure
            dt = time() - t_obs
            # Print/Save the result
            @error "OBS $name β = $(round(β; digits = 3)) failed" exception = (err, catch_backtrace())
            append_row(resultspath, name, warm, diff, β, NaN, 0, dt, false, complex(NaN, NaN), "ERROR: " * first(sprint(showerror, err), 150))
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
    @info "Warm-up (compilation) for differentiate_environment = $diff: not measured"
    t0 = time()
    tiny = (; base_kwargs..., differentiate_environment = diff, optimizer_maxiter = 1, maxiter_init = 2, boundary_maxiter = 100)
    Random.seed!(SEED_BASE)
    r1 = ClusterExpansions.bpeps_optimize(Os[1], pepsspace, envspace; tiny...)
    # second call with a warm start, so that the warm-start path is compiled as well
    r2 = ClusterExpansions.bpeps_optimize(Os[2], pepsspace, envspace; state0 = (r1.psi, r1.env2, r1.env3), tiny...)
    ClusterExpansions.bpeps_observables(Os[2], [obs], r2)
    @info "Warm-up for differentiate_environment = $diff took $(round(time() - t0; digits = 1)) s"
    return nothing
end

# ======================================================================================================================
# Summary table and plots (from results.csv)
# ======================================================================================================================
function print_summary(rows)
    println("\n===== Summary per case =====")
    println(rpad("case", 12), rpad("n obs", 7), rpad("total time (s)", 16), rpad("calls", 8), rpad("timed out", 11), "failed")
    for (name, _, _) in CASES
        sub = filter(r -> r.case == name, rows)
        isempty(sub) && continue
        println(rpad(name, 12), rpad(length(sub), 7), rpad(round(sum(r.time_s for r in sub); digits = 1), 16), rpad(sum(r.ncalls for r in sub), 8),
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

function make_plots(rows, outdir)
    colour(diff) = diff ? :crimson : :royalblue      # colour: differentiate_environment
    style(warm) = warm ? :dash : :solid              # line style: warm start (dashed) or cold start (solid)
    p_m = plot(; xlabel = "β", ylabel = "|⟨σᶻ⟩|", title = "magnetization (g = 0: 0 below βc)", legend = :topleft)
    p_f = plot(; xlabel = "β", ylabel = "f = -log(λ)", title = "cost (bound: f ≤ -3β)", legend = false)
    p_t = plot(; xlabel = "β", ylabel = "time per observation (s)", title = "time", yscale = :log10, legend = false)
    for (name, warm, diff) in CASES
        sub = sort(filter(r -> r.case == name, rows); by = r -> r.β)
        isempty(sub) && continue
        βs = [r.β for r in sub]
        label = "$(diff ? "diff env" : "fixed env"), $(warm ? "warm" : "cold")"
        for (p, ys, lab) in ((p_m, [hypot(r.m_re, r.m_im) for r in sub], label), (p_f, [r.f for r in sub], ""), (p_t, [max(r.time_s, 1.0e-1) for r in sub], ""))
            plot!(p, βs, ys; color = colour(diff), linestyle = style(warm), marker = :circle, markersize = 4, label = lab)
            idx = findall(r -> r.timed_out, sub)
            isempty(idx) || scatter!(p, βs[idx], ys[idx]; color = colour(diff), markershape = :xcross, markersize = 9, markerstrokewidth = 2, label = "")
        end
    end
    βline = sort(unique(r.β for r in rows))
    plot!(p_f, βline, -3 .* βline; color = :black, linestyle = :dot, label = "")
    vline!(p_m, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    vline!(p_f, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    plt = plot(p_m, p_f, p_t; layout = (1, 3), size = (1600, 480), margin = 6Plots.mm, bottom_margin = 8Plots.mm, left_margin = 8Plots.mm,
        plot_title = "vbPEPS: χpeps = $chipeps, χenv = $chienv, D_CE = $D_CE, g = $g (cross = timed out; dotted gray = βc)")
    savefig(plt, joinpath(outdir, "compare_vbpeps.png"))
    return nothing
end

# Label, colour and line style of a case in all plots (colour: differentiate_environment, line style: warm start)
case_label(warm, diff) = "$(diff ? "diff env" : "fixed env"), $(warm ? "warm" : "cold")"

# All plots; each one in its own try, so that a failing plot does not stop the others
function make_all_plots(rows, outdir)
    spread_specs = [(; name, label = case_label(warm, diff), color = diff ? :crimson : :royalblue, linestyle = warm ? :dash : :solid) for (name, warm, diff) in CASES]
    timing_specs = [(name, case_label(warm, diff), CASE_COLORS[name]) for (name, warm, diff) in CASES]
    plots = [
        "compare_vbpeps.png" => () -> make_plots(rows, outdir),
        "compare_vbpeps_spread.png" => () -> make_spread_plot(rows, outdir, spread_specs; file = "compare_vbpeps_spread.png",
            exclude_cases = SPREAD_EXCLUDE_CASES, exclude_flagged = SPREAD_EXCLUDE_FLAGGED, exclude_timed_out = SPREAD_EXCLUDE_TIMED_OUT, outlier_tol = SPREAD_OUTLIER_TOL),
        "compare_vbpeps_timing_phases.png" => () -> make_timing_phases_plot(outdir, timing_specs; label = "vbPEPS, 4 cases", file_prefix = "compare_vbpeps"),
        "compare_vbpeps_timing_self.png" => () -> make_timing_self_plot(outdir, timing_specs; label = "vbPEPS, 4 cases", file_prefix = "compare_vbpeps", top = TIMING_TOP),
    ]
    for (file, makeplot) in plots
        try
            makeplot()
            @info "Plot saved: $(joinpath(outdir, file))"
        catch err
            @error "Plot $file failed, the data is in results.csv (set REPLOT_DIR to retry)" exception = (err, catch_backtrace())
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
        plotdir = existing_run_dir(@__FILE__, REPLOT_DIR)
        rows = read_results(joinpath(plotdir, "results.csv"))
        print_summary(rows)
        make_all_plots(rows, plotdir)
        return nothing
    end

    # Create the output folder and csv path using current date/time
    t_run = time()
    outdir = new_run_dir(@__FILE__)
    resultspath = joinpath(outdir, "results.csv")

    # Create results.csv, with only the header
    open(io -> println(io, CSV_HEADER), resultspath, "w")

    # Write Julia version, threads and all settings to the file 
    open(joinpath(outdir, "settings.txt"), "w") do io
        println(io, "Julia $(VERSION), threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())")
        println(io, "J = $J, g = $g, D_CE = $D_CE, chipeps = $chipeps, chienv = $chienv, Δβ_iter = $Δβ_iter, β_obs = $β_obs")
        println(io, "SEED_BASE = $SEED_BASE, RUN_BUDGET_S = $RUN_BUDGET_S, MAX_SECONDS_OBS = $MAX_SECONDS_OBS, MIN_SECONDS_OBS = $MIN_SECONDS_OBS")
        println(io, "base_kwargs = $base_kwargs")
        println(io, "cases = $CASES")
    end
    @info "Output directory: $outdir"
    @info "Threads.nthreads() = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())"

    # Define the virtual spaces and the observables
    pepsspace = ℂ^chipeps
    envspace = ℂ^chienv
    obs = SpinOperators.σᶻ(T, spin_symmetry)

    # Build all PEPOs + print time
    t0 = time()
    Os = build_pepos(β_obs, Δβ_iter, D_CE, J, g)
    @info "CE PEPOs built in $(round(time() - t0; digits = 1)) s"

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
            run_case(name, warm, diff, Os, pepsspace, envspace, obs, resultspath, outdir)
        catch err
            @error "Case $name failed" exception = (err, catch_backtrace())
        end
    end

    rows = read_results(resultspath)
    print_summary(rows)
    make_all_plots(rows, outdir)
    @info "Total time of the run: $(round((time() - t_run) / 3600; digits = 2)) h"
    return nothing
end

main()
