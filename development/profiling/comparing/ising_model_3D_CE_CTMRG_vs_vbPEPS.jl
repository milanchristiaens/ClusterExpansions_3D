# Comparison of selected cases of the two Ising compare scripts of this folder, 3D Ising model, g = 0:
#
#     ising_model_3D_CE_CTMRG_compare.jl  (CE + CTMRG3D)   and   ising_model_3D_CE_vbPEPS_compare.jl  (CE + variational boundary PEPS)
#
# Nothing is run: we only read the results file of all Ising compare runs, data/profiling/comparing/ising_model_3D_CE_results.csv
# (see ISING_CSV_HEADER in ../tools/compare_tools.jl), take the cases of SELECTION (see the settings) and make one figure:
#     compare_algorithms_magnetization.png (left: |m| against β; right: |m| of every selected case minus the mean of the selected cases)
# It goes to a new run folder  data/profiling/comparing/ising_model_3D_CE_CTMRG_vs_vbPEPS/<date>_<time>/  together with selection.txt
# (the selection, with the run that "" stood for).
#
# What the comparison means:
#   - Only cases with the same PEPO (the same D_CE and Δβ) differ only in the contraction (χpeps, χenv, algorithm): warning if they differ.
#   - The time budget of a case (budget_s) and the number of BLAS threads are printed per selected case. On Hortense the two algorithms have the same CORE budget
#     (cores × budget_s: CTMRG3D 4 cores × 16 h, vbPEPS 2 cores × 32 h), so budget_s differs between them on purpose; the cores are in the job scripts.
#   - There is no reference value yet: the right panel compares the cases with each other (the mean of the selected cases), not with an exact m.
# A cross marks an observation that was stopped by its time limit (both algorithms) or that did not converge (CTMRG3D).
#
# Note on m: at g = 0 the exact <σᶻ> is 0 below βc (Z2 symmetry) and has an arbitrary sign above βc, so we plot |m|.

ENV["GKSwstype"] = "nul"     # no plot window: the plots are only saved to a file

using ClusterExpansions
using TimerOutputs
using Plots
using LinearAlgebra

# The helper functions of this script, in ../tools/:
#   timing_io.jl:     new_run_dir
#   compare_tools.jl: ISING_RESULTS_PATH, read_ising_results, plot_spread, βc_ising
include(joinpath(@__DIR__, "..", "tools", "timing_io.jl"))
include(joinpath(@__DIR__, "..", "tools", "compare_tools.jl"))

# ======================================================================================================================
# Settings
# ======================================================================================================================
# The selected cases: (algorithm, run, case)
#   algorithm = "CTMRG3D" or "vbPEPS"
#   run       = name of the run folder of that compare script (for example "20261116_1500"); "" = the latest run of this algorithm that has this case
#   case      = name of the case in that script: CTMRG3D "svd_cold", "svd_warm", "oblique_cold", "oblique_warm";
#               vbPEPS "false_cold", "false_warm", "true_cold", "true_warm"
const SELECTION = [
    ("CTMRG3D", "", "svd_warm"),
    ("CTMRG3D", "", "oblique_warm"),
    ("vbPEPS", "", "false_warm"),
    ("vbPEPS", "", "true_warm"),
]

# Settings of the spread plot (|m| of every selected case minus the mean |m| of the selected cases, per β)
const SPREAD_EXCLUDE_FLAGGED = false        # leave out observations that failed or whose check is not "ok"
const SPREAD_EXCLUDE_TIMED_OUT = false      # leave out observations that were stopped by the time limit
const SPREAD_OUTLIER_TOL = 0.1              # leave out a point whose |m| differs more than this from the median |m| of the cases at the same β

# Colours per algorithm (the n-th selected case of an algorithm gets the n-th colour), marker per algorithm, line style: dashed = warm, solid = cold
const ALG_COLORS = Dict("CTMRG3D" => [:royalblue, :deepskyblue, :navy, :teal], "vbPEPS" => [:crimson, :darkorange, :darkred, :magenta])
const ALG_MARKER = Dict("CTMRG3D" => :circle, "vbPEPS" => :diamond)

# ======================================================================================================================
# Selection
# ======================================================================================================================
# The rows of the selected cases, with case replaced by a key "algorithm case (run)" (unique, also when two runs have the same case name),
# and one plot spec per selected case. A selection without rows is left out with a warning.
function selected_rows(all)
    rows = NamedTuple[]
    specs = NamedTuple[]
    nsel = Dict("CTMRG3D" => 0, "vbPEPS" => 0)
    for (alg, run, case) in SELECTION
        haskey(ALG_MARKER, alg) || error("unknown algorithm \"$alg\" in SELECTION: use \"CTMRG3D\" or \"vbPEPS\"")
        cand = filter(r -> r.algorithm == alg && r.case == case, all)
        # "" = the latest run (run names are date_time, so the largest string is the latest)
        isempty(run) && !isempty(cand) && (run = maximum(r.run for r in cand))
        sub = filter(r -> r.run == run, cand)
        if isempty(sub)
            @warn "no rows for ($alg, \"$run\", $case) in $ISING_RESULTS_PATH: left out"
            continue
        end
        key = "$alg $case ($run)"
        nsel[alg] += 1
        cols = ALG_COLORS[alg]
        r1 = sub[1]
        push!(specs, (; name = key, alg, run, case, label = "$alg $case [$run]: D_CE $(r1.D_CE), χpeps $(r1.chipeps), χenv $(r1.chienv)",
                      color = cols[mod1(nsel[alg], length(cols))], linestyle = r1.warm ? :dash : :solid, marker = ALG_MARKER[alg]))
        append!(rows, [merge(r, (; case = key)) for r in sub])
    end
    return rows, specs
end

# ======================================================================================================================
# Summary table and warnings
# ======================================================================================================================
function print_summary(rows, specs)
    println("\n===== Selected cases =====")
    println(rpad("case", 44), rpad("D_CE", 6), rpad("χpeps", 7), rpad("χenv", 6), rpad("Δβ", 8), rpad("budget (s)", 12), rpad("BLAS", 6), rpad("n obs", 7),
        rpad("total time (s)", 16), rpad("timed out", 11), rpad("not conv.", 11), rpad("max overshoot (s)", 19), "failed")
    for c in specs
        sub = filter(r -> r.case == c.name, rows)
        r1 = sub[1]
        # overshoot of the time limit, for the observations that it stopped: CTMRG3D = time of the iterations - allowed,
        # vbPEPS = time of the whole observation - allowed (an upper bound: it includes the contraction of the observable)
        over(r) = (r.algorithm == "CTMRG3D" ? r.t_ctmrg : r.time_s) - r.allowed_s
        println(rpad(c.name, 44), rpad(r1.D_CE, 6), rpad(r1.chipeps, 7), rpad(r1.chienv, 6), rpad(r1.dβ, 8), rpad(r1.budget_s, 12), rpad(r1.blas, 6), rpad(length(sub), 7),
            rpad(round(sum(r.time_s for r in sub); digits = 1), 16), rpad(count(r -> r.timed_out, sub), 11), rpad(count(r -> r.converged === false, sub), 11),
            rpad(round(maximum((over(r) for r in sub if r.timed_out); init = 0.0); digits = 1), 19), count(r -> !isfinite(r.m_re), sub))
    end
    # the same PEPO, the same budget, the same number of BLAS threads?
    for (what, f) in (("D_CE", r -> r.D_CE), ("Δβ", r -> r.dβ))
        vals = unique(f(r) for r in rows)
        length(vals) > 1 && @warn "The selected cases have different $what: $vals (different PEPOs: not only the contraction differs)"
    end
    return nothing
end

# ======================================================================================================================
# Figure
# ======================================================================================================================
function plot_magnetization(rows, specs)
    p = plot(; xlabel = "β", ylabel = "|⟨σᶻ⟩|", title = "magnetization (g = 0: 0 below βc)", legend = :topleft, legendfontsize = 7)
    for c in specs
        sub = sort(filter(r -> r.case == c.name && isfinite(r.m_re), rows); by = r -> r.β)
        isempty(sub) && continue
        βs = [r.β for r in sub]
        ys = [hypot(r.m_re, r.m_im) for r in sub]
        plot!(p, βs, ys; color = c.color, linestyle = c.linestyle, marker = c.marker, markersize = 4, label = c.label)
        idx = findall(r -> r.timed_out || r.converged === false, sub)
        isempty(idx) || scatter!(p, βs[idx], ys[idx]; color = c.color, markershape = :xcross, markersize = 9, markerstrokewidth = 2, label = "")
    end
    vline!(p, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    return p
end

function make_figure(rows, specs, outdir)
    p_dev = plot_spread(rows, specs; exclude_flagged = SPREAD_EXCLUDE_FLAGGED, exclude_timed_out = SPREAD_EXCLUDE_TIMED_OUT, outlier_tol = SPREAD_OUTLIER_TOL)
    plt = plot(plot_magnetization(rows, specs), p_dev; layout = (1, 2), size = (1700, 620), margin = 6Plots.mm, bottom_margin = 8Plots.mm, left_margin = 8Plots.mm,
        plot_title = "CTMRG3D (blue, circles) against vbPEPS (red, diamonds); dashed = warm start (cross = timed out or not converged; dotted gray = βc)",
        plot_titlefontsize = 10)
    file = joinpath(outdir, "compare_algorithms_magnetization.png")
    savefig(plt, file)
    @info "Plot saved: $file"
    return nothing
end

# ======================================================================================================================
# Main
# ======================================================================================================================
function main()
    isfile(ISING_RESULTS_PATH) || error("there is no results file $ISING_RESULTS_PATH: run the two compare scripts first")
    rows, specs = selected_rows(read_ising_results())
    isempty(specs) && error("none of the cases of SELECTION has rows in $ISING_RESULTS_PATH")

    outdir = new_run_dir(@__FILE__)
    open(joinpath(outdir, "selection.txt"), "w") do io
        println(io, "results file: $ISING_RESULTS_PATH")
        println(io, "SELECTION = $SELECTION")
        println(io, "used (algorithm, run, case): ", join(["($(c.alg), $(c.run), $(c.case))" for c in specs], ", "))
        println(io, "SPREAD_EXCLUDE_FLAGGED = $SPREAD_EXCLUDE_FLAGGED, SPREAD_EXCLUDE_TIMED_OUT = $SPREAD_EXCLUDE_TIMED_OUT, SPREAD_OUTLIER_TOL = $SPREAD_OUTLIER_TOL")
    end
    @info "Output directory: $outdir"

    print_summary(rows, specs)
    try
        make_figure(rows, specs, outdir)
    catch err
        @error "Plot compare_algorithms_magnetization.png failed" exception = (err, catch_backtrace())
    end
    return nothing
end

main()
