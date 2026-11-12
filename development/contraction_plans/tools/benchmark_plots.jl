# benchmark_plots.jl -- the data side and the plots of development/contraction_plans/benchmark_contract_onesite.jl.
#
# Contents:
#   Where the data goes        data/contraction_plans/<script name>/<run>/   (bench_run_dir, existing_bench_run_dir)
#   results.csv of a run       BENCH_HEADER, append_bench_row, read_bench_results   (ONE row per timed plan, all repetitions in it)
#   Plots                      make_times_figure (ONE figure, 2 panels), and the list BENCH_PLOTS that
#                              make_all_benchmark_plots runs at the end of the benchmark
#
# More plots can be made/added later.
# `rows` are the rows of results.csv (see read_bench_results): they have the fields below, so a new plot never needs the benchmark to run again.

using Plots
using Plots.PlotMeasures # for `mm`, used for plot margins

# ======================================================================================================================
# Where the data of a script goes: data/contraction_plans/<script name>/<run>/
# ======================================================================================================================
# Every script has ONE folder, every run of the script is a subfolder named by the date and time of the run
# (for example 20261111_2230), with all files of that run in it (results.csv, settings.txt, the plots). RUN_DIR in the script is the name of such a run folder.
const BENCH_DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "contraction_plans"))      # <repo>/data/contraction_plans

# A new run folder (date and time of now) of the script; it is created. script_file is @__FILE__ of the script.
function bench_run_dir(script_file)
    dir = joinpath(BENCH_DATA_DIR, splitext(basename(script_file))[1], Libc.strftime("%Y%m%d_%H%M", time()))
    mkpath(dir)
    return dir
end

# The run folder with this name (the RUN_DIR of the script); an error that lists the existing run folders if there is none.
function existing_bench_run_dir(script_file, name)
    base = joinpath(BENCH_DATA_DIR, splitext(basename(script_file))[1])
    dir = joinpath(base, name)
    isdir(dir) && return dir
    runs = isdir(base) ? sort(filter(d -> isdir(joinpath(base, d)), readdir(base))) : String[]
    error("RUN_DIR = \"$name\": there is no folder $dir. Run folders of this script: " * (isempty(runs) ? "none" : join(runs, ", ")))
end

# ======================================================================================================================
# results.csv: one row per timed plan
# ======================================================================================================================
# Columns:
#   chi_env, chi_peps, D   bond dimensions (the mock environment is random and dense with these dimensions)
#   name, source           e.g. TREE_CURRENT#1, TREE_CURRENT: name of the plan and its contraction source
#   ram_gib                RAM budget (GiB) that the plan was made for (column ram_gib of the plan file)
#   cuts, slices           cut bonds (separated by a space) and number of slices
#   flops, peak_gib        cost model: flops of the whole sliced contraction (all slices) and peak of the intermediates in GiB
#   times_s                wall time of EVERY repetition, separated by a space (no compilation in it)
#   tracked_gib, rss_gib, pool_gib   peak of all temporaries (TrackingAllocator), peak resident memory of the child, memory asked from the OS by the reuse allocator
#   allocator, env, blas_threads    allocator ("reuse" / "manual"), environment ("mock seed 1" / "real"), BLAS threads
#   env_fp                 fingerprint of the environment: sum of the norms of the 27 tensors + M (equal for all plans of the same dimensions if they saw the SAME environment)
#   value_re, value_im     value of the contraction
#   check                  relative deviation of the value from the first plan of the same dimensions
#   note                   empty, or why the plan was skipped / failed (then times_s is empty)
const BENCH_HEADER = "chi_env,chi_peps,D,name,source,ram_gib,cuts,slices,flops,peak_gib,times_s,tracked_gib,rss_gib,pool_gib,allocator,env,blas_threads,env_fp,value_re,value_im,check,note"
const BENCH_NCOLS = 22

# Append one row (a NamedTuple with the fields below) to results.csv; the file is closed after every row, so the data is on disk.
# r.tmin and r.times: tmin is not written, it is recomputed when reading.
function append_bench_row(path, r)
    note = replace(String(r.note), ',' => ';', '\n' => ' ', '\r' => ' ')
    fields = (r.chi_env, r.chi_peps, r.D, r.name, r.source, r.ram_gib, r.cuts, r.slices, r.flops, r.peak, join(r.times, " "), r.tracked, r.rss, r.pool,
              r.alloc, r.env, r.blas, r.fp, r.re, r.im, r.check, note)
    open(path, "a") do io
        println(io, join(fields, ","))
    end
    return nothing
end

# Read results.csv back into a Vector of NamedTuples with the fields
#   chi_env, chi_peps, D, name, source, ram_gib, cuts, slices, flops, peak, times (Vector), tmin (fastest repetition, NaN if none), tracked, rss, pool, alloc, env, blas, fp, re, im, check, note
# A line that is cut off (for example because the computer was switched off while writing) is ignored.
function read_bench_results(path)
    rows = NamedTuple[]
    isfile(path) || return rows
    for line in Iterators.drop(eachline(path), 1)
        isempty(strip(line)) && continue
        p = split(line, ','; limit = BENCH_NCOLS)
        length(p) == BENCH_NCOLS || continue
        num(i) = (s = strip(p[i]); isempty(s) ? NaN : parse(Float64, s))
        times = isempty(strip(p[11])) ? Float64[] : parse.(Float64, split(strip(p[11])))
        push!(rows, (; chi_env = parse(Int, p[1]), chi_peps = parse(Int, p[2]), D = parse(Int, p[3]), name = String(p[4]), source = String(p[5]), ram_gib = num(6),
            cuts = String(p[7]), slices = num(8), flops = num(9), peak = num(10), times, tmin = isempty(times) ? NaN : minimum(times),
            tracked = num(12), rss = num(13), pool = num(14), alloc = String(p[15]), env = String(p[16]), blas = round(Int, num(17)), fp = num(18),
            re = num(19), im = num(20), check = num(21), note = String(strip(p[22]))))
    end
    return rows
end

# ======================================================================================================================
# Helpers for the plots
# ======================================================================================================================
# The four kinds of trees, always in this order with this colour and marker (the colour follows the tree, never its rank).
# Colours: the first four slots of the categorical palette of the dataviz skill; markers differ too, so that the plots can also be read without colour.
# (If a 5th kind is ever added, it needs a marker and colour here.)
const TREE_KINDS = [
    (; kind = "opt-true", color = "#eb6834", marker = :utriangle),
    (; kind = "ome-flops", color = "#1baf7a", marker = :diamond),
    (; kind = "current", color = "#2a78d6", marker = :circle),
    (; kind = "ome-sc", color = "#eda100", marker = :square),
]

# "TREE_CURRENT" -> "current", "OPT_TRUE" -> "opt-true", "ome-flops" -> "ome-flops", "ome-sc23" (any number) -> "ome-sc"; anything else is returned as it is
# (the old name TREE_FAST of OPT_TRUE is accepted too, so that results.csv files of runs before the rename can still be plotted)
function tree_kind(source)
    source == "TREE_CURRENT" && return "current"
    (source == "OPT_TRUE" || source == "TREE_FAST") && return "opt-true"
    startswith(source, "ome-sc") && return "ome-sc"
    return String(source)
end

bench_dims(r) = (r.chi_env, r.chi_peps, r.D)
dims_label(d) = "$(d[1]),$(d[2]),$(d[3])"

# The dimension sets that have at least one finite time, sorted by the mean time of their plans (fastest first, so the slowest set is at the right of the plots)
function dims_order(rows)
    ok = filter(r -> isfinite(r.tmin), rows)
    dims = unique(bench_dims(r) for r in ok)
    meantime(d) = (v = [r.tmin for r in ok if bench_dims(r) == d]; sum(v) / length(v))
    return sort(dims; by = meantime)
end

# Small horizontal shift of the 4 kinds around the integer x position of a dimension set, so that the points of one set do not lie on top of each other
dodge(k) = (k - (length(TREE_KINDS) + 1) / 2) * 0.1

# The text under the title: what is fixed in the run (taken from the rows, so it is also right when the plots are made again later)
function bench_label(rows)
    ok = filter(r -> isfinite(r.tmin), rows)
    isempty(ok) && return "no finished plan"
    uniq(f) = join(sort(unique(string(f(r)) for r in ok)), "/")
    reps = maximum(length(r.times) for r in ok)
    return "contract_onesite (27 tensors, dense), environment: $(uniq(r -> r.env)), plans made for RAM budget $(uniq(r -> r.ram_gib)) GiB, allocator $(uniq(r -> r.alloc)), " *
           "BLAS threads $(uniq(r -> r.blas)), time = fastest of $reps repetitions\n" *
           "varied: (χenv, χpeps, D) and tree + cut bonds        fixed: everything else"
end

# ======================================================================================================================
# The plots
# ======================================================================================================================
# Limits and tick positions of the log time axis: ticks at 1, 2, 3, 5, 7, 10, 20, ... seconds written as plain numbers (instead of 10^0.6)
function log_time_axis(rows)
    ts = [r.tmin for r in rows if isfinite(r.tmin)]
    lo, hi = minimum(ts) / 1.3, maximum(ts) * 1.3
    vals = Float64[]
    for e in floor(Int, log10(lo)):ceil(Int, log10(hi)), m in (1, 2, 3, 5, 7)
        v = m * 10.0^e
        lo <= v <= hi && push!(vals, v)
    end
    return (lo, hi), (vals, [v >= 1 ? string(round(Int, v)) : string(round(v; sigdigits = 2)) for v in vals])
end

# The times of the repetitions of a plan that are NOT the fastest one (the fastest is r.tmin); with REPS = 2 this is the slower repetition
function other_times(r)
    ts = filter(isfinite, collect(r.times))
    isempty(ts) || deleteat!(ts, argmin(ts))
    return ts
end

# Left panel: the time of every plan (solid point = fastest repetition, see-through points = the other repetitions);
# x = the dimension sets (sorted by mean time), y = time in seconds (log), one colour per tree. No lines.
function plot_bench_times(rows)
    order = dims_order(rows)
    n = length(order)
    xpos = Dict(d => i for (i, d) in enumerate(order))
    ylims, yticks = log_time_axis(rows)
    p = plot(; xlabel = "(χenv, χpeps, D), sorted by mean time", ylabel = "time of the contraction [s]", yscale = :log10, ylims, yticks, legend = :topleft,
        title = "time per plan", xticks = (1:n, dims_label.(order)), xrotation = 45, xlims = (0.5, n + 0.5), minorgrid = true, gridalpha = 0.25, minorgridalpha = 0.12)
    for (k, tk) in enumerate(TREE_KINDS)
        pts = sort([r for r in rows if isfinite(r.tmin) && tree_kind(r.source) == tk.kind]; by = r -> xpos[bench_dims(r)])
        isempty(pts) && continue
        xo = Float64[]; yo = Float64[]
        for r in pts, t in other_times(r)
            push!(xo, xpos[bench_dims(r)] + dodge(k)); push!(yo, t)
        end
        isempty(xo) || scatter!(p, xo, yo; color = tk.color, marker = tk.marker, markersize = 7, markeralpha = 0.35, markerstrokewidth = 0, label = "")
        scatter!(p, [xpos[bench_dims(r)] + dodge(k) for r in pts], [r.tmin for r in pts]; color = tk.color,
            marker = tk.marker, markersize = 7, markerstrokecolor = :white, markerstrokewidth = 1, label = tk.kind)
    end
    return p
end

# Right panel: the same points, but y = (time - mean time of this dimension set) / mean time, in percent. The mean is over the (fastest repetitions of the) plans of the set
# that have a time. Solid point = fastest repetition, see-through points = the other repetitions, and a bar from the fastest to the slowest repetition of the plan
# (the bar is as long as the real spread: a plan whose repetitions differ by 1 % has a bar of 1 % of the mean on this axis, which is only a few pixels).
# A set with only one plan has no deviation and is left out.
function plot_bench_deviation(rows)
    order = dims_order(rows)
    n = length(order)
    xpos = Dict(d => i for (i, d) in enumerate(order))
    means = Dict{Any,Float64}()
    for d in order
        v = [r.tmin for r in rows if isfinite(r.tmin) && bench_dims(r) == d]
        length(v) >= 2 && (means[d] = sum(v) / length(v))
    end
    p = plot(; xlabel = "(χenv, χpeps, D), sorted by mean time", ylabel = "(time - mean of the set) / mean of the set [%]", legend = false,
        title = "deviation from the mean of the dimension set (bar: fastest to slowest repetition)", xticks = (1:n, dims_label.(order)), xrotation = 45, xlims = (0.5, n + 0.5),
        gridalpha = 0.25)
    hline!(p, [0.0]; color = :black, linewidth = 1, label = "")
    for (k, tk) in enumerate(TREE_KINDS)
        pts = [r for r in rows if isfinite(r.tmin) && tree_kind(r.source) == tk.kind && haskey(means, bench_dims(r))]
        isempty(pts) && continue
        m(r) = means[bench_dims(r)]
        xo = Float64[]; yo = Float64[]; xb = Float64[]; yb = Float64[]
        for r in pts
            x = xpos[bench_dims(r)] + dodge(k)
            ts = other_times(r)
            for t in ts
                push!(xo, x); push!(yo, 100 * (t - m(r)) / m(r))
            end
            if !isempty(ts)
                append!(xb, (x, x, NaN))
                append!(yb, (100 * (r.tmin - m(r)) / m(r), 100 * (maximum(ts) - m(r)) / m(r), NaN))
            end
        end
        isempty(xb) || plot!(p, xb, yb; color = tk.color, linewidth = 3, linealpha = 0.5, label = "")
        isempty(xo) || scatter!(p, xo, yo; color = tk.color, marker = tk.marker, markersize = 7, markeralpha = 0.35, markerstrokewidth = 0, label = "")
        scatter!(p, [xpos[bench_dims(r)] + dodge(k) for r in pts], [100 * (r.tmin - m(r)) / m(r) for r in pts];
            color = tk.color, marker = tk.marker, markersize = 7, markerstrokecolor = :white, markerstrokewidth = 1, label = tk.kind)
    end
    return p
end

# ONE figure with the two panels next to each other
function make_times_figure(rows)
    return plot(plot_bench_times(rows), plot_bench_deviation(rows); layout = (1, 2), size = (1900, 760), left_margin = 10mm, bottom_margin = 12mm, top_margin = 4mm,
        plot_title = bench_label(rows), plot_titlefontsize = 10)
end

# All plots that are made at the end of the benchmark: file name => function(rows) returning a plot. A new plot = a new entry here.
const BENCH_PLOTS = Pair{String,Function}[
    "benchmark_times.png" => make_times_figure,
]

# Makes every plot of BENCH_PLOTS from the rows and saves it in outdir. Every plot is made in its own try, so that a plot that cannot be made does not stop the others
# (the data is in results.csv: set RUN_DIR in the benchmark script to make the plots again).
function make_all_benchmark_plots(rows, outdir)
    if !any(r -> isfinite(r.tmin), rows)
        @warn "no finished plan in the results: no plots made (the plans were skipped or failed, see the note column of results.csv)"
        return nothing
    end
    for (file, makeplot) in BENCH_PLOTS
        try
            plt = makeplot(rows)
            savefig(plt, joinpath(outdir, file))
            @info "Plot saved: $(joinpath(outdir, file))"
        catch err
            @error "Plot $file failed, the data is in results.csv (set RUN_DIR to retry)" exception = (err, catch_backtrace())
        end
    end
    return nothing
end