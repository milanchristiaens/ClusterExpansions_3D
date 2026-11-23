# The figures of the checks, written next to the data of each check: the init, exponent, west_updates, diff_env and decompositions checks, and PEPSKit against BetterPEPSKit (vbPEPS compare).
# The checks themselves run on Hortense or the laptop; copy their data to the laptop first.

using ClusterExpansions, Plots, Printf
using Plots.PlotMeasures

# Settings: the data per check, in data/checks/<check>/:
#   "all"  = every CSV file of the check together (a run that was repeated counts once, with the row of the newest file; see RUN_KEYS)
#   ""     = only the newest file (decompositions: the newest run folder); "<name>" = that file (without .csv) or run folder; nothing = skip this check
# The figures are written next to the newest file that was used.
const RUNS = Dict(
    "check_ctmrg3d_init"          => "all",
    "check_ctmrg3d_exponent"      => "all",
    "check_vbpeps_diff_env"       => "all",
    "check_ctmrg3d_west_updates"  => "all",
    "check_vbpeps_decompositions" => "",
)
# The columns that identify one run of a check (for "all": the same values = the same run, the newest file wins)
const RUN_KEYS = Dict(
    "check_ctmrg3d_init"          => ["chi_env", "chi_peps", "D_CE", "beta", "projectors", "init", "seed"],
    "check_ctmrg3d_exponent"      => ["chi_env", "chi_peps", "D_CE", "beta", "P"],
    "check_vbpeps_diff_env"       => ["chi_env", "chi_peps", "D_CE", "beta", "diff_env"],
    "check_ctmrg3d_west_updates"  => ["chi_env", "chi_peps", "D_CE", "beta", "projectors", "west_updates", "case"],
    "check_vbpeps_decompositions" => ["chi_env", "chi_peps", "D_CE", "network", "method"],
)
const DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "checks"))
# The results file of the vbPEPS compare with the rows of PEPSKit and BetterPEPSKit (profiles :timing and :equal); nothing = skip this figure.
# The Hortense rows: copy the Hortense results file to the laptop under another name (it must not overwrite the laptop file).
const IMPL_RESULTS = normpath(joinpath(@__DIR__, "..", "..", "data", "profiling", "comparing", "hortense_results.csv"))

include(joinpath(@__DIR__, "..", "reference", "ising_3d_reference.jl"))   # talapov_blote_magnetization, plot_ising_references!

# A small CSV reader (no CSV.jl in the environment): a vector of rows (Dict column => string); the fields have no commas
function read_rows(path)
    lines = filter(!isempty, readlines(path))
    header = split(lines[1], ",")
    return [Dict(zip(header, split(l, ","; limit = length(header)))) for l in lines[2:end]]
end
num(r, k) = parse(Float64, r[k])

# The rows of a check (see RUNS) and the newest file used (its name gives the names of the figures); nothing if there is no data
function data_of(check)
    run = RUNS[check]
    run === nothing && return nothing
    dir = joinpath(DATA_DIR, check)
    isdir(dir) || (@warn "$check: no folder $dir (copy it from Hortense first)"; return nothing)
    folders = check == "check_vbpeps_decompositions"     # this check has one run folder per run, with results.csv in it
    files = folders ? [joinpath(d, "results.csv") for d in readdir(dir; join = true) if isfile(joinpath(d, "results.csv"))] :
                      filter(f -> endswith(f, ".csv"), readdir(dir; join = true))
    if run == "" || run == "all"
        isempty(files) && (@warn "$check: no data in $dir"; return nothing)
        files = sort(files; by = mtime)
        run == "" && (files = files[end:end])
    else
        path = folders ? joinpath(dir, run, "results.csv") : joinpath(dir, endswith(run, ".csv") ? run : run * ".csv")
        isfile(path) || (@warn "$check: no file $path"; return nothing)
        files = [path]
    end
    header = readline(files[end])
    files = filter(f -> readline(f) == header, files)
    keycols = RUN_KEYS[check]
    byrun = Dict{Any, Dict}()          # the same run in a later file replaces the earlier row
    order = []
    for f in files, r in read_rows(f)
        k = Tuple(get(r, c, "") for c in keycols)
        haskey(byrun, k) || push!(order, k)
        byrun[k] = r
    end
    length(files) > 1 && @info "$check: $(length(order)) runs from $(length(files)) files ($(join(basename.(files), ", ")))"
    return [byrun[k] for k in order], files[end]
end
setname(r) = "($(r["chi_env"]),$(r["chi_peps"]),$(r["D_CE"]))"
pngpath(csv, tag) = replace(csv, r"\.csv$" => "_$tag.png")

# check_ctmrg3d_init: one panel per (set, β), x = the start (ones, random, Nishino-Okunishi): every seed is a point, a bar from the smallest to the largest seed; :svd left, :oblique right of each start.
# Two figures: <run>_init.png (y = |m|, with the Bethe and Talapov-Blöte values) and <run>_init_iterations.png (y = the CTMRG3D iterations).
const INIT_STARTS = [("ones", "ones"), ("random", "random"), ("nishino_okunishi", "N-O")]   # N-O = Nishino-Okunishi

function init_figure(rows, yof, ylabel; references = false)
    panels = []
    for s in unique(setname.(rows)), β in sort(unique(num.(rows, "beta")))
        sel = [r for r in rows if setname(r) == s && num(r, "beta") == β]
        ys = [yof(r) for r in sel]
        p = plot(; title = "$s, β = $β", xticks = (1:3, last.(INIT_STARTS)), xlims = (0.5, 3.5), ylabel, legend = :outerright, titlefontsize = 9,
                 ylims = references ? :auto : (0, 1.1 * maximum(ys)))
        for (proj, col, dx) in (("svd", :blue, -0.12), ("oblique", :red, 0.12))
            for (k, (init, _)) in enumerate(INIT_STARTS)
                rr = [r for r in sel if r["projectors"] == proj && r["init"] == init]
                length(rr) >= 2 && plot!(p, [k + dx, k + dx], collect(extrema(yof.(rr))); color = col, linewidth = 2, alpha = 0.5, label = "")
            end
            for conv in (true, false)
                rr = [r for r in sel if r["projectors"] == proj && (r["status"] == "converged") == conv]
                isempty(rr) && continue
                x = [findfirst(st -> st[1] == r["init"], INIT_STARTS) + dx for r in rr]
                scatter!(p, x, yof.(rr); color = col, markershape = conv ? :circle : :xcross, markersize = 5, markeralpha = 0.7,
                         label = ":$proj" * (conv ? "" : " (not converged)"))
            end
        end
        if references
            hline!(p, [ClusterExpansions.bethe_ising_magnetization(β)]; color = :gray, linestyle = :dash, label = "Bethe")
            hline!(p, [talapov_blote_magnetization(β)]; color = :black, label = "MC fit (TB)")
        end
        push!(panels, p)
    end
    return plot(panels...; layout = (cld(length(panels), 3), min(3, length(panels))), size = (1500, 380 * cld(length(panels), 3)),
                left_margin = 6mm, bottom_margin = 8mm, plot_title = "bar = smallest to largest seed (random, Nishino-Okunishi: seeds $(join(unique(r["seed"] for r in rows if r["init"] == "random"), ", ")))",
                plot_titlefontsize = 10)
end

function plot_init(rows, csv)
    savefig(init_figure(rows, r -> abs(num(r, "m")), "|m|"; references = true), pngpath(csv, "init")); @info "wrote $(pngpath(csv, "init"))"
    savefig(init_figure(rows, r -> num(r, "iterations"), "CTMRG3D iterations"), pngpath(csv, "init_iterations"))
    @info "wrote $(pngpath(csv, "init_iterations"))"
    init_table(rows)
end

# The table of check_ctmrg3d_init: per (set, β, projectors) the iterations and the time of :ones, and the mean (min-max) of the seeds of the other starts
function init_table(rows)
    stat(v) = isempty(v) ? "-" : length(v) == 1 ? @sprintf("%.0f", v[1]) : @sprintf("%.1f (%.0f-%.0f)", sum(v) / length(v), minimum(v), maximum(v))
    println("\n===== check_ctmrg3d_init: CTMRG3D iterations [time of the run in s, including the contraction of m] per start; * = not all converged =====")
    println(rpad("set", 12), rpad("β", 7), rpad("projectors", 11), rpad("ones", 18), rpad("random: mean (min-max)", 34), "Nishino-Okunishi: mean (min-max)")
    for s in unique(setname.(rows)), β in sort(unique(num.(rows, "beta"))), proj in ("svd", "oblique")
        sel = [r for r in rows if setname(r) == s && num(r, "beta") == β && r["projectors"] == proj]
        isempty(sel) && continue
        col(init) = (rr = [r for r in sel if r["init"] == init];
                     isempty(rr) ? "-" : stat(num.(rr, "iterations")) * " [" * stat(num.(rr, "time_s")) * "]" * (all(r -> r["status"] == "converged", rr) ? "" : " *"))
        println(rpad(s, 12), rpad(β, 7), rpad(proj, 11), rpad(col("ones"), 18), rpad(col("random"), 34), col("nishino_okunishi"))
    end
end

# check_ctmrg3d_exponent: per set |m| against β per P (+ references) and the ϵ history per (β, P)
function plot_exponent(rows, csv)
    panels = []
    for s in unique(setname.(rows))
        sel = [r for r in rows if setname(r) == s]
        βs = sort(unique(num.(sel, "beta")))
        p1 = plot(; title = "$s: |m| against β", xlabel = "β", ylabel = "|m|", legend = :topleft, titlefontsize = 9)
        plot_ising_references!(p1, minimum(βs) - 0.01, maximum(βs) + 0.01)
        p2 = plot(; title = "$s: ϵ per iteration", xlabel = "iteration", ylabel = "ϵ", yscale = :log10, legend = :outerright, titlefontsize = 9)
        for (k, P) in enumerate(sort(unique(num.(sel, "P"))))
            rr = sort([r for r in sel if num(r, "P") == P]; by = r -> num(r, "beta"))
            conv = [r["status"] == "converged" for r in rr]
            plot!(p1, num.(rr, "beta"), abs.(num.(rr, "m")); color = k, marker = :circle, label = "P = $P")
            any(.!conv) && scatter!(p1, num.(rr[.!conv], "beta"), abs.(num.(rr[.!conv], "m")); color = k, markershape = :xcross, markersize = 7, label = "")
            for (j, r) in enumerate(rr)
                ϵ = parse.(Float64, split(r["eps_all"], ";"))
                plot!(p2, 1:length(ϵ), max.(ϵ, 1e-16); color = k, linestyle = (:solid, :dash, :dot)[mod1(j, 3)], label = "P = $P, β = $(r["beta"])")
            end
        end
        push!(panels, p1, p2)
    end
    fig = plot(panels...; layout = (length(panels) ÷ 2, 2), size = (1400, 420 * length(panels) ÷ 2), left_margin = 6mm, bottom_margin = 6mm)
    savefig(fig, pngpath(csv, "exponent")); @info "wrote $(pngpath(csv, "exponent"))"
end

# check_vbpeps_diff_env: per set f and |m| against β, without (false) and with (true) the extra AD term
function plot_diff_env(rows, csv)
    panels = []
    for s in unique(setname.(rows))
        sel = [r for r in rows if setname(r) == s]
        p1 = plot(; title = "$s: f", xlabel = "β", ylabel = "f", legend = :bottomleft, titlefontsize = 9)
        p2 = plot(; title = "$s: |m|", xlabel = "β", ylabel = "|m|", legend = :topleft, titlefontsize = 9)
        βs = sort(unique(num.(sel, "beta")))
        plot_ising_references!(p2, minimum(βs) - 0.01, maximum(βs) + 0.01)
        for (d, col, sh) in (("false", :blue, :circle), ("true", :red, :utriangle))
            rr = sort([r for r in sel if r["diff_env"] == d]; by = r -> num(r, "beta"))
            isempty(rr) && continue
            lab = d == "false" ? "fixed environment" : "with the AD term"
            plot!(p1, num.(rr, "beta"), num.(rr, "f"); color = col, marker = sh, label = lab)
            plot!(p2, num.(rr, "beta"), abs.(num.(rr, "m")); color = col, marker = sh, label = lab)
        end
        push!(panels, p1, p2)
    end
    fig = plot(panels...; layout = (length(panels) ÷ 2, 2), size = (1200, 400 * length(panels) ÷ 2), left_margin = 6mm, bottom_margin = 6mm)
    savefig(fig, pngpath(csv, "diff_env")); @info "wrote $(pngpath(csv, "diff_env"))"
end

# check_ctmrg3d_west_updates: one row of three panels per (set, projectors): |m|, the CTMRG3D time and the number of West updates (partial_ctmrgstep calls)
# against β, one line per west_updates (xcross = not converged: stopped by the iteration cap, the stagnation stop or the time limit)
function plot_west_updates(rows, csv)
    haskey(rows[1], "case") && return plot_west_updates_old(rows, csv)
    panels = []
    for s in unique(setname.(rows)), proj in unique(getindex.(rows, "projectors"))
        sel = [r for r in rows if setname(r) == s && r["projectors"] == proj]
        isempty(sel) && continue
        βs = sort(unique(num.(sel, "beta")))
        p1 = plot(; title = "$s :$proj: |m|", xlabel = "β", ylabel = "|m|", legend = :topleft, titlefontsize = 9)
        plot_ising_references!(p1, minimum(βs) - 0.01, maximum(βs) + 0.01)
        p2 = plot(; title = "$s :$proj: CTMRG3D time (s)", xlabel = "β", legend = false, titlefontsize = 9, ylims = (0, :auto))
        p3 = plot(; title = "$s :$proj: West updates (partial_ctmrgstep calls)", xlabel = "β", legend = false, titlefontsize = 9, ylims = (0, :auto))
        for (k, w) in enumerate(unique(getindex.(sel, "west_updates")))
            rr = sort([r for r in sel if r["west_updates"] == w]; by = r -> num(r, "beta"))
            nc = [r["status"] != "converged" for r in rr]
            x = num.(rr, "beta")
            for (p, y) in ((p1, abs.(num.(rr, "m"))), (p2, num.(rr, "t_ctm_s")), (p3, num.(rr, "partial_steps")))
                plot!(p, x, y; color = k, marker = :circle, label = "west_updates = $w")
                any(nc) && scatter!(p, x[nc], y[nc]; color = k, markershape = :xcross, markersize = 8, label = "")
            end
        end
        push!(panels, p1, p2, p3)
    end
    fig = plot(panels...; layout = (length(panels) ÷ 3, 3), size = (1600, 400 * length(panels) ÷ 3), left_margin = 6mm, bottom_margin = 6mm)
    savefig(fig, pngpath(csv, "west_updates")); @info "wrote $(pngpath(csv, "west_updates"))"
end

# The CSV of the old version of the check (column case, own PEPOs with D = 4 and 5)
function plot_west_updates_old(rows, csv)
    p1 = plot(; title = "|m|", xlabel = "β", legend = :topleft); p2 = plot(; title = "CTMRG time (s)", xlabel = "β", legend = false)
    p3 = plot(; title = "partial_ctmrgstep calls", xlabel = "β", legend = false)
    for c in unique(getindex.(rows, "case"))
        rr = sort([r for r in rows if r["case"] == c]; by = r -> num(r, "beta"))
        plot!(p1, num.(rr, "beta"), abs.(num.(rr, "m")); marker = :circle, label = c)
        plot!(p2, num.(rr, "beta"), num.(rr, "t_ctm_s"); marker = :circle)
        plot!(p3, num.(rr, "beta"), num.(rr, "partial_steps"); marker = :circle)
    end
    fig = plot(p1, p2, p3; layout = (1, 3), size = (1500, 420), left_margin = 6mm, bottom_margin = 6mm)
    savefig(fig, pngpath(csv, "plots")); @info "wrote $(pngpath(csv, "plots"))"
end

# check_vbpeps_decompositions (results.csv of a run folder): per network the time per CTMRG step of both methods against the sets (χenv, χpeps, D_CE),
# sorted by n = χenv · bond (the size of the decomposed matrix), and the speed-up t_dense / t_block-Krylov against n (> 1: block-Krylov is faster)
function plot_decompositions(rows, csv)
    rows = [r for r in rows if isfinite(num(r, "t_step_s")) && num(r, "t_step_s") > 0]
    dset(r) = "($(r["chi_env"]),$(r["chi_peps"]),$(r["D_CE"]))"
    panels = []
    speedups = []                       # (network, marker, [(n, t_dense / t_block-Krylov)])
    for (k, (network, marker)) in enumerate((("<psi|psi>", :circle), ("<psi|O|psi>", :square)))
        sub = [r for r in rows if r["network"] == network]
        isempty(sub) && continue
        sets = sort(unique(dset.(sub)); by = d -> (num(first(r for r in sub if dset(r) == d), "n"), d))
        xpos = Dict(d => i for (i, d) in enumerate(sets))
        p = plot(; title = "$network: time per CTMRG step", ylabel = "time per step (s)", yscale = :log10, legend = :topleft, titlefontsize = 10,
                 xticks = (1:length(sets), ["$d n=$(Int(num(first(r for r in sub if dset(r) == d), "n")))" for d in sets]), xrotation = 60,
                 xlims = (0.5, length(sets) + 0.5))
        for (method, col, dx) in (("dense", :darkorange, -0.08), ("block-Krylov", :royalblue, 0.08))
            rr = sort([r for r in sub if r["method"] == method]; by = r -> xpos[dset(r)])
            plot!(p, [xpos[dset(r)] + dx for r in rr], num.(rr, "t_step_s"); color = col, marker, linealpha = 0.4, label = method)
        end
        pairs = [(num(a, "n"), num(a, "t_step_s") / num(b, "t_step_s")) for a in sub for b in sub
                 if a["method"] == "dense" && b["method"] == "block-Krylov" && dset(a) == dset(b)]
        push!(speedups, (network, marker, pairs))
        push!(panels, p)
    end
    isempty(panels) && return @warn "check_vbpeps_decompositions: no finished set in $csv"
    # the speed-up against n: log axes with ticks at 1, 2, 5 × 10^k written as plain numbers, limits just around the data
    ns = [x for (_, _, ps) in speedups for (x, _) in ps]; rs = [y for (_, _, ps) in speedups for (_, y) in ps]
    ticks(lo, hi) = (t = [m * 10.0^e for e in floor(Int, log10(lo)):ceil(Int, log10(hi)) for m in (1, 2, 5) if lo <= m * 10.0^e <= hi];
                     (t, [x >= 1 ? string(round(Int, x)) : string(x) for x in t]))
    xl = (minimum(ns) / 1.25, maximum(ns) * 1.25); yl = (min(minimum(rs), 1.0) / 1.25, max(maximum(rs), 1.0) * 1.25)
    pr = plot(; title = "speed-up of block-Krylov: t_dense / t_block-Krylov (> 1: block-Krylov faster)", xlabel = "n = χenv · bond",
              ylabel = "t_dense / t_block-Krylov", xscale = :log10, yscale = :log10, legend = :topleft, titlefontsize = 10,
              xlims = xl, ylims = yl, xticks = ticks(xl...), yticks = ticks(yl...), minorgrid = true, minorgridalpha = 0.1)
    hline!(pr, [1.0]; color = :black, label = "")
    for (k, (network, marker, ps)) in enumerate(speedups)
        scatter!(pr, first.(ps), last.(ps); color = k, marker, label = network)
    end
    out = joinpath(dirname(csv), "decompositions_times.png")
    fig = plot(panels..., pr; layout = (1, length(panels) + 1), size = (2200, 700), left_margin = 8mm, bottom_margin = 24mm,
               plot_title = "fast vbPEPS (BetterPEPSKit): dense against block-Krylov, β = $(rows[1]["beta"]), $(rows[1]["steps"]) timed C4v CTMRG steps per set")
    savefig(fig, out); @info "wrote $out"
end

# PEPSKit (Sander's vbPEPS, slow) against BetterPEPSKit (Lander's, fast): the vbPEPS rows of the results file whose setting ends in the implementation (fixed_env_pepskit, fixed_env_betterpepskit).
# Two rows are the same observation if case, setting, β and (χenv, χpeps, D_CE) agree; per implementation we take the newest run. Panels: time per cost-function call, |m| of both, and |Δf|, |Δ|m|| between them.
function plot_vbpeps_impls(path)
    isfile(path) || return @warn "vbPEPS implementations: no file $path (copy the Hortense results file first, see the Hortense guide)"
    lines = filter(!isempty, readlines(path))
    header = split(lines[1], ",")
    rows = [Dict(zip(header, split(l, ","; limit = length(header)))) for l in lines[2:end] if l != lines[1]]
    impl_of(r) = (m = match(r"_(pepskit|betterpepskit)(_exactpepo)?$", r["setting"]); m === nothing ? nothing : m.captures[1])
    rows = [r for r in rows if r["algorithm"] == "vbPEPS" && impl_of(r) !== nothing && !isempty(r["f"]) && !isempty(r["m_re"])]
    key(r) = (r["case"], replace(r["setting"], r"_(pepskit|betterpepskit)" => ""), num(r, "beta"), r["chi_env"], r["chi_peps"], r["D_CE"])
    newest(rr) = rr[argmax([r["run"] for r in rr])]
    pairs = []
    for k in unique(key.(rows))
        a = [r for r in rows if key(r) == k && impl_of(r) == "pepskit"]
        b = [r for r in rows if key(r) == k && impl_of(r) == "betterpepskit"]
        (isempty(a) || isempty(b)) && continue
        push!(pairs, (k, newest(a), newest(b)))
    end
    isempty(pairs) && return @warn "vbPEPS implementations: no observation with rows of both PEPSKit and BetterPEPSKit in $path"
    sort!(pairs; by = p -> (parse.(Int, (p[1][4], p[1][5], p[1][6])), p[1][3]))
    labels = ["($(k[4]),$(k[5]),$(k[6])) β=$(k[3])" * (endswith(k[2], "_exactpepo") ? " ex" : "") for (k, _, _) in pairs]
    x = 1:length(pairs)
    percall(r) = num(r, "time_s") / max(num(r, "iterations"), 1)
    done(r) = r["timed_out"] != "true"
    common = (; xticks = (x, labels), xrotation = 45, xlims = (0.5, length(pairs) + 0.5), titlefontsize = 10)
    p1 = plot(; title = "time per cost-function call (s)", yscale = :log10, legend = :topleft, common...)
    p2 = plot(; title = "|m| (open marker = timed out)", legend = :topleft, common...)
    p3 = plot(; title = "difference PEPSKit - BetterPEPSKit (open = a run timed out)", yscale = :log10, legend = :topleft, common...)
    for (impl, i, col, dx) in (("PEPSKit", 2, :firebrick, -0.1), ("BetterPEPSKit", 3, :royalblue, 0.1))
        rs = [p[i] for p in pairs]
        scatter!(p1, x .+ dx, percall.(rs); color = col, label = impl)
        scatter!(p2, x .+ dx, abs.(num.(rs, "m_re")); color = [done(r) ? col : :white for r in rs], markerstrokecolor = col, label = impl)
    end
    both = [done(p[2]) && done(p[3]) for p in pairs]
    df = [max(abs(num(p[2], "f") - num(p[3], "f")), 1e-16) for p in pairs]
    dm = [max(abs(abs(num(p[2], "m_re")) - abs(num(p[3], "m_re"))), 1e-16) for p in pairs]
    scatter!(p3, x, df; marker = :square, color = [b ? :black : :white for b in both], markerstrokecolor = :black, label = "|Δf|")
    scatter!(p3, x, dm; marker = :diamond, color = [b ? :green : :white for b in both], markerstrokecolor = :green, label = "|Δ|m||")
    outdir = mkpath(joinpath(DATA_DIR, "vbpeps_implementations"))
    out = joinpath(outdir, "pepskit_vs_betterpepskit.png")
    fig = plot(p1, p2, p3; layout = (1, 3), size = (1900, 620), left_margin = 8mm, bottom_margin = 18mm,
               plot_title = "vbPEPS: PEPSKit (Sander) against BetterPEPSKit (Lander), the same observation and start state; newest run per implementation")
    savefig(fig, out); @info "wrote $out"
    println("\n===== vbPEPS: PEPSKit against BetterPEPSKit (newest run of each per observation) =====")
    println(rpad("observation", 26), rpad("case", 12), rpad("calls P / B", 14), rpad("s per call P / B (×)", 26), rpad("f P", 16), rpad("f B", 16), rpad("|Δf|", 10),
            rpad("|m| P", 12), rpad("|m| B", 12), "timed out P / B")
    for (j, (k, a, b)) in enumerate(pairs)
        println(rpad(labels[j], 26), rpad(k[1], 12), rpad("$(a["iterations"]) / $(b["iterations"])", 14),
                rpad(@sprintf("%.2f / %.2f (%.1f)", percall(a), percall(b), percall(a) / percall(b)), 26), rpad(@sprintf("%.10f", num(a, "f")), 16), rpad(@sprintf("%.10f", num(b, "f")), 16),
                rpad(@sprintf("%.1e", df[j]), 10), rpad(@sprintf("%.6f", abs(num(a, "m_re"))), 12), rpad(@sprintf("%.6f", abs(num(b, "m_re"))), 12),
                "$(a["timed_out"]) / $(b["timed_out"])")
    end
end

for (check, f) in (("check_ctmrg3d_init", plot_init), ("check_ctmrg3d_exponent", plot_exponent), ("check_vbpeps_diff_env", plot_diff_env),
                   ("check_ctmrg3d_west_updates", plot_west_updates), ("check_vbpeps_decompositions", plot_decompositions))
    d = data_of(check)
    d === nothing && continue
    try
        f(d...)
    catch err
        @error "$check: the plot failed (we continue with the next check)" exception = (err, catch_backtrace())
    end
end
if IMPL_RESULTS !== nothing
    try
        plot_vbpeps_impls(IMPL_RESULTS)
    catch err
        @error "vbPEPS implementations: the plot failed" exception = (err, catch_backtrace())
    end
end
