# plot_checks.jl -- figures of the checks (23 Nov), next to the data of each check:
#   check_ctmrg3d_init          <run>_init.png             |m| per start (:ones, :random seeds, :nishino_okunishi seeds) and projector kind, one panel per (set, β)
#                               <run>_init_iterations.png  the CTMRG3D iterations of the same runs (which start converges fastest), + a table on screen
#   check_ctmrg3d_exponent      <run>_exponent.png         |m| against β per P (with the Talapov-Blöte and Bethe references) and the ϵ history per (β, P)
#   check_vbpeps_diff_env       <run>_diff_env.png         f and |m| against β, with (true) and without (false) the extra AD term, per set
#   check_ctmrg3d_west_updates  <run>_west_updates.png     per (set, projectors): |m|, CTMRG3D time and West updates against β, one line per west_updates
#   check_vbpeps_decompositions <run folder>/decompositions_times.png   time per CTMRG step, dense against block-Krylov, per set (sorted by n), + speed-up
#   vbPEPS implementations      data/checks/vbpeps_implementations/pepskit_vs_betterpepskit.png   PEPSKit (Sander's vbPEPS) against BetterPEPSKit
#                               (Lander's): time per cost-function call, |m| and the differences |Δf|, |Δm| of the same observation (rows of the results file of
#                               the vbPEPS compare, profiles :timing and :equal), + a table on screen
# The figures of the contraction benchmark: development/contraction_plans/plot_benchmarks.jl.
# The checks themselves run on Hortense or the laptop; copy their data to the laptop first (Hortense guide, section 21).
# Run in the ClusterExpansions environment (Plots from the default environment):
#   PowerShell, clone folder:  julia +1.12 --project=. development/checks/plot_checks.jl

using ClusterExpansions, Plots, Printf
using Plots.PlotMeasures

# Settings: the data per check, by its name in data/checks/<check>/ (a CSV file; for check_vbpeps_decompositions a run folder); "" = the newest one there;
# nothing = skip this check
const RUNS = Dict(
    "check_ctmrg3d_init"          => "",
    "check_ctmrg3d_exponent"      => "",
    "check_vbpeps_diff_env"       => "",
    "check_ctmrg3d_west_updates"  => "",
    "check_vbpeps_decompositions" => "",
)
const DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "checks"))
# The results file of the vbPEPS compare with the rows of PEPSKit and BetterPEPSKit (profiles :timing and :equal); nothing = skip this figure.
# The Hortense rows: copy the Hortense results file to the laptop under another name (it must not overwrite the laptop file), see the Hortense guide.
const IMPL_RESULTS = normpath(joinpath(@__DIR__, "..", "..", "data", "profiling", "comparing", "hortense_results.csv"))

include(joinpath(@__DIR__, "..", "reference", "ising_3d_reference.jl"))   # talapov_blote_magnetization, plot_ising_references!

# A small CSV reader (no CSV.jl in the environment): a vector of rows (Dict column => string); the fields have no commas
function read_rows(path)
    lines = filter(!isempty, readlines(path))
    header = split(lines[1], ",")
    return [Dict(zip(header, split(l, ","; limit = length(header)))) for l in lines[2:end]]
end
num(r, k) = parse(Float64, r[k])

function csv_of(check)
    run = RUNS[check]
    run === nothing && return nothing
    dir = joinpath(DATA_DIR, check)
    isdir(dir) || (@warn "$check: no folder $dir (copy it from Hortense first)"; return nothing)
    folders = check == "check_vbpeps_decompositions"     # this check has one run folder per run, with results.csv in it
    if isempty(run)
        files = folders ? [joinpath(d, "results.csv") for d in readdir(dir; join = true) if isfile(joinpath(d, "results.csv"))] :
                          filter(f -> endswith(f, ".csv"), readdir(dir; join = true))
        isempty(files) && (@warn "$check: no data in $dir"; return nothing)
        return files[argmax(mtime.(files))]
    end
    path = folders ? joinpath(dir, run, "results.csv") : joinpath(dir, endswith(run, ".csv") ? run : run * ".csv")
    isfile(path) || (@warn "$check: no file $path"; return nothing)
    return path
end
setname(r) = "($(r["chi_env"]),$(r["chi_peps"]),$(r["D_CE"]))"
pngpath(csv, tag) = replace(csv, r"\.csv$" => "_$tag.png")

# check_ctmrg3d_init: one panel per (set, β); x = the start, y = |m|; :svd and :oblique side by side; open marker = not converged
function plot_init(csv)
    rows = read_rows(csv)
    startlabel(r) = r["init"] == "ones" ? "ones" : (r["init"] == "random" ? "rand $(r["seed"])" : "NO $(r["seed"])")
    starts = unique(startlabel.(rows))
    panels = []
    for s in unique(setname.(rows)), β in sort(unique(num.(rows, "beta")))
        sel = [r for r in rows if setname(r) == s && num(r, "beta") == β]
        p = plot(; title = "$s, β = $β", xticks = (1:length(starts), starts), xrotation = 45, ylabel = "|m|", legend = :outerright, titlefontsize = 9)
        for (proj, col, dx) in (("svd", :blue, -0.12), ("oblique", :red, 0.12))
            for conv in (true, false)
                rr = [r for r in sel if r["projectors"] == proj && (r["status"] == "converged") == conv]
                isempty(rr) && continue
                scatter!(p, [findfirst(==(startlabel(r)), starts) + dx for r in rr], [abs(num(r, "m")) for r in rr]; color = col,
                         markershape = conv ? :circle : :xcross, markersize = 5, label = ":$proj" * (conv ? "" : " (not converged)"))
            end
        end
        hline!(p, [ClusterExpansions.bethe_ising_magnetization(β)]; color = :gray, linestyle = :dash, label = "Bethe")
        hline!(p, [talapov_blote_magnetization(β)]; color = :black, label = "MC fit (TB)")
        push!(panels, p)
    end
    fig = plot(panels...; layout = (cld(length(panels), 3), min(3, length(panels))), size = (1500, 380 * cld(length(panels), 3)),
               left_margin = 6mm, bottom_margin = 10mm)
    savefig(fig, pngpath(csv, "init")); @info "wrote $(pngpath(csv, "init"))"
    plot_init_iterations(rows, csv, starts, startlabel)
end

# check_ctmrg3d_init, the iterations: the same panels with y = the CTMRG3D iterations (the contraction of m is the same for every start, so the
# iterations decide which start is fastest); xcross = not converged within the iteration cap (or timed out). Then a table: per (set, β, projectors)
# the iterations and the time of :ones, and the mean (min-max) of the :random and :nishino_okunishi seeds.
function plot_init_iterations(rows, csv, starts, startlabel)
    panels = []
    for s in unique(setname.(rows)), β in sort(unique(num.(rows, "beta")))
        sel = [r for r in rows if setname(r) == s && num(r, "beta") == β]
        p = plot(; title = "$s, β = $β", xticks = (1:length(starts), starts), xrotation = 45, ylabel = "CTMRG3D iterations", legend = :outerright,
                 titlefontsize = 9, ylims = (0, 1.1 * maximum(num.(sel, "iterations"))))
        for (proj, col, dx) in (("svd", :blue, -0.12), ("oblique", :red, 0.12))
            for conv in (true, false)
                rr = [r for r in sel if r["projectors"] == proj && (r["status"] == "converged") == conv]
                isempty(rr) && continue
                scatter!(p, [findfirst(==(startlabel(r)), starts) + dx for r in rr], [num(r, "iterations") for r in rr]; color = col,
                         markershape = conv ? :circle : :xcross, markersize = 5, label = ":$proj" * (conv ? "" : " (not converged)"))
            end
        end
        push!(panels, p)
    end
    fig = plot(panels...; layout = (cld(length(panels), 3), min(3, length(panels))), size = (1500, 380 * cld(length(panels), 3)),
               left_margin = 6mm, bottom_margin = 10mm)
    savefig(fig, pngpath(csv, "init_iterations")); @info "wrote $(pngpath(csv, "init_iterations"))"

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
function plot_exponent(csv)
    rows = read_rows(csv)
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
function plot_diff_env(csv)
    rows = read_rows(csv)
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
function plot_west_updates(csv)
    rows = read_rows(csv)
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

# The CSV of the old version of the check (before 23 Nov: column case, own PEPOs with D = 4 and 5)
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
function plot_decompositions(csv)
    rows = [r for r in read_rows(csv) if isfinite(num(r, "t_step_s")) && num(r, "t_step_s") > 0]
    dset(r) = "($(r["chi_env"]),$(r["chi_peps"]),$(r["D_CE"]))"
    panels = []
    pr = plot(; title = "speed-up of block-Krylov: t_dense / t_block-Krylov", xlabel = "n = χenv · bond", ylabel = "t_dense / t_block-Krylov",
              xscale = :log10, yscale = :log10, legend = :topleft, titlefontsize = 10)
    hline!(pr, [1.0]; color = :black, label = "")
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
        scatter!(pr, first.(pairs), last.(pairs); color = k, marker, label = network)
        push!(panels, p)
    end
    isempty(panels) && return @warn "check_vbpeps_decompositions: no finished set in $csv"
    out = joinpath(dirname(csv), "decompositions_times.png")
    fig = plot(panels..., pr; layout = (1, length(panels) + 1), size = (2200, 700), left_margin = 8mm, bottom_margin = 24mm,
               plot_title = "fast vbPEPS (BetterPEPSKit): dense against block-Krylov, β = $(rows[1]["beta"]), $(rows[1]["steps"]) timed C4v CTMRG steps per set")
    savefig(fig, out); @info "wrote $out"
end

# PEPSKit (Sander's vbPEPS, slow) against BetterPEPSKit (Lander's, fast): the vbPEPS rows of the results file whose setting ends in the implementation
# (fixed_env_pepskit, fixed_env_betterpepskit, optionally followed by _exactpepo). Two rows are the same observation if case, setting (without the
# implementation), β and (χenv, χpeps, D_CE) agree; per implementation we take the newest run. Panels: time per cost-function call, |m| of both, and
# |Δf|, |Δ|m|| between them (equal values within the LBFGS tolerance are expected only if both converged: an open marker = at least one timed out).
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
    csv = csv_of(check)
    csv === nothing && continue
    try
        f(csv)
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
