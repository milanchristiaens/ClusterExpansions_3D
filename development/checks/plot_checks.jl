# plot_checks.jl -- figures of the CSV files of the checks (23 Nov), next to each CSV file:
#   check_ctmrg3d_init       <run>_init.png       |m| per start (:ones, :random seeds, :nishino_okunishi seeds) and projector kind, one panel per (set, β)
#   check_ctmrg3d_exponent   <run>_exponent.png   |m| against β per P (with the Talapov-Blöte and Bethe references) and the ϵ history per (β, P)
#   check_vbpeps_diff_env    <run>_diff_env.png   f and |m| against β, with (true) and without (false) the extra AD term, per set
#   check_ctmrg3d_west_updates  west_updates_comparison.png   |m|, CTMRG time and partial_ctmrgstep calls against β per case
# The figure of check_vbpeps_decompositions comes from that script itself (REPLOT_DIR, see its header).
# The checks themselves run on Hortense or the laptop; copy their data/checks/<check>/ folders to the laptop first (see the results report).
# Run in the ClusterExpansions environment (Plots from the default environment):
#   PowerShell, clone folder:  julia +1.12 --project=. development/checks/plot_checks.jl

using ClusterExpansions, Plots, Printf
using Plots.PlotMeasures

# Settings: the CSV file per check, by its name in data/checks/<check>/; "" = the newest .csv file there; nothing = skip this check
const RUNS = Dict(
    "check_ctmrg3d_init"         => "",
    "check_ctmrg3d_exponent"     => "",
    "check_vbpeps_diff_env"      => "",
    "check_ctmrg3d_west_updates" => "",
)
const DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "checks"))

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
    if isempty(run)
        files = filter(f -> endswith(f, ".csv"), readdir(dir; join = true))
        isempty(files) && (@warn "$check: no .csv file in $dir"; return nothing)
        return files[argmax(mtime.(files))]
    end
    path = joinpath(dir, endswith(run, ".csv") ? run : run * ".csv")
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
    fig = plot(panels...; layout = (length(panels) ÷ 3 + (length(panels) % 3 > 0), min(3, length(panels))), size = (1500, 380 * cld(length(panels), 3)),
               left_margin = 6mm, bottom_margin = 10mm)
    savefig(fig, pngpath(csv, "init")); @info "wrote $(pngpath(csv, "init"))"
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

# check_ctmrg3d_west_updates: |m|, CTMRG time and partial_ctmrgstep calls against β, one line per case
function plot_west_updates(csv)
    rows = read_rows(csv)
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

for (check, f) in (("check_ctmrg3d_init", plot_init), ("check_ctmrg3d_exponent", plot_exponent),
                   ("check_vbpeps_diff_env", plot_diff_env), ("check_ctmrg3d_west_updates", plot_west_updates))
    csv = csv_of(check)
    csv === nothing && continue
    try
        f(csv)
    catch err
        @error "$check: the plot failed (we continue with the next check)" exception = (err, catch_backtrace())
    end
end
