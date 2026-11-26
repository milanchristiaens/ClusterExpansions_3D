# Plots of the big compare runs (3D Ising, g = 0, exact PEPO) from the Hortense results file: |m| against β of several runs together (left) and the difference to the MC fit L = ∞ (right).
# We build the figures up step by step (MC line alone, then one more curve per figure), so that every run is seen against the reference.

ENV["GKSwstype"] = "nul"     # no plot window: the plots are only saved to a file

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using Plots
using LinearAlgebra

include(joinpath(@__DIR__, "..", "tools", "compare_tools.jl"))                          # read_ising_results, compare_figure, βc_ising
include(joinpath(@__DIR__, "..", "..", "reference", "ising_3d_reference.jl"))           # talapov_blote_magnetization

# ======================================================================================================================
# Settings
# ======================================================================================================================
const RESULTS_FILE = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "profiling", "comparing", "hortense_results.csv"))
const OUT_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "profiling", "comparing", "big_compare_plots"))

# The curves: a run is found by the end of its run name (the newest run in the file with that ending); a run without rows is left out
const VBPEPS_SUFFIX = "_small_bpk"
const SERIES = [
    (; key = :svd,       suffix = "_big_svd",              label = "CTMRG3D :svd",                color = :royalblue),
    (; key = :oblique,   suffix = "_big_oblique",          label = "CTMRG3D :oblique",            color = :crimson),
    (; key = :edgeplain, suffix = "_big_octantedgeplain",  label = "octant, P_f :edge",           color = :olive),
    (; key = :edge,      suffix = "_big_octantedge",       label = "octant, P_f :edge_env",       color = :teal),
    (; key = :lead,      suffix = "_big_octantedgelead",   label = "octant, P_f :edge_lead",      color = :purple),
    (; key = :vbpeps,    suffix = VBPEPS_SUFFIX,           label = "vbPEPS",      color = :darkorange),
]

# The results file does not say whether LBFGS of a vbPEPS observation converged: here we list, per run (end of the run name), the β where the job log says
# that it stopped at the iteration cap or without progress (‖∇f‖ above the tolerance)
const LOG_NOT_CONVERGED = Dict("_small_bpk" => [0.21, 0.215, 0.22])

# The figures: file name, the curves in it, and whether the Bethe / BP line is in it
const FIGURES = [
    (; file = "1_mc_fit",                   series = Symbol[],                              bethe = false),
    (; file = "2_svd",                      series = [:svd],                                bethe = false),
    (; file = "3_svd_bethe",                series = [:svd],                                bethe = true),
    (; file = "4_oblique",                  series = [:svd, :oblique],                      bethe = true),
    (; file = "5_octant_edgeplain",         series = [:svd, :oblique, :edgeplain],          bethe = true),
    (; file = "6_octant_edge",              series = [:svd, :oblique, :edgeplain, :edge],   bethe = true),
    (; file = "7_octant_edgelead",          series = [:svd, :oblique, :edgeplain, :edge, :lead], bethe = true),
    (; file = "8_edgelead_vbpeps",          series = [:svd, :oblique, :lead, :vbpeps],      bethe = true),
]

# The same axes in every figure
const BETA_RANGE = (0.19, 0.255)
const M_YLIMS = (-0.02, 0.85)
const DIFF_YLIMS = (-0.6, 0.7)

# Markers: a small dot per observation, a small cross on top of it when the observation did not converge, timed out or has an unphysical f
const DOT_SIZE = 3
const CROSS_SIZE = 5
const CROSS_STROKE = 1

# The legend sits in the difference panel, in its empty lower right; the colours are the same in the magnetization panel
const LEGEND_POSITION = :bottomright

# ======================================================================================================================
# Data
# ======================================================================================================================
latest_run(rows, suffix) = (runs = unique(r.run for r in rows if endswith(r.run, suffix)); isempty(runs) ? nothing : maximum(runs))

magnetization(r) = hypot(r.m_re, r.m_im)
flagged(r) = r.algorithm == "vbPEPS" && !(r.check in ("ok", ""))     # unphysical f (check_cost)
log_not_converged(r) = any(endswith(r.run, k) && any(b -> isapprox(b, r.β), v) for (k, v) in LOG_NOT_CONVERGED)
not_converged(r) = r.converged === false || r.timed_out || flagged(r) || log_not_converged(r)      # CTMRG3D: ϵ not below the tolerance; vbPEPS: timed out, unphysical f or LBFGS not converged

# The rows of every curve inside BETA_RANGE, sorted by β, with the dimensions (and the CE PEPO, if it is not the exact one) in the label
function load_series(rows_all)
    data = Dict{Symbol, Any}()
    for s in SERIES
        run = latest_run(rows_all, s.suffix)
        if run === nothing
            @info "No run ending in $(s.suffix) in $RESULTS_FILE: $(s.label) is left out"
            continue
        end
        rows = sort(filter(r -> r.run == run && isfinite(r.m_re) && BETA_RANGE[1] <= r.β <= BETA_RANGE[2], rows_all); by = r -> r.β)
        isempty(rows) && continue
        exact = endswith(rows[1].setting, "_exactpepo")
        label = "$(s.label), (χenv, χpeps) = ($(rows[1].chienv), $(rows[1].chipeps))" * (exact ? "" : ", CE PEPO D_CE = $(rows[1].D_CE)")
        data[s.key] = merge(s, (; run = run, rows = rows, label = label, exact = exact))
        @info "$(s.label): run $run, $(length(rows)) β, $(count(not_converged, rows)) not converged, timed out or unphysical f"
    end
    return data
end

# ======================================================================================================================
# Figures
# ======================================================================================================================
function make_figure(spec, data, βs)
    p_m = plot(; xlabel = "β", ylabel = "|⟨σᶻ⟩|", title = "magnetization (g = 0: 0 below βc)", legend = false, xlims = BETA_RANGE, ylims = M_YLIMS)
    p_d = plot(; xlabel = "β", ylabel = "|m| − m (Talapov-Blöte)", title = "difference to the Talapov-Blöte reference", legend = LEGEND_POSITION, legendfontsize = 8, xlims = BETA_RANGE, ylims = DIFF_YLIMS)

    mc = talapov_blote_magnetization.(βs)
    plot!(p_m, βs, mc; color = :black, linewidth = 1.5, label = "")
    hline!(p_d, [0.0]; color = :black, linewidth = 1.5, label = "Talapov-Blöte")

    if spec.bethe
        bethe = ClusterExpansions.bethe_ising_magnetization.(βs)
        plot!(p_m, βs, bethe; color = :gray, linewidth = 1.5, linestyle = :dash, label = "")
        plot!(p_d, βs, bethe .- mc; color = :gray, linewidth = 1.5, linestyle = :dash, label = "BP")
    end

    n_curves = 0
    for key in spec.series
        haskey(data, key) || continue
        s = data[key]
        n_curves += 1
        b = [r.β for r in s.rows]
        m = magnetization.(s.rows)
        d = m .- talapov_blote_magnetization.(b)
        plot!(p_m, b, m; color = s.color, marker = :circle, markersize = DOT_SIZE, label = "")
        plot!(p_d, b, d; color = s.color, marker = :circle, markersize = DOT_SIZE, label = s.label)
        idx = findall(not_converged, s.rows)
        if !isempty(idx)
            scatter!(p_m, b[idx], m[idx]; color = s.color, markershape = :xcross, markersize = CROSS_SIZE, markerstrokewidth = CROSS_STROKE, label = "")
            scatter!(p_d, b[idx], d[idx]; color = s.color, markershape = :xcross, markersize = CROSS_SIZE, markerstrokewidth = CROSS_STROKE, label = "")
        end
    end
    n_curves > 0 && scatter!(p_d, Float64[], Float64[]; color = :gray, markershape = :xcross, markersize = CROSS_SIZE, markerstrokewidth = CROSS_STROKE, label = "not converged, timed out or unphysical f")

    vline!(p_m, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    vline!(p_d, [βc_ising]; color = :gray, linestyle = :dot, label = "")

    all_exact = all(data[k].exact for k in spec.series if haskey(data, k))
    title = "3D Ising model, g = 0, " * (all_exact ? "exact PEPO (D_CE = 2), " : "PEPO kind per curve in the legend, ") * "cold start (dotted gray = βc)"
    return compare_figure([p_m, p_d], title; height = 520)
end

function main()
    isfile(RESULTS_FILE) || error("$RESULTS_FILE does not exist")
    data = load_series(read_ising_results(RESULTS_FILE))
    mkpath(OUT_DIR)
    βs = collect(range(BETA_RANGE[1], BETA_RANGE[2]; length = 400))
    for spec in FIGURES
        missing_keys = [k for k in spec.series if !haskey(data, k)]
        isempty(missing_keys) || @info "Figure $(spec.file): no data for $(join(missing_keys, ", ")), left out of this figure"
        savefig(make_figure(spec, data, βs), joinpath(OUT_DIR, spec.file * ".png"))
        @info "Plot saved: $(joinpath(OUT_DIR, spec.file * ".png"))"
    end
    return nothing
end

main()