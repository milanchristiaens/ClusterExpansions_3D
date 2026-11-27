# Plots of the bond-dimension scan (submit_scan.sh): per method |m| against β for the top set (run ending in _big_<method>) and every smaller set
# (runs ending in _scan_<method>_<dims>), with Talapov-Blöte and BP, and the difference to Talapov-Blöte.

ENV["GKSwstype"] = "nul"

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using Plots
using LinearAlgebra

include(joinpath(@__DIR__, "..", "tools", "compare_tools.jl"))
include(joinpath(@__DIR__, "..", "..", "reference", "ising_3d_reference.jl"))

# ======================================================================================================================
# Settings
# ======================================================================================================================
const RESULTS_FILE = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "profiling", "comparing", "hortense_results.csv"))
const OUT_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "profiling", "comparing", "scan_plots"))
const METHODS = ["svd", "oblique", "octantunweighted", "octantsigmaweighted", "octantleadweighted", "vbpeps"]
const BETA_RANGE = (0.19, 0.255)
# ======================================================================================================================

magnetization(r) = hypot(r.m_re, r.m_im)
not_converged(r) = r.converged === false || r.timed_out || (r.algorithm == "vbPEPS" && !(r.check in ("ok", "")))

# the rows of every set of a method: the newest row per β of all runs with the same ending (as the big plots)
function method_sets(rows_all, method)
    sets = Dict{Tuple{Int, Int}, Vector{Any}}()
    for r in rows_all
        (endswith(r.run, "_big_$method") || occursin("_scan_$(method)_", r.run)) || continue
        (isfinite(r.m_re) && BETA_RANGE[1] <= r.β <= BETA_RANGE[2]) || continue
        push!(get!(sets, (r.chienv, r.chipeps), Any[]), r)
    end
    out = Dict{Tuple{Int, Int}, Vector{Any}}()
    for (k, rs) in sets
        newest = Dict{Float64, Any}()
        for r in rs
            b = round(r.β; digits = 8)
            (!haskey(newest, b) || r.run > newest[b].run) && (newest[b] = r)
        end
        out[k] = sort(collect(values(newest)); by = r -> r.β)
    end
    return out
end

function main()
    rows = read_ising_results(RESULTS_FILE)
    mkpath(OUT_DIR)
    βs = collect(range(BETA_RANGE..., length = 400))
    tb = talapov_blote_magnetization.(βs)
    bethe = ClusterExpansions.bethe_ising_magnetization.(βs)
    for method in METHODS
        sets = method_sets(rows, method)
        isempty(sets) && (@info "no runs for $method"; continue)
        keys_sorted = sort(collect(keys(sets)); by = k -> (k[2], k[1]))
        cols = palette(:viridis, max(length(keys_sorted), 2))
        p_m = plot(; xlabel = "β", ylabel = "|⟨σᶻ⟩|", title = "$method: magnetization", legend = false, xlims = BETA_RANGE)
        p_d = plot(; xlabel = "β", ylabel = "|m| − m (Talapov-Blöte)", title = "$method: difference to Talapov-Blöte", legend = :topright, legendfontsize = 7, xlims = BETA_RANGE)
        plot!(p_m, βs, tb; color = :black, label = "")
        plot!(p_m, βs, bethe; color = :gray, linestyle = :dash, label = "")
        hline!(p_d, [0.0]; color = :black, label = "Talapov-Blöte")
        plot!(p_d, βs, bethe .- tb; color = :gray, linestyle = :dash, label = "BP")
        for (i, k) in enumerate(keys_sorted)
            rs = sets[k]
            b = [r.β for r in rs]; m = magnetization.(rs); d = m .- talapov_blote_magnetization.(b)
            plot!(p_m, b, m; color = cols[i], marker = :circle, markersize = 3, label = "")
            plot!(p_d, b, d; color = cols[i], marker = :circle, markersize = 3, label = "(χenv, χpeps) = $k")
            bad = findall(not_converged, rs)
            isempty(bad) || scatter!(p_d, b[bad], d[bad]; color = cols[i], markershape = :xcross, markersize = 5, label = "")
        end
        vline!(p_m, [βc_ising]; color = :gray, linestyle = :dot, label = "")
        vline!(p_d, [βc_ising]; color = :gray, linestyle = :dot, label = "")
        savefig(compare_figure([p_m, p_d], "bond-dimension scan, $method, exact PEPO, warm start (cold near βc); cross = not converged"; height = 520),
                joinpath(OUT_DIR, "scan_$method.png"))
        @info "Plot saved: $(joinpath(OUT_DIR, "scan_$method.png"))"
    end
    return nothing
end

main()
