# Plots of the BP comparison (models_3D_CE_bp_compare.jl): per model the order parameter (|⟨σᶻ⟩| or ρs = |⟨b⟩|²) and the second observable (⟨σˣ⟩ or ρ = ⟨n⟩)
# against β, one curve per case (the newest row per case and β; CTMRG3D, vbPEPS), BP dashed, and the QMC points of the transverse-field Ising model (largest L) if there are any.

ENV["GKSwstype"] = "nul"

using Plots
using Printf

# ======================================================================================================================
# Settings
# ======================================================================================================================
const RESULTS_FILE = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "profiling", "comparing", "models_3D_CE_results.csv"))
const QMC_FILE = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "reference", "tfim3d_qmc.csv"))
const OUT_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "profiling", "comparing", "models_bp_plots"))
const COLORS = Dict("bp" => :gray, "svd_P05" => :royalblue, "svd_P0" => :crimson, "svdclass_P0" => :darkorange,
                    "octantunweighted" => :olive, "octantsigmaweighted" => :teal, "octantleadweighted" => :purple,
                    "vbpeps_warm" => :saddlebrown, "vbpeps_cold" => :sandybrown)
# ======================================================================================================================

function read_csv(path)
    lines = readlines(path)
    names = Symbol.(split(lines[1], ','))
    rows = [Dict(zip(names, split(l, ','; limit = length(names)))) for l in lines[2:end] if !isempty(strip(l)) && l != lines[1]]
    return rows
end
num(s) = isempty(s) ? NaN : parse(Float64, s)

# the newest row per (case, β)
function newest(rows)
    best = Dict{Tuple{String, Float64}, Any}()
    for r in rows
        k = (r[:case], num(r[:beta]))
        (!haskey(best, k) || r[:run] > best[k][:run]) && (best[k] = r)
    end
    return collect(values(best))
end

function qmc_points(g)
    isfile(QMC_FILE) || return nothing
    rows = filter(r -> isapprox(num(r[:g]), g), read_csv(QMC_FILE))
    isempty(rows) && return nothing
    Lmax = maximum(parse(Int, r[:L]) for r in rows)
    rows = sort(filter(r -> parse(Int, r[:L]) == Lmax, rows); by = r -> num(r[:beta]))
    return (; L = Lmax, β = num.(getindex.(rows, :beta)), m = num.(getindex.(rows, :m_abs)), dm = num.(getindex.(rows, :dm_abs)),
            sx = num.(getindex.(rows, :sx)))
end

function main()
    rows = read_csv(RESULTS_FILE)
    mkpath(OUT_DIR)
    for model in sort(unique(r[:model] for r in rows))
        rs = newest(filter(r -> r[:model] == model && isfinite(num(r[:obs1])), rows))
        kind = rs[1][:obs1_name]           # sz (Ising), Sz (hard-core bosons as spins), n (Bose-Hubbard)
        ising = kind == "sz"
        p1 = plot(; xlabel = "β", ylabel = ising ? "|⟨σᶻ⟩|" : "ρs = |⟨b⟩|²", title = model, legend = :topleft)
        p2 = plot(; xlabel = "β", ylabel = ising ? "⟨σˣ⟩" : kind == "Sz" ? "|Re⟨b⟩|" : "ρ = ⟨n⟩", title = model, legend = false)
        for case in sort(unique(r[:case] for r in rs))
            c = sort(filter(r -> r[:case] == case, rs); by = r -> num(r[:beta]))
            β = num.(getindex.(c, :beta))
            x1, x2 = num.(getindex.(c, :obs1)), num.(getindex.(c, :obs2))
            o1 = ising ? abs.(x1) : kind == "Sz" ? x1 .^ 2 .+ x2 .^ 2 : abs2.(x2)
            o2 = ising ? x2 : kind == "Sz" ? abs.(x1) : x1
            style = case == "bp" ? (; linestyle = :dash, marker = :none) : (; marker = :circle, markersize = 3)
            plot!(p1, β, o1; color = get(COLORS, case, :black), label = case, style...)
            plot!(p2, β, o2; color = get(COLORS, case, :black), label = case, style...)
            bad = findall(r -> r[:check] != "ok" && !startswith(r[:check], "ok"), c)
            isempty(bad) || scatter!(p1, β[bad], o1[bad]; color = get(COLORS, case, :black), markershape = :xcross, markersize = 5, label = "")
        end
        if ising
            q = qmc_points(parse(Float64, replace(model, "tfim_g" => "")))
            if q !== nothing
                scatter!(p1, q.β, q.m; yerror = q.dm, color = :black, marker = :diamond, markersize = 3, label = "QMC L = $(q.L)")
                scatter!(p2, q.β, q.sx; color = :black, marker = :diamond, markersize = 3, label = "")
            end
        end
        plt = plot(p1, p2; layout = (1, 2), size = (1400, 520), margin = 5Plots.mm)
        savefig(plt, joinpath(OUT_DIR, "$model.png"))
        @info "Plot saved: $(joinpath(OUT_DIR, "$model.png"))"
    end
    return nothing
end

main()
