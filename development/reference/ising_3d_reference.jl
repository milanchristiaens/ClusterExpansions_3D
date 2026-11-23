# Reference values for the magnetization of the classical 3D Ising model (simple cubic, J = 1, no field), to plot next to CTMRG3D / vbPEPS / BP.
# talapov_blote_magnetization(β) is the fit of Talapov & Blöte, J. Phys. A 29, 5727 (1996), extrapolated to L = ∞ (stated for β < 0.2995, above that an extrapolation).
using DelimitedFiles

const ISING3D_BETAC = 0.2216544
const TB_RANGE_BETA_MAX = ISING3D_BETAC / (1 - 0.26)

function talapov_blote_magnetization(β)
    β <= ISING3D_BETAC && return 0.0
    t = 1 - ISING3D_BETAC / β
    return t^0.32694109 * (1.6919045 - 0.34357731 * t^0.50842026 - 0.42572366 * t)
end

# columns: L, beta, abs_m, abs_m_err, rms_m, energy, energy_err, sweeps, seconds
function read_mc_csv(path)
    data, header = readdlm(path, ',', header = true)
    cols = Dict(String(h) => data[:, i] for (i, h) in enumerate(vec(header)))
    return (; L = Int.(cols["L"]), beta = Float64.(cols["beta"]), abs_m = Float64.(cols["abs_m"]), abs_m_err = Float64.(cols["abs_m_err"]),
            rms_m = Float64.(cols["rms_m"]), energy = Float64.(cols["energy"]), energy_err = Float64.(cols["energy_err"]))
end

const ISING3D_MC_CSV = joinpath(@__DIR__, "..", "..", "data", "reference", "ising3d_mc.csv")

# Reference lines in a plot of |m| against β: Talapov-Blöte (dashed beyond its fit range), Bethe lattice z = 6 and the MC points of ising_3d_mc.jl (largest L)
function plot_ising_references!(p, βmin, βmax; mc_csv = ISING3D_MC_CSV)
    βs = range(βmin, βmax; length = 400)
    b1 = [β for β in βs if β <= TB_RANGE_BETA_MAX]; b2 = [β for β in βs if β >= TB_RANGE_BETA_MAX]
    plot!(p, b1, talapov_blote_magnetization.(b1); color = :black, linewidth = 1.5, label = "MC fit L = ∞ (Talapov-Blöte)")
    isempty(b2) || plot!(p, b2, talapov_blote_magnetization.(b2); color = :black, linewidth = 1.5, linestyle = :dash, label = "")
    plot!(p, βs, ClusterExpansions.bethe_ising_magnetization.(βs); color = :gray, linewidth = 1.5, linestyle = :dash, label = "Bethe / BP (exact network)")
    if isfile(mc_csv)
        mc = read_mc_csv(mc_csv)
        L = maximum(mc.L)
        idx = [i for i in eachindex(mc.L) if mc.L[i] == L && βmin <= mc.beta[i] <= βmax]
        isempty(idx) || scatter!(p, mc.beta[idx], mc.abs_m[idx]; yerror = mc.abs_m_err[idx], color = :black, markershape = :utriangle, markersize = 4,
                                 label = "Wolff MC ⟨|m|⟩, L = $L")
    end
    return p
end
