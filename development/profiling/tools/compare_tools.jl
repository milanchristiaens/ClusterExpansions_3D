# compare_tools.jl -- helpers for the compare scripts (comparing/) and for the vbPEPS profiling script.
# Used by: comparing/ising_model_3D_CE_vbPEPS_compare.jl and ising_model_3D_CE_vbPEPS_profiling.jl (check_cost).
# Contents:
#   βc_ising, check_cost            sanity check of the cost f of a vbPEPS optimization
#   build_pepos                     the CE PEPOs at all β of a list, built once
#   CSV_HEADER, append_row, read_results   results.csv of a compare run (its columns are those of the 4 vbPEPS cases: warm_start, differentiate_environment)
#   median_of, spread_points, plot_spread   difference of |m| of every case to the mean |m| of the cases
# The functions get their settings as arguments; they do not read the settings of the script that includes this file.
# Needs (the including script has them): using TensorKit, TensorKitTensors, MPSKit, PEPSKit, ClusterExpansions, Plots
# The timing plots of the compare scripts are in timing_plots.jl.

# ======================================================================================================================
# Sanity check of the final cost f = −ln λ = −(1/N) ln Z:
# Where: 
# - f = β F / N = free energy per site multiplied by β (F = E − T·S -> f = β e − s) (Low T, E wins: f → −3β) (high T, S wins, f → −ln 2)
# - Z = Σ_σ exp(β Σ_bonds σᵢσⱼ), 
# - λ = λ3/λ2 =  <psi| O |psi> / <psi|psi>, λ3 and λ2 are PER-site values, so also λ = Z^{1/N} = Λ^{1/(Lx Ly)}.)
# - Λ is the largest eigenvalue of T (transfer operator in z plane): Z = Tr(T^Lz) ≈ Λ^Lz
# - N = Lx·Ly·Lz sites and 3N bounds
# - H = −Σ_bonds σᵢσⱼ
# classical 3D Ising (J = 1, g = 0)
#
# Rigorous bounds for the exact f:  -(log 2 + 3β) <= f <= -3β  (all-up product state gives -3β, so an optimized f above -3β means the optimization failed).
# lower bound (saturated af β=0): Every bond factor is at most e^β (if both bonds alligned), 2^N configurations in total, 3N bonds
#             Therefore Z<=2^N e^{3Nβ}, with corresponding f ≥ −(ln 2 + 3β)
# Upper bound (saturated af β=+∞): all Boltzmann weigths are positive, so Z is at least the weigth of any single configurations
#              Z is summ over all configurations, an easy configuration is one where all spins are UP, then we get
#              contribution of e^{3Nβ}, therefore Z ≥ e^{3Nβ}, this corresponds f ≤ −3β
# 
# High temperature series: f < -(ln 2 + 3 ln cosh β + 3 (tanh β)⁴)
#       This can be derived as follows:  e^{βσσ} = cosh β (1 + tanh β · σσ)
#       Z = Σ_σ exp(β Σ_bonds σᵢσⱼ) = 2^N (cosh β)^{3N} Σ_{closed loops} (tanh β)^{loop length}. Only loops survive: then the σ's pair up into σ² = 1 
#       ln Z / N = ln 2 + 3 ln cosh β + 3 t⁴ + (higher orders in t), with t = tanh β.
# Below βc the high-temperature series is a reference; the margin 0.01 is our choice.
# ======================================================================================================================

const βc_ising = 0.2216546   # not sure

# This function performs a small check to see wether or not the converged f's are physical. Returns "ok" or "CHECK: ...".
# reference = true: also writes the high-temperature series value into the text (below βc).
function check_cost(β, f; reference = false)
    isfinite(f) || return "no result"
    msgs = String[]
    ref = ""
    f > -3β + 1.0e-6 && push!(msgs, "f above -3β: worse than the all-up product state")
    f < -(log(2) + 3β) - 1.0e-6 && push!(msgs, "f below the exact lower bound: unphysical")
    if β < βc_ising
        fref = -(log(2) + 3 * log(cosh(β)) + 3 * tanh(β)^4)
        reference && (ref = " (series $(round(fref; digits = 5)))")
        f < fref - 0.01 && push!(msgs, "f more than 0.01 below the high-temperature series: unphysical")
    end
    return (isempty(msgs) ? "ok" : "CHECK: " * join(msgs, " / ")) * ref
end

# ======================================================================================================================
# CE: the PEPOs for all β, built once
# ======================================================================================================================
# CE: the PEPOs at all β of the list β_obs (increasing, multiples of Δβ_iter), built once; a compare script uses exactly these for all its cases.
function build_pepos(β_obs, Δβ_iter, D, J, g)
    @assert issorted(β_obs) "β_obs must be increasing"
    trunc_alg = NoEnvTruncation(truncrank(D))
    ce_alg = ising_operators(J, g, 0.0; T = BigFloat, symmetry = nothing)
    ΔO = ClusterExpansions.evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64)
    O = ClusterExpansions.evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)
    Os = Any[]
    β_now = 0.0
    for β in β_obs
        # number of time steps from the previous β to this one
        nsteps = round(Int, (β - β_now) / Δβ_iter)
        @assert abs(nsteps - (β - β_now) / Δβ_iter) < 1.0e-8 "β = $β is not a multiple of Δβ_iter = $Δβ_iter"
        for _ in 1:nsteps
            O, = approximate_state((O, ΔO), trunc_alg; rot_invariance = false)
        end
        β_now = β
        push!(Os, deepcopy(O))   # copy, so that later steps cannot change an earlier O
    end
    return Os
end

# ======================================================================================================================
# results.csv
# ======================================================================================================================
const CSV_HEADER = "case,warm_start,differentiate_environment,beta,f,cost_function_calls,time_s,timed_out,m_re,m_im,check"

function append_row(path, case, warm, diff, β, f, ncalls, time_s, timed_out, m, check)
    open(path, "a") do io    # the file is closed after every row, so the data is on disk
        println(io, join((case, warm, diff, β, f, ncalls, round(time_s; digits = 2), timed_out, real(m), imag(m), replace(check, ',' => ' ', '\n' => ' ', '\r' => ' ')), ","))
    end
    return nothing
end

function read_results(path)
    rows = NamedTuple[]
    for line in Iterators.drop(eachline(path), 1)
        isempty(strip(line)) && continue
        p = split(line, ',')
        push!(rows, (;
            case = String(p[1]), warm = p[2] == "true", diff = p[3] == "true", β = parse(Float64, p[4]), f = parse(Float64, p[5]),
            ncalls = parse(Int, p[6]), time_s = parse(Float64, p[7]), timed_out = p[8] == "true",
            m_re = parse(Float64, p[9]), m_im = parse(Float64, p[10]), check = String(p[11]),
        ))
    end
    return rows
end

# ======================================================================================================================
# Difference to the mean of the cases
# ======================================================================================================================
# Median of a vector
function median_of(v)
    sv = sort(v)
    n = length(sv)
    return isodd(n) ? sv[(n + 1) ÷ 2] : (sv[n ÷ 2] + sv[n ÷ 2 + 1]) / 2
end

# The observations that we use in the spread plot: leaves out the cases, flagged / failed / timed out observations and outliers (see the keyword arguments).
# Prints what is left out. The magnetization is |m| because the sign is arbitrary (Z2).
function spread_points(rows; exclude_cases = String[], exclude_flagged = true, exclude_timed_out = true, outlier_tol = 0.1)
    reason(r) = if r.case in exclude_cases
        "case left out"
    elseif !(isfinite(r.m_re) && isfinite(r.f))
        "failed"
    elseif exclude_flagged && r.check != "ok"
        "flagged (unphysical f)"
    elseif exclude_timed_out && r.timed_out
        "timed out"
    else
        ""
    end
    kept = NamedTuple[]
    println("\n===== Spread plot: observations left out =====")
    for β in sort(unique(r.β for r in rows))
        at = NamedTuple[]
        for r in filter(r -> r.β == β, rows)
            why = reason(r)
            isempty(why) ? push!(at, r) : println("  β = $β, $(r.case): $why")
        end
        isempty(at) && continue
        med = median_of([hypot(r.m_re, r.m_im) for r in at])
        for r in at
            if abs(hypot(r.m_re, r.m_im) - med) > outlier_tol
                println("  β = $β, $(r.case): outlier, |m| = $(round(hypot(r.m_re, r.m_im); sigdigits = 4)) against median $(round(med; sigdigits = 4))")
            else
                push!(kept, r)
            end
        end
    end
    return kept
end

# One panel: |m| of every case minus the mean |m| of the cases that we use (absolute difference, for all β, so it also works where |m| is almost 0). Returns the plot (the script puts it in a figure).
# case_specs = [(name = "false_cold", label = "fixed env, cold", color = :royalblue, linestyle = :solid), ...]; spread_kwargs go to spread_points.
# β with fewer than 2 remaining cases are not shown. The mean is over the cases that are left per β, so it can change from β to β when a case is left out.
function plot_spread(rows, case_specs; spread_kwargs...)
    kept = spread_points(rows; spread_kwargs...)
    absm(r) = hypot(r.m_re, r.m_im)
    βs = Float64[]; means = Float64[]
    for β in sort(unique(r.β for r in kept))
        v = [absm(r) for r in kept if r.β == β]
        length(v) < 2 && continue
        push!(βs, β); push!(means, sum(v) / length(v))
    end
    isempty(βs) && return plot(; title = "no β with at least 2 cases: no difference shown", xlabel = "β", legend = false)

    # |m| of every case minus the mean, per β
    series = NamedTuple[]
    for c in case_specs
        pts = sort(filter(r -> r.case == c.name && r.β in βs, kept); by = r -> r.β)
        isempty(pts) && continue
        push!(series, (; name = c.name, label = c.label, color = c.color, linestyle = c.linestyle, βs = [r.β for r in pts], devs = [absm(r) - means[findfirst(==(r.β), βs)] for r in pts]))
    end
    # We set the limits of the axes ourselves from the data (otherwise hline!/vline! on an empty plot fix the axes to 0..2)
    ymax = max(1.15 * maximum([maximum(abs, s.devs) for s in series]; init = 0.0), 1.0e-6)
    βall = [r.β for r in rows]
    p = plot(; xlabel = "β", ylabel = "|m| of the case - mean", title = "difference to the mean (y-axis zoomed to ±$(round(ymax; sigdigits = 2)))", legend = false,
        ylims = (-ymax, ymax), xlims = (minimum(βall) - 0.02, maximum(βall) + 0.02))
    for s in series
        plot!(p, s.βs, s.devs; color = s.color, linestyle = s.linestyle, marker = :circle, markersize = 4, label = s.label)
    end
    hline!(p, [0.0]; color = :black, label = "")
    vline!(p, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    return p
end