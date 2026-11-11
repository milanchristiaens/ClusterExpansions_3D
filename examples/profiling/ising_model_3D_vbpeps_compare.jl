# Comparison of 4 vbPEPS cases for the 3D Ising model (CE + variational boundary PEPS):
#
#     warm start (false / true)  x  differentiate_environment (false / true)
#
# Everything uses the same CE PEPOs (built once, same for all cases) and the same random seed per observation (Random.seed!(SEED_BASE + i) before observation i),
# so the cases differ only in the two settings above. For every case and every β we store: final cost f, number of cost-function calls, time, magnetization m.
# Each observation is appended to results.csv (and closed) immediately, so a crash halfway keeps all earlier data.
# Everything goes to its own subdirectory  data/profiling/vbpeps_compare_<date>_<time>/  :
#     settings.txt, results.csv, timing_<case>.csv (TimerOutputs rows of each case),
#     compare_vbpeps.png (|m|, f and time against β), compare_vbpeps_spread.png (|m| of the cases around their mean), compare_vbpeps_timing_phases.png and compare_vbpeps_timing_self.png (timings per section, 4 bars per section)
#
# Order: false/cold, false/warm, [compile of the `true` pullback, this was ~1h40 in an earlier run], true/cold, true/warm.
# Compilation happens once per process (warm-up with tiny settings, not measured).
#
# Time limit: every case gets RUN_BUDGET_S seconds in total and every observation at most MAX_SECONDS_OBS seconds (see the settings).
# An observation that runs out of time stops after its current LBFGS iteration and is marked timed_out = true (a cross in the plot):
# its f and m are then the values of the last iterate, not of a converged optimization.
#
# Note on m: at g = 0 the exact <σᶻ> is 0 below βc (Z2 symmetry) and has an arbitrary sign above βc, so we plot |m|. A nonzero m below βc is a contraction/optimization artefact.

ENV["GKSwstype"] = "nul"     # no plot window: the plots are only saved to a file

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using TimerOutputs
using Plots
using Random
using LinearAlgebra
BLAS.set_num_threads(1)

const to = ClusterExpansions.to # the timer that the package code writes to

# ======================================================================================================================
# Settings
# ======================================================================================================================
const T = Float64
const spin_symmetry = Trivial
const J = 1.0
const g = 0.0
const D_CE = 2
const chipeps = 2
const chienv = 8
const Δβ_iter = 0.005
# The β values of the observations (increasing; every value must be a multiple of Δβ_iter).
const BETAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

const SEED_BASE = 1000              # observation i uses Random.seed!(SEED_BASE + i), the same in all 4 cases

# Time limits (seconds). Total of the run is about: compile of `true` (~6000 s) + compile of `false` (a few minutes) + 4 * RUN_BUDGET_S (+ at most one LBFGS iteration overshoot per observation).
const RUN_BUDGET_S = 2400.0         # per case: 40 minutes
const MAX_SECONDS_OBS = 420.0       # per observation (the per-observation limit is lowered further when the case budget is running out)
const MIN_SECONDS_OBS = 30.0        # but every observation gets at least this

# Settings of the vbPEPS (the same for all cases, except warm_start and differentiate_environment which are the 4 cases)
const base_kwargs = (;
    boundary_alg_type = SimultaneousCTMRG, symmetrization = nothing,
    boundary_tol = 1.0e-10, boundary_maxiter = 1500, boundary_verbosity = 1,
    gradient_tol = 1.0e-6, gradient_maxiter = 5, gradient_krylovdim = 20,
    optimizer_memory = 64, optimizer_tol = 1.0e-5, optimizer_maxiter = 10,
    maxiter_init = 20,
    optimizer_verbosity = 2,
)

# (name, warm_start, differentiate_environment); the order matters: the `true` cases come last, because their compilation takes very long
const CASES = [
    ("false_cold", false, false),
    ("false_warm", true, false),
    ("true_cold", false, true),
    ("true_warm", true, true),
]



# If REPLOT_DIR is the NAME of a folder in data/profiling with a results.csv (for example "vbpeps_compare_20261111_1248"), we only redo all the plots from it
# and nothing else is run (no calculation, no compilation of the pullback). Leave it "" for a normal run.
const REPLOT_DIR = "vbpeps_compare_20261111_1248"

# Settings of the spread plot (|m| of every case minus the mean |m| of the cases, per β)
const SPREAD_EXCLUDE_CASES = String[]       # cases that we leave out completely, for example ["true_warm"]
const SPREAD_EXCLUDE_FLAGGED = true         # leave out observations that failed or whose check is not "ok" (unphysical f)
const SPREAD_EXCLUDE_TIMED_OUT = true      # leave out observations that were stopped by the time limit
const SPREAD_OUTLIER_TOL = 0.1             # leave out a point whose |m| differs more than this from the median |m| of the cases at the same β

# Settings of the timing plots (4 bars per section, one per case)
const TIMING_TOP = 4                     # number of sections in the self-time plot
const CASE_COLORS = Dict("false_cold" => :steelblue, "false_warm" => :lightskyblue, "true_cold" => :firebrick, "true_warm" => :lightsalmon)   # blue = fixed env, red = diff env, dark = cold, light = warm

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

# This function performs a small check to see wether or not the converged f's are physical
function check_cost(β, f)
    isfinite(f) || return "no result"
    msgs = String[]
    f > -3β + 1.0e-6 && push!(msgs, "f above -3β: worse than the all-up product state")
    f < -(log(2) + 3β) - 1.0e-6 && push!(msgs, "f below the exact lower bound: unphysical")
    if β < βc_ising
        fref = -(log(2) + 3 * log(cosh(β)) + 3 * tanh(β)^4)
        f < fref - 0.01 && push!(msgs, "f more than 0.01 below the high-temperature series: unphysical")
    end
    return isempty(msgs) ? "ok" : "CHECK: " * join(msgs, " / ")
end

# timer_rows and write_timer_csv (read the TimerOutput back and write the CSV)
include(joinpath(@__DIR__, "timing_tools.jl"))

# read_timer_csv, shorten, total_time and self_time_by_name (plot helpers of the timing scripts)
include(joinpath(@__DIR__, "timing_plots.jl"))

# ======================================================================================================================
# CE: the PEPOs for all β, built once (every case uses exactly these)
# ======================================================================================================================
function build_pepos()
    @assert issorted(BETAS) "BETAS must be increasing"
    trunc_alg = NoEnvTruncation(truncrank(D_CE))
    ce_alg = ising_operators(J, g, 0.0; T = BigFloat, symmetry = nothing)
    ΔO = ClusterExpansions.evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64)
    O = ClusterExpansions.evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)
    Os = Any[]
    β_now = 0.0
    for β in BETAS
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
# One case
# ======================================================================================================================
function run_case(name, warm, diff, Os, pepsspace, envspace, obs, resultspath, outdir)
    # name = e.g. "true_warm"

    @info "===== Case $name: warm_start = $warm, differentiate_environment = $diff, budget = $(RUN_BUDGET_S) s ====="
    reset_timer!(to)
    ClusterExpansions.reset_max_alloc!()
    kwargs = (; base_kwargs..., differentiate_environment = diff)
    state = nothing           # holds (psi, env2, env3) for the warm start
    t_case = time()

    # Loop over all observations (at different β, using different O's)
    for i in eachindex(Os)
        β = BETAS[i]
        # Use same random seed in the different cases
        Random.seed!(SEED_BASE + i)
        # time we allow this observation: at most MAX_SECONDS_OBS, and lower if the case budget runs out
        # RUN_BUDGET_S - (time() - t_case) = the budget left
        # length(Os) - i + 1 = observations left (including this one)
        # We allow twice the fair share (*2)
        # We keep this budget between 30 s and 420 s.
        allowed = clamp(2 * (RUN_BUDGET_S - (time() - t_case)) / (length(Os) - i + 1), MIN_SECONDS_OBS, MAX_SECONDS_OBS)
        
        # Start of observation
        t_obs = time()
        try
            res = ClusterExpansions.@timeit to "expectation_value (vbPEPS and observable)" begin
                # The actual optimisation (may be with warm start)
                result = ClusterExpansions.bpeps_optimize(Os[i], pepsspace, envspace; state0 = warm ? state : nothing, max_seconds = allowed, kwargs...)
                # Measures ⟨σᶻ⟩ on the converged state
                # Returns a named tuple as res
                (; result, m = ClusterExpansions.bpeps_observables(Os[i], [obs], result)[1])
            end
            # Total time of this observation (optimisation + observation)
            dt = time() - t_obs
            # See if f is realistic
            check = check_cost(β, res.result.cost)
            
            # Print/save result
            println("OBS $name β = $(round(β; digits = 3)): f = $(res.result.cost), calls = $(res.result.numfg), time = $(round(dt; digits = 1)) s, timed out = $(res.result.timed_out), m = $(res.m), ", check)
            append_row(resultspath, name, warm, diff, β, res.result.cost, res.result.numfg, dt, res.result.timed_out, ComplexF64(res.m), check)
            
            # If warm, then update state to the current psi and env's (even if not enough time)
            warm && (state = (res.result.psi, res.result.env2, res.result.env3))
        # Runs if anything inside the try block threw an error
        catch err
            # Time untill failure
            dt = time() - t_obs
            # Print/Save the result
            @error "OBS $name β = $(round(β; digits = 3)) failed" exception = (err, catch_backtrace())
            append_row(resultspath, name, warm, diff, β, NaN, 0, dt, false, complex(NaN, NaN), "ERROR: " * first(sprint(showerror, err), 150))
            # If warm, don't update state (reuse latest succesfull one)
        end
        # Force the log output to be writte now
        flush(stdout); flush(stderr)
    end

    @info "Case $name finished: $(round(time() - t_case; digits = 1)) s, peak memory of the process so far $(round(Sys.maxrss() / 2^30; digits = 2)) GiB"
    try
        write_timer_csv(joinpath(outdir, "timing_$(name).csv"), timer_rows(to))
    catch err
        @error "Writing the timer csv of $name failed" exception = (err, catch_backtrace())
    end

    # Force garbage collection between cases (releases memory)
    GC.gc()
    return nothing
end

# Warm-up: compile everything with tiny settings (not measured, nothing is saved)
function warmup(diff, Os, pepsspace, envspace, obs)
    @info "Warm-up (compilation) for differentiate_environment = $diff: not measured"
    t0 = time()
    tiny = (; base_kwargs..., differentiate_environment = diff, optimizer_maxiter = 1, maxiter_init = 2, boundary_maxiter = 100)
    Random.seed!(SEED_BASE)
    r1 = ClusterExpansions.bpeps_optimize(Os[1], pepsspace, envspace; tiny...)
    # second call with a warm start, so that the warm-start path is compiled as well
    r2 = ClusterExpansions.bpeps_optimize(Os[2], pepsspace, envspace; state0 = (r1.psi, r1.env2, r1.env3), tiny...)
    ClusterExpansions.bpeps_observables(Os[2], [obs], r2)
    @info "Warm-up for differentiate_environment = $diff took $(round(time() - t0; digits = 1)) s"
    return nothing
end

# ======================================================================================================================
# Summary table and plots (from results.csv)
# ======================================================================================================================
function print_summary(rows)
    println("\n===== Summary per case =====")
    println(rpad("case", 12), rpad("n obs", 7), rpad("total time (s)", 16), rpad("calls", 8), rpad("timed out", 11), "failed")
    for (name, _, _) in CASES
        sub = filter(r -> r.case == name, rows)
        isempty(sub) && continue
        println(rpad(name, 12), rpad(length(sub), 7), rpad(round(sum(r.time_s for r in sub); digits = 1), 16), rpad(sum(r.ncalls for r in sub), 8),
            rpad(count(r -> r.timed_out, sub), 11), count(r -> !isfinite(r.f), sub))
    end
    println("\n===== m (real part) per β =====")
    βs = sort(unique(r.β for r in rows))
    println(rpad("β", 8), join([rpad(name, 14) for (name, _, _) in CASES]))
    for β in βs
        line = rpad(round(β; digits = 3), 8)
        for (name, _, _) in CASES
            r = filter(r -> r.case == name && r.β == β, rows)
            line *= rpad(isempty(r) ? "-" : string(round(r[1].m_re; sigdigits = 4)), 14)
        end
        println(line)
    end
    return nothing
end

function make_plots(rows, outdir)
    colour(diff) = diff ? :crimson : :royalblue      # colour: differentiate_environment
    style(warm) = warm ? :dash : :solid              # line style: warm start (dashed) or cold start (solid)
    p_m = plot(; xlabel = "β", ylabel = "|⟨σᶻ⟩|", title = "magnetization (g = 0: 0 below βc)", legend = :topleft)
    p_f = plot(; xlabel = "β", ylabel = "f = -log(λ)", title = "cost (bound: f ≤ -3β)", legend = false)
    p_t = plot(; xlabel = "β", ylabel = "time per observation (s)", title = "time", yscale = :log10, legend = false)
    for (name, warm, diff) in CASES
        sub = sort(filter(r -> r.case == name, rows); by = r -> r.β)
        isempty(sub) && continue
        βs = [r.β for r in sub]
        label = "$(diff ? "diff env" : "fixed env"), $(warm ? "warm" : "cold")"
        for (p, ys, lab) in ((p_m, [hypot(r.m_re, r.m_im) for r in sub], label), (p_f, [r.f for r in sub], ""), (p_t, [max(r.time_s, 1.0e-1) for r in sub], ""))
            plot!(p, βs, ys; color = colour(diff), linestyle = style(warm), marker = :circle, markersize = 4, label = lab)
            idx = findall(r -> r.timed_out, sub)
            isempty(idx) || scatter!(p, βs[idx], ys[idx]; color = colour(diff), markershape = :xcross, markersize = 9, markerstrokewidth = 2, label = "")
        end
    end
    βline = sort(unique(r.β for r in rows))
    plot!(p_f, βline, -3 .* βline; color = :black, linestyle = :dot, label = "")
    vline!(p_m, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    vline!(p_f, [βc_ising]; color = :gray, linestyle = :dot, label = "")
    plt = plot(p_m, p_f, p_t; layout = (1, 3), size = (1600, 480), margin = 6Plots.mm, bottom_margin = 8Plots.mm, left_margin = 8Plots.mm,
        plot_title = "vbPEPS: χpeps = $chipeps, χenv = $chienv, D_CE = $D_CE, g = $g (cross = timed out; dotted gray = βc)")
    savefig(plt, joinpath(outdir, "compare_vbpeps.png"))
    return nothing
end

# Median of a vector
function median_of(v)
    sv = sort(v)
    n = length(sv)
    return isodd(n) ? sv[(n + 1) ÷ 2] : (sv[n ÷ 2] + sv[n ÷ 2 + 1]) / 2
end

# The observations that we use in the spread plot: leaves out the cases, flagged / failed / timed out observations and outliers (see the settings at the top).
# Prints what is left out. The magnetization is |m| because the sign is arbitrary (Z2).
function spread_points(rows)
    reason(r) = if r.case in SPREAD_EXCLUDE_CASES
        "case left out"
    elseif !(isfinite(r.m_re) && isfinite(r.f))
        "failed"
    elseif SPREAD_EXCLUDE_FLAGGED && r.check != "ok"
        "flagged (unphysical f)"
    elseif SPREAD_EXCLUDE_TIMED_OUT && r.timed_out
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
            if abs(hypot(r.m_re, r.m_im) - med) > SPREAD_OUTLIER_TOL
                println("  β = $β, $(r.case): outlier, |m| = $(round(hypot(r.m_re, r.m_im); sigdigits = 4)) against median $(round(med; sigdigits = 4))")
            else
                push!(kept, r)
            end
        end
    end
    return kept
end

# Upper panel: mean |m| of the cases that we use, with the range (min to max) as a band. Lower panel: |m| of every case minus this mean, so the y-axis can be zoomed in.
# β with fewer than 2 remaining cases are not shown. The mean is over the cases that are left per β, so it can change from β to β when a case is left out.
function make_spread_plot(rows, outdir)
    kept = spread_points(rows)
    absm(r) = hypot(r.m_re, r.m_im)
    βs = Float64[]; means = Float64[]; los = Float64[]; his = Float64[]
    for β in sort(unique(r.β for r in kept))
        v = [absm(r) for r in kept if r.β == β]
        length(v) < 2 && continue
        push!(βs, β); push!(means, sum(v) / length(v)); push!(los, minimum(v)); push!(his, maximum(v))
    end
    p_mean = plot(βs, means; ribbon = (means .- los, his .- means), fillalpha = 0.25, color = :black, marker = :circle, markersize = 4,
        xlabel = "β", ylabel = "mean |⟨σᶻ⟩|", title = "mean of the cases used (band = min to max)", legend = false)
    vline!(p_mean, [βc_ising]; color = :gray, linestyle = :dot)
    
    # |m| of every case minus the mean, per β
    series = NamedTuple[]
    for (name, warm, diff) in CASES
        pts = sort(filter(r -> r.case == name && r.β in βs, kept); by = r -> r.β)
        isempty(pts) && continue
        push!(series, (; name, warm, diff, βs = [r.β for r in pts], devs = [absm(r) - means[findfirst(==(r.β), βs)] for r in pts]))
    end
    # We set the limits of the axes ourselves from the data (otherwise hline!/vline! on an empty plot fix the axes to 0..2)
    ymax = max(1.15 * maximum([maximum(abs, s.devs) for s in series]; init = 0.0), 1.0e-6)
    p_dev = plot(; xlabel = "β", ylabel = "|m| of the case - mean", title = "difference to the mean (y-axis zoomed to ±$(round(ymax; sigdigits = 2)))", legend = :outertopright,
        ylims = (-ymax, ymax), xlims = (minimum(βs) - 0.02, maximum(βs) + 0.02))
    for s in series
        plot!(p_dev, s.βs, s.devs; color = s.diff ? :crimson : :royalblue, linestyle = s.warm ? :dash : :solid, marker = :circle, markersize = 4,
            label = "$(s.diff ? "diff env" : "fixed env"), $(s.warm ? "warm" : "cold")")
    end
    hline!(p_dev, [0.0]; color = :black, label = "")
    vline!(p_dev, [βc_ising]; color = :gray, linestyle = :dot, label = "")

    plt = plot(p_mean, p_dev; layout = (2, 1), size = (950, 750), left_margin = 8Plots.mm, bottom_margin = 5Plots.mm)
    savefig(plt, joinpath(outdir, "compare_vbpeps_spread.png"))
    return nothing
end

# Timing plots: the same plots as timing_CE_phases.png and timing_CE_self.png of ising_model_3D_timing.jl, but with 4 bars (one per case) per section.
# They read timing_<case>.csv of the run folder (written after every case). The times are summed over all observations of a case.
# The percentage in brackets is relative to the total time of the observations of that case, followed by the number of calls.

# timing_<case>.csv of the cases that have one: (name, label, color, rows)
function read_case_timings(outdir)
    cases = NamedTuple[]
    for (name, warm, diff) in CASES
        path = joinpath(outdir, "timing_$(name).csv")
        if isfile(path)
            push!(cases, (; name, label = "$(diff ? "diff env" : "fixed env"), $(warm ? "warm" : "cold")", color = CASE_COLORS[name], rows = read_timer_csv(path)))
        else
            @warn "No $path: case $name is left out of the timing plots"
        end
    end
    isempty(cases) && error("No timing_<case>.csv in $outdir")
    return cases
end

# Horizontal bars like hbars (timing_plots.jl), but with one bar per case in every group.
# values[c][i], labels[c][i] = bar of case c in group i; the first group is at the top.
function hbars_grouped(names, values, labels, cases; title = "", xlabel = "", figsize = (1500, 900))
    nc = length(cases)
    n = length(names)
    xmax = max(maximum(maximum(v) for v in values), 1.0e-9)
    maxlen = maximum(length, names)
    bw = 0.8 / nc                                   # height of one bar
    y0(i, c) = i - 0.4 + (c - 1) * bw               # lower edge of the bar of case c in group i
    anns = [(values[c][i] + 0.01 * xmax, y0(i, c) + bw / 2, Plots.text(labels[c][i], 7, :left)) for c in 1:nc for i in 1:n]
    plt = plot(;
        title, xlabel, legend = :bottomright, size = figsize,
        yticks = (1:n, names), ylims = (0.4, n + 0.6), yflip = true,
        xlims = (0, 1.5 * xmax), grid = :x,
        left_margin = (1.8 * maxlen + 4) * mm, bottom_margin = 8mm, ytickfontsize = 9,
        annotations = anns,
    )
    for c in 1:nc, i in 1:n
        lo = y0(i, c)
        plot!(plt, Plots.Shape([0, values[c][i], values[c][i], 0], [lo, lo, lo + bw, lo + bw]); fillcolor = cases[c].color, linecolor = cases[c].color, label = i == 1 ? cases[c].label : "")
    end
    return plt
end

# "12.3 s (5.1%)  40x", nothing for a section that a case does not have
bar_label(t, grand, n) = n == 0 ? "" : "$(round(t; sigdigits = 3)) s ($(round(100 * t / grand; digits = 1))%)  $(n)x"

# Main phases of the run (sections at depth 1, so the direct parts of "expectation_value ..."), by total time. Last group: time that is in no depth-1 section.
function make_timing_phases_plot(rows, outdir)
    cases = read_case_timings(outdir)
    rest_name = "(rest: code outside the sections above)"
    grands = [total_time(c.rows) for c in cases]
    # per case: section name => (total time, number of calls)
    secs = [Dict{String, Tuple{Float64, Int}}() for _ in cases]
    for (k, c) in enumerate(cases), r in c.rows
        r.depth == 1 || continue
        t, n = get(secs[k], r.name, (0.0, 0))
        secs[k][r.name] = (t + r.total, n + r.ncalls)
    end
    rests = [grands[k] - sum((t for (t, _) in values(secs[k])); init = 0.0) for k in eachindex(cases)]
    names = unique(vcat([collect(keys(d)) for d in secs]...))
    sort!(names; by = nm -> -sum(get(d, nm, (0.0, 0))[1] for d in secs))     # largest total time (summed over the cases) first
    any(rests[k] > 0.01 * grands[k] for k in eachindex(cases)) && push!(names, rest_name)
    values_ = [[nm == rest_name ? rests[k] : get(secs[k], nm, (0.0, 0))[1] for nm in names] for k in eachindex(cases)]
    calls = [[nm == rest_name ? 1 : get(secs[k], nm, (0.0, 0))[2] for nm in names] for k in eachindex(cases)]
    labels = [[bar_label(values_[k][i], grands[k], calls[k][i]) for i in eachindex(names)] for k in eachindex(cases)]
    plt = hbars_grouped(shorten.(names), values_, labels, cases; title = "vbPEPS, 4 cases: main phases of the observations",
        xlabel = "total time [s]  (% of case total, calls)",        figsize = (1500, 220 + 110 * length(names)))
    savefig(plt, joinpath(outdir, "compare_vbpeps_timing_phases.png"))
    return nothing
end

# The TIMING_TOP sections with the largest self time (summed over the cases); same-named sections are merged like in plot_self_time.
function make_timing_self_plot(rows, outdir)
    cases = read_case_timings(outdir)
    grands = [total_time(c.rows) for c in cases]
    flats = [Dict(r.name => (r.ncalls, r.self) for r in self_time_by_name(c.rows)) for c in cases]      # name => (calls, self time)
    names = unique(vcat([collect(keys(f)) for f in flats]...))
    sort!(names; by = nm -> -sum(get(f, nm, (0, 0.0))[2] for f in flats))
    names = names[1:min(TIMING_TOP, length(names))]
    values_ = [[get(f, nm, (0, 0.0))[2] for nm in names] for f in flats]
    calls = [[get(f, nm, (0, 0.0))[1] for nm in names] for f in flats]
    labels = [[bar_label(values_[k][i], grands[k], calls[k][i]) for i in eachindex(names)] for k in eachindex(cases)]
    plt = hbars_grouped(shorten.(names), values_, labels, cases; title = "vbPEPS, 4 cases: top $(length(names)) sections by self time",
        xlabel = "self time [s]  (% of case total, calls)",        figsize = (1500, 220 + 110 * length(names)))
    savefig(plt, joinpath(outdir, "compare_vbpeps_timing_self.png"))
    return nothing
end

# All plots; each one in its own try, so that a failing plot does not stop the others
function make_all_plots(rows, outdir)
    for (plotfun, file) in ((make_plots, "compare_vbpeps.png"), (make_spread_plot, "compare_vbpeps_spread.png"), (make_timing_phases_plot, "compare_vbpeps_timing_phases.png"), (make_timing_self_plot, "compare_vbpeps_timing_self.png"))
        try
            plotfun(rows, outdir)
            @info "Plot saved: $(joinpath(outdir, file))"
        catch err
            @error "Plot $file failed, the data is in results.csv (set REPLOT_DIR to retry)" exception = (err, catch_backtrace())
        end
    end
    return nothing
end

# ======================================================================================================================
# Main
# ======================================================================================================================
function main()
    # Replot mode if run was already done, then it just reprints/replots the stored results
    if !isempty(REPLOT_DIR)
        plotdir = normpath(joinpath(@__DIR__, "..", "..", "data", "profiling", REPLOT_DIR))
        rows = read_results(joinpath(plotdir, "results.csv"))
        print_summary(rows)
        make_all_plots(rows, plotdir)
        return nothing
    end

    # Create the output folder and csv path using current date/time
    t_run = time()
    outdir = normpath(joinpath(@__DIR__, "..", "..", "data", "profiling", "vbpeps_compare_" * Libc.strftime("%Y%m%d_%H%M", time())))
    mkpath(outdir)
    resultspath = joinpath(outdir, "results.csv")

    # Create results.csv, with only the header
    open(io -> println(io, CSV_HEADER), resultspath, "w")

    # Write Julia version, threads and all settings to the file 
    open(joinpath(outdir, "settings.txt"), "w") do io
        println(io, "Julia $(VERSION), threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())")
        println(io, "J = $J, g = $g, D_CE = $D_CE, chipeps = $chipeps, chienv = $chienv, Δβ_iter = $Δβ_iter, BETAS = $BETAS")
        println(io, "SEED_BASE = $SEED_BASE, RUN_BUDGET_S = $RUN_BUDGET_S, MAX_SECONDS_OBS = $MAX_SECONDS_OBS, MIN_SECONDS_OBS = $MIN_SECONDS_OBS")
        println(io, "base_kwargs = $base_kwargs")
        println(io, "cases = $CASES")
    end
    @info "Output directory: $outdir"
    @info "Threads.nthreads() = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())"

    # Define the virtual spaces and the observables
    pepsspace = ℂ^chipeps
    envspace = ℂ^chienv
    obs = SpinOperators.σᶻ(T, spin_symmetry)

    # Build all PEPOs + print time
    t0 = time()
    Os = build_pepos()
    @info "CE PEPOs built in $(round(time() - t0; digits = 1)) s"

    # compile and run, `false` first and `true` last
    # Compiled remembers wether warm-up has been done (for both false and true env)
    compiled = Dict(false => false, true => false)
    for (name, warm, diff) in CASES
        if !compiled[diff]
            try
                warmup(diff, Os, pepsspace, envspace, obs)
            catch err
                @error "Warm-up for differentiate_environment = $diff failed (we continue)" exception = (err, catch_backtrace())
            end
            # Update compiled to show that the diff= has been compiled
            compiled[diff] = true
        end
        try
            run_case(name, warm, diff, Os, pepsspace, envspace, obs, resultspath, outdir)
        catch err
            @error "Case $name failed" exception = (err, catch_backtrace())
        end
    end

    rows = read_results(resultspath)
    print_summary(rows)
    make_all_plots(rows, outdir)
    @info "Total time of the run: $(round((time() - t_run) / 3600; digits = 2)) h"
    return nothing
end

main()
