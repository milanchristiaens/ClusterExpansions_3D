# Check of the CTMRG3D option `west_updates` (how many West updates, partial_ctmrgstep, per side and per CTMRG3D iteration): :auto, 1, 2 and 4, on the 3D Ising PEPOs of the PEPO file; prints numbers, tests nothing.
# For every set in DIMS, every β, every projector kind and every west_updates it runs MPSKit.expectation_value from the same cold start and stores m, the iterations, the West updates, the last ϵ and the times in data/checks/check_ctmrg3d_west_updates/<date>_<time>.csv.

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using LinearAlgebra
using Random
using JLD2
using TimerOutputs
const CE = ClusterExpansions

# load_pepos (the PEPO file of stage 1)
include(joinpath(@__DIR__, "..", "..", "initial_pepos", "pepo_files.jl"))

# ======================================================================================================================
# Settings
# ======================================================================================================================
const DIMS = [(8, 4, 3), (11, 5, 3)]     # (χenv, χpeps, D_CE), the sets of check_ctmrg3d_init.jl
const BETAS = [0.19, 0.21, 0.225, 0.25]  # below, near and above βc ≈ 0.2217 (must be in the PEPO file)
const PROJECTORS = [:svd, :oblique]
const WEST_UPDATES = [:auto, 1, 2, 4]
const BLAS_THREADS = 4                   # CTMRG3D gets its parallelism from BLAS (as the CTMRG3D compare)

# CTMRG3D settings: those of the CTMRG3D compare (ising_model_3D_CE_CTMRG_compare.jl)
const ctm_miniter = 3
const ctm_maxiter = 80
const ctm_tol = 1e-7
const INIT = :random                     # the cold start of the compare
const INIT_SEED = 1
const STAGNATION_WINDOW = 20             # the stagnation stop of the compare, only for :oblique (0 = off)
const STAGNATION_FACTOR = 0.5
const MAX_SECONDS = 1200.0               # time limit of the CTMRG3D iterations of one run (s)
const RAM_GIB = 3                        # RAM budget of the sliced contraction of m (contract_onesite_auto)
const MACHINE = get(ENV, "CE_CHECK_MACHINE", "laptop")   # contract_onesite_auto uses the plan times of this machine
# false: contract_onesite_auto slices the current tree itself (plans = nothing); the Hortense job sets CE_CHECK_PLANS=false
const USE_PLANS = get(ENV, "CE_CHECK_PLANS", "true") == "true"
const ONESITE_KWARGS = USE_PLANS ? (; machine = MACHINE) : (; machine = MACHINE, plans = nothing)

const J = 1.0
const g = 0.0
const Δβ_iter = 0.0025
const T = Float64
const spin_symmetry = Trivial
const to = CE.to                         # the timer that the package code writes to

# ======================================================================================================================
# One run
# ======================================================================================================================
# Total number of calls and total time (s) of the timer entry `label`, wherever it is nested in the TimerOutput tree
function timer_total(t::TimerOutput, label)
    n, s = 0, 0.0
    for (k, v) in t.inner_timers
        if k == label
            n += TimerOutputs.ncalls(v); s += TimerOutputs.time(v) / 1e9
        end
        n2, s2 = timer_total(v, label)
        n += n2; s += s2
    end
    return n, s
end

function run_one(PEPO, obs, dims, projectors, west_updates)
    chienv, chipeps, _ = dims
    reset_timer!(to)
    t0 = time()
    expvals, _, ϵs, timed_out = MPSKit.expectation_value(PEPO, [obs], ℂ^chipeps, ℂ^chienv; miniter = ctm_miniter, maxiter = ctm_maxiter, tol = ctm_tol,
                                                          verbosity = 0, west_updates, projectors, init = INIT, init_seed = INIT_SEED, max_seconds = MAX_SECONDS,
                                                          stagnation_window = projectors === :oblique ? STAGNATION_WINDOW : 0, stagnation_factor = STAGNATION_FACTOR,
                                                          return_env = true, ram_gib = RAM_GIB, onesite_kwargs = ONESITE_KWARGS)
    dt = time() - t0
    npart, _ = timer_total(to, "partial_ctmrgstep")
    _, t_ctm = timer_total(to, "EV: leading_boundary (CTMRG3D iterations)")
    _, t_num = timer_total(to, "EV: contract_onesite numerator (M)")
    _, t_den = timer_total(to, "EV: contract_onesite denominator (O)")
    conv = ϵs[end] < ctm_tol
    status = conv ? "converged" : timed_out ? "timed out" : length(ϵs) < ctm_maxiter ? "stopped (no progress)" : "not converged"
    return (; m = real(expvals[1]), iters = length(ϵs), npart, eps = ϵs[end], status, t_ctm, t_contract = t_num + t_den, time_s = dt)
end

# ======================================================================================================================
# CSV and summary
# ======================================================================================================================
const HEADER = "chi_env,chi_peps,D_CE,beta,projectors,west_updates,m,iterations,partial_steps,eps_last,status,t_ctm_s,t_contract_s,time_s"

function append_row(path, dims, β, projectors, west_updates, r)
    open(path, "a") do io
        println(io, join((dims..., β, projectors, west_updates, r.m, r.iters, r.npart, r.eps, r.status, round(r.t_ctm; digits = 2),
                          round(r.t_contract; digits = 2), round(r.time_s; digits = 2)), ","))
    end
    return nothing
end

# Per (set, β, projectors): every west_updates with m, |m - m of :auto|, iterations, West updates and the CTMRG3D time
function print_summary(rows)
    fmt(x) = isfinite(x) ? string(round(x; sigdigits = 6)) : "-"
    println("\n===== west_updates: m, |Δm| to :auto, iterations, West updates (partial_ctmrgstep), CTMRG3D time =====")
    for dims in DIMS, β in BETAS, projectors in PROJECTORS
        sub = filter(r -> r.dims == dims && r.β == β && r.projectors == projectors, rows)
        isempty(sub) && continue
        ref = filter(r -> r.west_updates === :auto, sub)
        m_ref = isempty(ref) ? NaN : abs(ref[1].m)
        println("$dims β = $β :$projectors")
        for r in sub
            println("    ", rpad(string(r.west_updates), 7), rpad("m = $(fmt(r.m))", 20), rpad("|Δ|m|| = $(fmt(abs(abs(r.m) - m_ref)))", 24),
                    rpad("$(r.iters) it", 8), rpad("$(r.npart) West updates", 20), rpad("t_CTM = $(round(r.t_ctm; digits = 1)) s", 20), r.status)
        end
    end
    return nothing
end

# ======================================================================================================================
# Main
# ======================================================================================================================
function main()
    BLAS.set_num_threads(BLAS_THREADS)
    outdir = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "checks", "check_ctmrg3d_west_updates"))
    mkpath(outdir)
    path = joinpath(outdir, Libc.strftime("%Y%m%d_%H%M", time()) * ".csv")
    write(path, HEADER * "\n")
    @info "Results: $path (threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads()))"
    obs = SpinOperators.σᶻ(T, spin_symmetry)

    # Warm-up (compilation, not stored): the smallest β and the first set, every projector kind and every west_updates, two iterations
    let PEPO = only(load_pepos([BETAS[1]], Δβ_iter, DIMS[1][3], J, g)), t0 = time()
        for projectors in PROJECTORS, w in WEST_UPDATES
            MPSKit.expectation_value(PEPO, [obs], ℂ^DIMS[1][2], ℂ^DIMS[1][1]; miniter = 1, maxiter = 2, tol = ctm_tol, projectors, west_updates = w,
                                     init = INIT, return_env = true, ram_gib = RAM_GIB, onesite_kwargs = ONESITE_KWARGS)
        end
        @info "Warm-up took $(round(time() - t0; digits = 1)) s"
    end

    rows = NamedTuple[]
    for dims in DIMS
        PEPOs = load_pepos(BETAS, Δβ_iter, dims[3], J, g)
        for (i, β) in enumerate(BETAS), projectors in PROJECTORS, w in WEST_UPDATES
            r = try
                run_one(PEPOs[i], obs, dims, projectors, w)
            catch err
                @error "$dims β = $β $projectors west_updates = $w failed" exception = (err, catch_backtrace())
                (; m = NaN, iters = 0, npart = 0, eps = NaN, status = "ERROR: " * replace(first(sprint(showerror, err), 100), "," => ";", "\n" => " "),
                 t_ctm = NaN, t_contract = NaN, time_s = NaN)
            end
            append_row(path, dims, β, projectors, w, r)
            push!(rows, (; dims, β, projectors, west_updates = w, r...))
            println("RUN $dims β = $β :$projectors west_updates = $w: m = $(r.m), $(r.iters) it, $(r.npart) West updates, ϵ = $(r.eps), $(r.status), ",
                    "t_CTM = $(round(r.t_ctm; digits = 1)) s, t_contract = $(round(r.t_contract; digits = 1)) s")
            flush(stdout)
        end
    end
    print_summary(rows)
    return nothing
end

main()
