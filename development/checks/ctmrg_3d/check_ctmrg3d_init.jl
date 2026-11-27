# Check of the cold start of CTMRG3D: init = :ones against :random and :nishino_okunishi (several seeds each), for both projector kinds (:svd and :oblique), on the 3D Ising PEPOs of the PEPO file; prints numbers, tests nothing.
# For every set in DIMS, every β in BETAS and every projector kind it stores m, the iterations, the last ϵ and the corner spectrum per start in data/checks/check_ctmrg3d_init/<date>_<time>.csv.

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using LinearAlgebra
using Random
using JLD2
const CE = ClusterExpansions

# load_pepos, pepo_file (the PEPO file of stage 1)
include(joinpath(@__DIR__, "..", "..", "initial_pepos", "pepo_files.jl"))

# ======================================================================================================================
# Settings
# ======================================================================================================================
const DIMS = [(8, 4, 3), (11, 5, 3)]     # (χenv, χpeps, D_CE); (11, 5, 3) is the laptop set of the CTMRG3D compare
const BETAS = [0.19, 0.21, 0.25]         # below, near and above βc ≈ 0.2217 (must be in the PEPO file)
const PROJECTORS = [:svd, :oblique]
const SEEDS = [1, 2, 3]                  # init_seed of the random starts
const BLAS_THREADS = 4                   # CTMRG3D gets its parallelism from BLAS (as the CTMRG3D compare on the laptop)

# CTMRG3D settings: those of the CTMRG3D compare (ising_model_3D_CE_CTMRG_compare.jl)
const ctm_miniter = 3
const ctm_maxiter = 80
const ctm_tol = 1e-7
const WEST_UPDATES = :1
const MAX_SECONDS = 600.0                # time limit of the CTMRG3D iterations of one run (s)
const RAM_GIB = 3                        # RAM budget of the sliced contraction (contract_onesite_auto); nothing = unsliced contract_onesite
const MACHINE = get(ENV, "CE_CHECK_MACHINE", "laptop")   # contract_onesite_auto uses the plan times of this machine
# false: contract_onesite_auto slices the current tree itself (plans = nothing)
const USE_PLANS = get(ENV, "CE_CHECK_PLANS", "true") == "true"
const ONESITE_KWARGS = USE_PLANS ? (; machine = MACHINE) : (; machine = MACHINE, plans = nothing)

const RANK_TOL = 1e-12                   # a corner singular value (relative to the largest) above this counts for the rank
const NSPEC = 4                          # number of corner singular values written into the CSV

const J = 1.0
const g = 0.0
const Δβ_iter = 0.0025
const T = Float64
const spin_symmetry = Trivial

# ======================================================================================================================
# One run
# ======================================================================================================================
# The corner spectrum of C[1] (corner_spectrum of src/ctmrg3D/ctmrg3D_helpers.jl), sorted and divided by the largest value
function corner_values(env)
    s = sort(abs.(diag(convert(Array, CE.corner_spectrum(env)))); rev = true)
    return s ./ s[1]
end

function run_one(PEPO, obs, dims, β, projectors, init, seed)
    chienv, chipeps, _ = dims
    t0 = time()
    expvals, env, ϵs, timed_out = MPSKit.expectation_value(PEPO, [obs], ℂ^chipeps, ℂ^chienv; miniter = ctm_miniter, maxiter = ctm_maxiter, tol = ctm_tol,
                                                            verbosity = 0, west_updates = WEST_UPDATES, projectors, init, init_seed = seed,
                                                            max_seconds = MAX_SECONDS, return_env = true, ram_gib = RAM_GIB,
                                                            onesite_kwargs = ONESITE_KWARGS)
    dt = time() - t0
    s = corner_values(env)
    conv = ϵs[end] < ctm_tol
    return (; m = real(expvals[1]), iters = length(ϵs), eps = ϵs[end], status = conv ? "converged" : timed_out ? "timed out" : "not converged",
            time_s = dt, rank = count(>(RANK_TOL), s), spec = [i <= length(s) ? s[i] : 0.0 for i in 1:NSPEC])
end

# ======================================================================================================================
# CSV and summary
# ======================================================================================================================
const HEADER = "chi_env,chi_peps,D_CE,beta,projectors,init,seed,m,iterations,eps_last,status,time_s,rank," * join(("s$i" for i in 1:NSPEC), ",")

function append_row(path, dims, β, projectors, init, seed, r)
    open(path, "a") do io
        println(io, join((dims..., β, projectors, init, seed, r.m, r.iters, r.eps, r.status, round(r.time_s; digits = 2), r.rank, r.spec...), ","))
    end
    return nothing
end

function print_summary(rows)
    fmt(x) = string(round(x; sigdigits = 6))
    println("\n===== |m| per start (ones; random: mean ± spread over the seeds), iterations, corner rank =====")
    println(rpad("(χenv, χpeps, D_CE)", 21), rpad("β", 7), rpad("projectors", 11), rpad("|m| ones (it, rank)", 30), rpad("|m| random mean ± spread", 36),
            rpad("|m| Nishino-Okunishi mean ± spread (mean it)", 48), "|ones - random|, |N-O - random|")
    for dims in DIMS, β in BETAS, projectors in PROJECTORS
        sub = filter(r -> r.dims == dims && r.β == β && r.projectors == projectors, rows)
        o = filter(r -> r.init == :ones, sub)
        rnd = filter(r -> r.init == :random && isfinite(r.m), sub)
        isempty(o) && isempty(rnd) && continue
        ones_txt = isempty(o) || !isfinite(o[1].m) ? "-" : "$(fmt(abs(o[1].m))) ($(o[1].iters), $(o[1].rank)$(o[1].status == "converged" ? "" : ", " * o[1].status))"
        ms = [abs(r.m) for r in rnd]
        mean = isempty(ms) ? NaN : sum(ms) / length(ms)
        spread = isempty(ms) ? NaN : maximum(ms) - minimum(ms)
        rnd_txt = isempty(ms) ? "-" : "$(fmt(mean)) ± $(fmt(spread)) ($(length(ms)) seeds)"
        no = filter(r -> r.init == :nishino_okunishi && isfinite(r.m), sub)
        mn = [abs(r.m) for r in no]
        no_mean = isempty(mn) ? NaN : sum(mn) / length(mn)
        no_txt = isempty(mn) ? "-" : "$(fmt(no_mean)) ± $(fmt(maximum(mn) - minimum(mn))) ($(round(sum(r.iters for r in no) / length(no); digits = 1)) it)"
        diff = isempty(o) || isempty(ms) ? "-" : fmt(abs(abs(o[1].m) - mean))
        diff_no = isempty(mn) || isempty(ms) ? "-" : fmt(abs(no_mean - mean))
        println(rpad("$dims", 21), rpad(β, 7), rpad(projectors, 11), rpad(ones_txt, 30), rpad(rnd_txt, 36), rpad(no_txt, 48), diff, ", ", diff_no)
    end
    return nothing
end

# ======================================================================================================================
# Main
# ======================================================================================================================
function main()
    BLAS.set_num_threads(BLAS_THREADS)
    outdir = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "checks", "check_ctmrg3d_init"))
    mkpath(outdir)
    path = joinpath(outdir, Libc.strftime("%Y%m%d_%H%M", time()) * ".csv")
    write(path, HEADER * "\n")
    @info "Results: $path (threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads()))"
    obs = SpinOperators.σᶻ(T, spin_symmetry)

    # Warm-up (compilation, not stored): the smallest β, the first set, both projector kinds, a few iterations
    let PEPO = only(load_pepos([BETAS[1]], Δβ_iter, DIMS[1][3], J, g)), t0 = time()
        for projectors in PROJECTORS
            MPSKit.expectation_value(PEPO, [obs], ℂ^DIMS[1][2], ℂ^DIMS[1][1]; miniter = 1, maxiter = 2, tol = ctm_tol, projectors, init = :random,
                                     return_env = true, ram_gib = RAM_GIB, onesite_kwargs = ONESITE_KWARGS)
        end
        @info "Warm-up took $(round(time() - t0; digits = 1)) s"
    end

    rows = NamedTuple[]
    for dims in DIMS
        PEPOs = load_pepos(BETAS, Δβ_iter, dims[3], J, g)
        for (i, β) in enumerate(BETAS), projectors in PROJECTORS, (init, seed) in [(:ones, 0); [(:random, s) for s in SEEDS]; [(:nishino_okunishi, s) for s in SEEDS]]
            r = try
                run_one(PEPOs[i], obs, dims, β, projectors, init, max(seed, 1))
            catch err
                @error "$dims β = $β $projectors $init seed $seed failed" exception = (err, catch_backtrace())
                (; m = NaN, iters = 0, eps = NaN, status = "ERROR: " * first(sprint(showerror, err), 100), time_s = NaN, rank = 0, spec = fill(NaN, NSPEC))
            end
            append_row(path, dims, β, projectors, init, seed, r)
            push!(rows, (; dims, β, projectors, init, seed, r...))
            println("RUN $dims β = $β $projectors $init", init == :ones ? "" : " seed $seed", ": m = $(r.m), $(r.iters) it, ϵ = $(r.eps), $(r.status), ",
                    "$(round(r.time_s; digits = 1)) s, corner rank $(r.rank), spectrum $(round.(r.spec; sigdigits = 3))")
            flush(stdout)
        end
    end
    print_summary(rows)
    return nothing
end

main()
