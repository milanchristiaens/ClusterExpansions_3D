# Check of the projector exponent P of CTMRG3D (src/ctmrg3D/): the :svd projectors U * S^P (projector_exponent; original P = 1/2, P = 0 = isometries),
# for P in EXPONENTS,
# on the 3D Ising PEPOs of the PEPO file (stage 1, development/initial_pepos/ising_model_3D_CE_pepos.jl). Prints numbers, tests nothing.
#
# For every set in DIMS, every β in BETAS and every P we run MPSKit.expectation_value (the settings of the CTMRG3D compare, except maxiter = CTM_MAXITER,
# because P = 0 converges slowly or not at all) from the cold start INIT (seed INIT_SEED), and store per run:
#   m = <σᶻ>, the number of iterations, the last ϵ, converged or not (or timed out), the time, the corner spectrum of the converged environment
#   (rank = values above RANK_TOL) and the ϵ of every iteration (column eps_all, separated by ";": the convergence curve).
# Error source: contraction/environment (3) only; the PEPO (D_CE, Δβ, β) is the same for all starts.
#
# Every run is appended to data/checks/check_ctmrg3d_exponent/<date>_<time>.csv immediately (a crash keeps the earlier runs).
# Run in the ClusterExpansions environment, with the threads of the CTMRG3D compare (PowerShell, in the clone folder):
#   julia +1.12 -t 4 --project=. development/checks/ctmrg_3d/check_ctmrg3d_exponent.jl
# On Hortense: development/hpc/job_check_ctmrg3d.pbs (round 1 of the Hortense guide; MACHINE is then "hortense" via CE_CHECK_MACHINE).

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
const DIMS = [(8, 4, 3), (11, 5, 3)]     # (χenv, χpeps, D_CE)
const BETAS = [0.19, 0.21, 0.25]         # below, near and above βc ≈ 0.2217 (must be in the PEPO file)
const EXPONENTS = [0.0, 0.5, 1.0]        # P of the :svd projectors U * S^P
const INIT = :random                     # cold start of every run (the start of the CTMRG3D compare)
const INIT_SEED = 1
const BLAS_THREADS = 4                   # CTMRG3D gets its parallelism from BLAS (as the CTMRG3D compare on the laptop)

# CTMRG3D settings: those of the CTMRG3D compare (ising_model_3D_CE_CTMRG_compare.jl)
const ctm_miniter = 3
const CTM_MAXITER = 300                  # more than the 80 of the compare: P = 0 converges slowly
const ctm_tol = 1e-7
const WEST_UPDATES = :auto
const MAX_SECONDS = 1800.0               # time limit of the CTMRG3D iterations of one run (s; P = 0 converges slowly)
const RAM_GIB = 3                        # RAM budget of the sliced contraction (contract_onesite_auto); nothing = unsliced contract_onesite

# reads the environment variable CE_CHECK_MACHINE, if not set: use "laptop", pbs files all export CE_CHECK_MACHINE=hortense
const MACHINE = get(ENV, "CE_CHECK_MACHINE", "laptop")   
# contract_onesite_auto uses the plan times of this machine

# false: contract_onesite_auto slices the current tree itself (plans = nothing), as the pilots on Hortense in round 1 (the plans are timed in the same round)
const USE_PLANS = get(ENV, "CE_CHECK_PLANS", "true") == "true"
const ONESITE_KWARGS = USE_PLANS ? (; machine = MACHINE) : (; machine = MACHINE, plans = nothing)

const RANK_TOL = 1e-12                   # a corner singular value (relative to the largest) above this counts for the rank
const NSPEC = 4                          # number of corner singular values written into the CSV

const J = 1.0
const g = 0.0
const Δβ_iter = 0.005
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

function run_one(PEPO, obs, dims, β, P)
    chienv, chipeps, _ = dims
    t0 = time()
    expvals, env, ϵs, timed_out = MPSKit.expectation_value(PEPO, [obs], ℂ^chipeps, ℂ^chienv; miniter = ctm_miniter, maxiter = CTM_MAXITER, tol = ctm_tol,
                                                            verbosity = 0, west_updates = WEST_UPDATES, projectors = :svd, projector_exponent = P,
                                                            init = INIT, init_seed = INIT_SEED, max_seconds = MAX_SECONDS, return_env = true,
                                                            ram_gib = RAM_GIB, onesite_kwargs = ONESITE_KWARGS)
    dt = time() - t0
    s = corner_values(env)
    conv = ϵs[end] < ctm_tol
    return (; m = real(expvals[1]), iters = length(ϵs), eps = ϵs[end], status = conv ? "converged" : timed_out ? "timed out" : "not converged",
            time_s = dt, rank = count(>(RANK_TOL), s), spec = [i <= length(s) ? s[i] : 0.0 for i in 1:NSPEC], eps_all = ϵs)
end

# ======================================================================================================================
# CSV and summary
# ======================================================================================================================
const HEADER = "chi_env,chi_peps,D_CE,beta,P,m,iterations,eps_last,status,time_s,rank," * join(("s$i" for i in 1:NSPEC), ",") * ",eps_all"

function append_row(path, dims, β, P, r)
    open(path, "a") do io
        println(io, join((dims..., β, P, r.m, r.iters, r.eps, r.status, round(r.time_s; digits = 2), r.rank, r.spec..., join(r.eps_all, ";")), ","))
    end
    return nothing
end

function print_summary(rows)
    fmt(x) = string(round(x; sigdigits = 6))
    println("\n===== |m|, iterations, last ϵ, status, time and corner rank per P (:svd) =====")
    println(rpad("(χenv, χpeps, D_CE)", 21), rpad("β", 7), rpad("P", 6), rpad("|m|", 14), rpad("it", 6), rpad("ϵ", 12), rpad("status", 16), rpad("time (s)", 10), "rank")
    for r in rows
        println(rpad("$(r.dims)", 21), rpad(r.β, 7), rpad(r.P, 6), rpad(fmt(abs(r.m)), 14), rpad(r.iters, 6), rpad(fmt(r.eps), 12), rpad(r.status, 16),
                rpad(round(r.time_s; digits = 1), 10), r.rank)
    end
    return nothing
end

# ======================================================================================================================
# Main
# ======================================================================================================================
function main()
    BLAS.set_num_threads(BLAS_THREADS)
    outdir = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "checks", "check_ctmrg3d_exponent"))
    mkpath(outdir)
    path = joinpath(outdir, Libc.strftime("%Y%m%d_%H%M", time()) * ".csv")
    write(path, HEADER * "\n")
    @info "Results: $path (threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads()))"
    obs = SpinOperators.σᶻ(T, spin_symmetry)

    # Warm-up (compilation, not stored): the smallest β, the first set, a few iterations
    let PEPO = only(load_pepos([BETAS[1]], Δβ_iter, DIMS[1][3], J, g)), t0 = time()
        MPSKit.expectation_value(PEPO, [obs], ℂ^DIMS[1][2], ℂ^DIMS[1][1]; miniter = 1, maxiter = 2, tol = ctm_tol, projectors = :svd, projector_exponent = 0.0,
                                 init = INIT, return_env = true, ram_gib = RAM_GIB, onesite_kwargs = ONESITE_KWARGS)
        @info "Warm-up took $(round(time() - t0; digits = 1)) s"
    end

    rows = NamedTuple[]
    for dims in DIMS
        PEPOs = load_pepos(BETAS, Δβ_iter, dims[3], J, g)
        for (i, β) in enumerate(BETAS), P in EXPONENTS
            r = try
                run_one(PEPOs[i], obs, dims, β, P)
            catch err
                @error "$dims β = $β P = $P failed" exception = (err, catch_backtrace())
                (; m = NaN, iters = 0, eps = NaN, status = "ERROR: " * first(sprint(showerror, err), 100), time_s = NaN, rank = 0, spec = fill(NaN, NSPEC), eps_all = Float64[])
            end
            append_row(path, dims, β, P, r)
            push!(rows, (; dims, β, P, r...))
            println("RUN $dims β = $β P = $P: m = $(r.m), $(r.iters) it, ϵ = $(r.eps), $(r.status), $(round(r.time_s; digits = 1)) s, corner rank $(r.rank), ",
                    "spectrum $(round.(r.spec; sigdigits = 3))")
            flush(stdout)
        end
    end
    print_summary(rows)
    return nothing
end

main()
