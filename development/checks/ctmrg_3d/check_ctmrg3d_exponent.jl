# Check of the projector exponent P of CTMRG3D (the :svd projectors U * S^P; original P = 1/2, P = 0 = isometries) on the 3D Ising PEPOs of the PEPO file; prints numbers, tests nothing.
# For every set in DIMS, every β in BETAS and every P in EXPONENTS it stores m, the iterations, the last ϵ, the corner spectrum and the ϵ of every iteration in data/checks/check_ctmrg3d_exponent/<date>_<time>.csv.
# For P in M_STOP_EXPONENTS the corners become rank one and ϵ stops changing, so there we stop on the change of m instead (m every M_EVERY iterations).

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
# The β (must be in the PEPO file); a job can choose others with the environment variable CE_CHECK_BETAS, separated by _ (for example 0.2_0.22)
const BETAS = haskey(ENV, "CE_CHECK_BETAS") ? parse.(Float64, split(ENV["CE_CHECK_BETAS"], "_")) : [0.19, 0.21, 0.25]   # default: below, near and above βc ≈ 0.2217
const EXPONENTS = [0.0, 0.5, 1.0]        # P of the :svd projectors U * S^P
const INIT = :random                     # cold start of every run (the start of the CTMRG3D compare)
const INIT_SEED = 1
const BLAS_THREADS = 4                   # CTMRG3D gets its parallelism from BLAS (as the CTMRG3D compare on the laptop)

# CTMRG3D settings: those of the CTMRG3D compare (ising_model_3D_CE_CTMRG_compare.jl)
const ctm_miniter = 3
const CTM_MAXITER = 300                  # more than the 80 of the compare: P = 0 converges slowly
const ctm_tol = 1e-7
const WEST_UPDATES = :1
const MAX_SECONDS = 1800.0               # time limit of the CTMRG3D iterations of one run (s; P = 0 converges slowly)
# Stop on the change of m (only for the P in M_STOP_EXPONENTS): after at least M_MINITER iterations we compute m every M_EVERY iterations and stop when it
# changed less than M_TOL twice in a row; ϵ is then not used. M_MAX_SECONDS: time limit of these runs (the contractions of m count for it).
# CHOP_RTOL: in these runs we set the entries of C, E and A below CHOP_RTOL × their largest entry to zero after every step. The small corner values shrink
# geometrically at P = 1 and become subnormal numbers, which make the contraction of m many times slower; at 1e-30 the result does not change.
const M_STOP_EXPONENTS = [1.0]
const M_MINITER = 20
const M_EVERY = 5
const M_TOL = 1e-6
const M_MAX_SECONDS = 7200.0
const CHOP_RTOL = 1e-30
const RAM_GIB = 3                        # RAM budget of the sliced contraction (contract_onesite_auto); nothing = unsliced contract_onesite

# reads the environment variable CE_CHECK_MACHINE, if not set: use "laptop", pbs files all export CE_CHECK_MACHINE=hortense
const MACHINE = get(ENV, "CE_CHECK_MACHINE", "laptop")   
# contract_onesite_auto uses the plan times of this machine

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

function run_one(PEPO, obs, dims, β, P)
    P in M_STOP_EXPONENTS && return run_one_mstop(PEPO, obs, dims, β, P)
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
            time_s = dt, rank = count(>(RANK_TOL), s), spec = [i <= length(s) ? s[i] : 0.0 for i in 1:NSPEC], eps_all = ϵs, m_all = Float64[])
end

# O and M of the PEPO, as in expectation_value (ctmrg3D_main.jl)
function O_and_M(PEPO, obs)
    @tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]
    O = flip(permute(O, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])
    @tensor M[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 2; -1 -2 -3 -4 -5 -6] * obs[2; 1]
    M = flip(permute(M, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])
    return O, M
end

# Entries below CHOP_RTOL × the largest entry of t set to zero (see CHOP_RTOL)
function chop!(t)
    x = t.data
    thr = CHOP_RTOL * maximum(abs, x)
    x[abs.(x) .< thr] .= 0
    return t
end

# The same run, but we do the CTMRG3D iterations here and stop on the change of m (see M_STOP_EXPONENTS); m_all: m at every check
function run_one_mstop(PEPO, obs, dims, β, P)
    chienv, chipeps, _ = dims
    t0 = time()
    O, M = O_and_M(PEPO, obs)
    env = CE.CTMRG3D(O; trunc_env = truncrank(chienv), trunc_bpeps = truncrank(chipeps), Vbp = ℂ^chipeps, Vv = ℂ^chienv, miniter = ctm_miniter,
                     maxiter = CTM_MAXITER, tol = ctm_tol, verbosity = 0, west_updates = WEST_UPDATES, projectors = :svd, projector_exponent = P,
                     init = INIT, init_seed = INIT_SEED)
    mval() = real(CE.contract_onesite_dispatch(env, M, RAM_GIB; ONESITE_KWARGS...) / CE.contract_onesite_dispatch(env, O, RAM_GIB; ONESITE_KWARGS...))
    S_old = id(CE.envspace(env))
    ϵs, m_all = Float64[], Float64[]
    small, status = 0, "not converged"
    for i in 1:CTM_MAXITER
        S_new = CE.ctmrgstep(env)
        foreach(chop!, env.C); foreach(chop!, env.E); foreach(chop!, env.A)
        push!(ϵs, CE.spectrum_change(S_new, S_old))
        S_old = copy(S_new)
        if i >= M_MINITER && (i - M_MINITER) % M_EVERY == 0
            push!(m_all, mval())
            small = length(m_all) >= 2 && abs(m_all[end] - m_all[end - 1]) < M_TOL ? small + 1 : 0
            small >= 2 && (status = "converged (m)"; break)
        end
        time() - t0 > M_MAX_SECONDS && (status = "timed out"; break)
    end
    isempty(m_all) && push!(m_all, mval())
    s = corner_values(env)
    return (; m = m_all[end], iters = length(ϵs), eps = ϵs[end], status, time_s = time() - t0, rank = count(>(RANK_TOL), s),
            spec = [i <= length(s) ? s[i] : 0.0 for i in 1:NSPEC], eps_all = ϵs, m_all)
end

# ======================================================================================================================
# CSV and summary
# ======================================================================================================================
const HEADER = "chi_env,chi_peps,D_CE,beta,P,m,iterations,eps_last,status,time_s,rank," * join(("s$i" for i in 1:NSPEC), ",") * ",eps_all,m_all"

function append_row(path, dims, β, P, r)
    open(path, "a") do io
        println(io, join((dims..., β, P, r.m, r.iters, r.eps, r.status, round(r.time_s; digits = 2), r.rank, r.spec..., join(r.eps_all, ";"), join(r.m_all, ";")), ","))
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
                (; m = NaN, iters = 0, eps = NaN, status = "ERROR: " * first(sprint(showerror, err), 100), time_s = NaN, rank = 0, spec = fill(NaN, NSPEC), eps_all = Float64[],
                   m_all = Float64[])
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
