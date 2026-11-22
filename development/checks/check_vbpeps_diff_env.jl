# Check of the extra term of the gradient of the vbPEPS (Lander's, BetterPEPSKit, development/fast_vbpeps/fast_bpeps.jl): differentiate_environment = false
# (fixed environment, the gradient without d(cost)/d(env) · d(env)/d(ψ)) against true (with it, ImplicitGradient). Prints numbers, tests nothing.
#
#     dE/dψ = ∂E/∂ψ + (∂E/∂env)(∂env/∂ψ)
#
# Theoretically for C4V and Oh symmetry, both should give the same result, while differentiate_environment = false should cost way less resources, so the compare runs use false only.
# For every set in DIMS and every β in BETAS: one cold observation for each value of differentiate_environment, from the same start state (the PEPO file)
# and seed, with the settings of the vbPEPS compare (FAST_KWARGS there: dense decompositions, C4v CTMRG). Per run we store the cost f, m = <σᶻ>,
# the number of cost-function calls, timed out or not, and the time; the summary prints |Δf| and |Δm| between false and true.
# Error source: contraction/optimization (3) only.
#
# Runs in the BetterPEPSKit environment (no Plots); every run is appended to data/checks/check_vbpeps_diff_env/<date>_<time>.csv immediately.
#   julia +1.12 -t 2 --project=development/fast_vbpeps development/checks/check_vbpeps_diff_env.jl
# On Hortense: development/hpc/job_check_vbpeps_diff_env.pbs (2 cores, round 1 of the Hortense guide). Needs the PEPO file (stage 1).

using TensorKit
using TimerOutputs
using Random
using LinearAlgebra
using JLD2

basename(dirname(Base.active_project())) == "fast_vbpeps" ||
    error("Run this check in the BetterPEPSKit environment: julia -t 2 --project=development/fast_vbpeps development/checks/check_vbpeps_diff_env.jl")
include(joinpath(@__DIR__, "..", "fast_vbpeps", "fast_bpeps.jl"))           # module FastBpeps
using BetterPEPSKit
#   pepo_files.jl:  load_bpeps_tensors (the PEPO file of stage 1)
include(joinpath(@__DIR__, "..", "profiling", "tools", "pepo_files.jl"))

# ======================================================================================================================
# Settings
# ======================================================================================================================
const DIMS = [(14, 2, 2), (27, 3, 3)]    # (χenv, χpeps, D_CE): the laptop set of the vbPEPS compare and the smallest Hortense candidate
const BETAS = [0.19, 0.21, 0.25]
const DIFFS = [false, true]              # differentiate_environment
const MAX_SECONDS = 3600.0               # time limit of one observation (s)
const SEED = 1000
const BLAS_THREADS = 1                   # as the vbPEPS compare: one Julia thread per core, 1 BLAS thread

const J = 1.0
const g = 0.0
const Δβ_iter = 0.005
const MAXITER_INIT = 20                  # power-method steps of the start states in the PEPO file

# The settings of the vbPEPS compare (FAST_KWARGS and base_kwargs there)
const FAST_KWARGS = (; boundary_c4v = true, symmetric = true, sparse = false, truncation_window = 4, truncation_rtol = 1.0e-3, gradient_maxiter = 20,
                     gradient_krylovdim = 30, boundary_tol = 1.0e-10, boundary_maxiter = 1500, boundary_verbosity = 1, gradient_tol = 1.0e-6,
                     optimizer_memory = 64, optimizer_tol = 1.0e-5, optimizer_maxiter = 100, optimizer_verbosity = 1)

# ======================================================================================================================
# Main
# ======================================================================================================================
const HEADER = "chi_env,chi_peps,D_CE,beta,diff_env,f,m,calls,timed_out,time_s,check"

function main()
    BLAS.set_num_threads(BLAS_THREADS)
    outdir = normpath(joinpath(@__DIR__, "..", "..", "data", "checks", "check_vbpeps_diff_env"))
    mkpath(outdir)
    path = joinpath(outdir, Libc.strftime("%Y%m%d_%H%M", time()) * ".csv")
    write(path, HEADER * "\n")
    @info "Results: $path (threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads()))"

    # Warm-up (compilation, not stored): the first set and β, both values, one LBFGS iteration
    let (chienv, chipeps, D_CE) = DIMS[1], inp = only(load_bpeps_tensors(BETAS[1:1], Δβ_iter, D_CE, J, g, chipeps; maxiter_init = MAXITER_INIT)), t0 = time()
        for diff in DIFFS
            FastBpeps.run_observation("warmup", inp.O, inp.M, inp.psi0, chipeps, chienv; warm = false, seed = SEED,
                                      FAST_KWARGS..., differentiate_environment = diff, optimizer_maxiter = 1, max_seconds = MAX_SECONDS)
        end
        @info "Warm-up took $(round(time() - t0; digits = 1)) s"
    end

    rows = NamedTuple[]
    for (chienv, chipeps, D_CE) in DIMS
        inputs = load_bpeps_tensors(BETAS, Δβ_iter, D_CE, J, g, chipeps; maxiter_init = MAXITER_INIT)
        for (i, β) in enumerate(BETAS), diff in DIFFS
            FastBpeps.reset_timer()
            t0 = time()
            r = try
                res = FastBpeps.run_observation("check", inputs[i].O, inputs[i].M, inputs[i].psi0, chipeps, chienv; warm = false, seed = SEED,
                                                FAST_KWARGS..., differentiate_environment = diff, max_seconds = MAX_SECONDS)
                (; f = res.cost, m = real(res.m), calls = res.numfg, timed_out = res.timed_out, time_s = time() - t0, check = "ok")
            catch err
                @error "($chienv, $chipeps, $D_CE) β = $β diff = $diff failed" exception = (err, catch_backtrace())
                (; f = NaN, m = NaN, calls = 0, timed_out = false, time_s = time() - t0, check = "ERROR: " * first(sprint(showerror, err), 100))
            end
            open(path, "a") do io
                println(io, join((chienv, chipeps, D_CE, β, diff, r.f, r.m, r.calls, r.timed_out, round(r.time_s; digits = 2), replace(r.check, "," => ";")), ","))
            end
            push!(rows, (; dims = (chienv, chipeps, D_CE), β, diff, r...))
            println("RUN ($chienv, $chipeps, $D_CE) β = $β diff_env = $diff: f = $(r.f), m = $(r.m), $(r.calls) calls, timed out = $(r.timed_out), ",
                    "$(round(r.time_s; digits = 1)) s, $(r.check)")
            flush(stdout)
        end
    end

    println("\n===== fixed (false) against differentiated (true) environment =====")
    println(rpad("(χenv, χpeps, D_CE)", 21), rpad("β", 7), rpad("|Δf|", 12), rpad("|Δ|m||", 12), rpad("calls false/true", 18), "time false/true (s)")
    for (dims, β) in unique((r.dims, r.β) for r in rows)
        a = filter(r -> r.dims == dims && r.β == β && !r.diff, rows)
        b = filter(r -> r.dims == dims && r.β == β && r.diff, rows)
        (isempty(a) || isempty(b)) && continue
        a, b = a[1], b[1]
        println(rpad("$dims", 21), rpad(β, 7), rpad(round(abs(a.f - b.f); sigdigits = 3), 12), rpad(round(abs(abs(a.m) - abs(b.m)); sigdigits = 3), 12),
                rpad("$(a.calls)/$(b.calls)", 18), "$(round(a.time_s; digits = 1))/$(round(b.time_s; digits = 1))")
    end
    @info "Peak memory of the process $(round(Sys.maxrss() / 2^30; digits = 2)) GiB"
    return nothing
end

main()
