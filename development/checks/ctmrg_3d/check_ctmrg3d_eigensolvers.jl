# Check of the eigensolvers of ρ_F in CTMRG3DOctant (:subspace, :block_lanczos, :dense_block_lanczos, :dense_eigen) on the exact Ising PEPO; prints numbers, tests nothing.
# Part 1 (one step): on the same converged environment every solver finds P_c once; we compare the time, the number of products ρ_F·v and the eigenvalues and
# eigenvectors with :dense_eigen. Part 2 (whole runs): every solver runs the octant from the same cold start; we compare iterations, time and m.

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using LinearAlgebra
using Printf
using JLD2
const CE = ClusterExpansions

# exact_ising_pepo
include(joinpath(@__DIR__, "..", "..", "initial_pepos", "pepo_files.jl"))

# ======================================================================================================================
# Settings (a Hortense job may pass CE_CHECK_DIMS, for example 16_6_2, CE_CHECK_BETAS, for example 0.21_0.23, CE_CHECK_PARTS and CE_CHECK_MAXITER)
# ======================================================================================================================
parse_dims(s) = [Tuple(parse.(Int, split(x, "_"))) for x in split(s, ",")]
const DIMS = parse_dims(get(ENV, "CE_CHECK_DIMS", "8_3_2,12_4_2"))       # (χenv, χpeps, D = 2): the exact PEPO
const BETAS = parse.(Float64, split(get(ENV, "CE_CHECK_BETAS", "0.21_0.23"), "_"))
const PARTS = parse.(Int, split(get(ENV, "CE_CHECK_PARTS", "1,2"), ","))     # which parts run
const SOLVERS = [:subspace, :block_lanczos, :dense_block_lanczos, :dense_eigen]
const FACE_PROJECTOR = :lead_weighted
const PRE_STEPS = 40                    # part 1: steps with :subspace before the comparison (the environment of a run near its end)
const REPS = 3                          # part 1: repetitions per solver (the fastest counts)
const MINITER = 3                       # part 2: the settings of the octant compare runs
const MAXITER = parse(Int, get(ENV, "CE_CHECK_MAXITER", "150"))
const TOL = 1e-7
const INIT = :random
const RAM_GIB = 3                       # the sliced contraction of m (nothing = unsliced)
const MACHINE = get(ENV, "CE_CHECK_MACHINE", "laptop")
const BLAS_THREADS = parse(Int, get(ENV, "CE_CHECK_BLAS", "4"))
# ======================================================================================================================

function O_and_M(PEPO, obs)
    @tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]
    O = flip(permute(O, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])
    @tensor M[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 2; -1 -2 -3 -4 -5 -6] * obs[2; 1]
    M = flip(permute(M, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])
    return O, M
end

octant(O, dims, solver) = CE.CTMRG3DOctant(O; Vv = ℂ^dims[1], Vbp = ℂ^dims[2], miniter = MINITER, maxiter = MAXITER, tol = TOL, init = INIT,
                                           face_projector = FACE_PROJECTOR, eigensolver = solver)

# smallest cosine of the principal angles between the column spaces of U and V (1 = the same space)
overlap(U, V) = minimum(svdvals(U' * V))

# Part 1: P_c of one step on the same environment
function part1(O, dims, β)
    env = octant(O, dims, :subspace)
    for _ in 1:PRE_STEPS
        CE.ctmrgstep(env)
    end
    χ, p, D = size(env.e, 1), size(env.e, 3), size(env.o, 1)
    nkeep = min(env.chi_env, χ * p * p * D)
    B = CE._octant_B(env.c, env.e, env.a)
    res = Dict{Tuple{Symbol, Bool}, Any}()
    for solver in [:dense_eigen; filter(!=(:dense_eigen), SOLVERS)], warm in (true, false)
        e2 = deepcopy(env)
        e2.eigensolver = solver
        warm || (e2.P = nothing)
        best, out, ncols = Inf, nothing, 0
        for _ in 1:REPS
            CE._octant_rho_columns[] = 0
            t = @elapsed out = CE._octant_top_eigvecs(e2, B, nkeep)
            best, ncols = min(best, t), CE._octant_rho_columns[]
        end
        res[(solver, warm)] = (; t = best, ncols, U = out[1], λ = out[2])
    end
    F = χ * p * p * D
    for warm in (true, false), solver in SOLVERS
        r, ref = res[(solver, warm)], res[(:dense_eigen, warm)]
        k = min(length(r.λ), length(ref.λ))
        dλ = maximum(abs.(r.λ[1:k] .- ref.λ[1:k])) / ref.λ[1]
        ov = size(r.U, 2) == size(ref.U, 2) ? overlap(ref.U, r.U) : NaN
        @printf("P1 %s β=%.4f F=%d %-20s %-5s time %8.3f s  products %6d (%.1f F)  max|Δλ|/λ1 %.1e  overlap %.12f  kept %d\n",
                string(dims), β, F, solver, warm ? "warm" : "cold", r.t, r.ncols, r.ncols / F, dλ, ov, size(r.U, 2))
        println(OUT[], join((1, dims..., β, F, solver, warm, r.t, r.ncols, dλ, ov, size(r.U, 2), "", "", ""), ","))
    end
    flush(stdout); flush(OUT[])
    return nothing
end

# Part 2: whole runs from the same cold start
function part2(O, M, dims, β)
    for solver in SOLVERS
        env = octant(O, dims, solver)
        CE._octant_rho_columns[] = 0
        t = @elapsed (env, ϵs, _) = leading_boundary(env)
        ncols = CE._octant_rho_columns[]
        m = real(CE.contract_onesite_dispatch(env, M, RAM_GIB; machine = MACHINE, plans = nothing) /
                 CE.contract_onesite_dispatch(env, O, RAM_GIB; machine = MACHINE, plans = nothing))
        @printf("P2 %s β=%.4f %-20s %4d it  ϵ %.1e  time %9.1f s (%.3f s/it)  products %8d  m %.10f\n", string(dims), β, solver, length(ϵs), ϵs[end], t,
                t / length(ϵs), ncols, m)
        println(OUT[], join((2, dims..., β, "", solver, "", t, ncols, "", "", "", length(ϵs), ϵs[end], m), ","))
        flush(stdout); flush(OUT[])
    end
    return nothing
end

const OUT = Ref{IO}(stdout)

function main()
    BLAS.set_num_threads(BLAS_THREADS)
    outdir = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "checks", "check_ctmrg3d_eigensolvers"))
    mkpath(outdir)
    path = joinpath(outdir, Libc.strftime("%Y%m%d_%H%M", time()) * ".csv")
    OUT[] = open(path, "w")
    println(OUT[], "part,chi_env,chi_peps,D,beta,F,solver,warm,time_s,products,dlambda_rel,overlap,kept,iterations,eps_last,m")
    @info "Results: $path (threads $(Threads.nthreads()), BLAS threads $(BLAS.get_num_threads()))"
    obs = SpinOperators.σᶻ(Float64, Trivial)
    # warm-up (compilation) at tiny dimensions
    let (O, M) = O_and_M(exact_ising_pepo(BETAS[1]), obs)
        for solver in SOLVERS
            env = CE.CTMRG3DOctant(O; Vv = ℂ^4, Vbp = ℂ^2, miniter = 1, maxiter = 2, tol = TOL, init = INIT, eigensolver = solver)
            leading_boundary(env)
        end
        CE.contract_onesite_dispatch(octant(O, (4, 2, 2), :subspace), M, RAM_GIB; machine = MACHINE, plans = nothing)
    end
    for dims in DIMS, β in BETAS
        O, M = O_and_M(exact_ising_pepo(β), obs)
        1 in PARTS && part1(O, dims, β)
        2 in PARTS && part2(O, M, dims, β)
    end
    close(OUT[])
    return nothing
end

main()
