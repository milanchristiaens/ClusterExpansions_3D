# env_loader_pepo.jl -- the REAL environment of the contraction benchmark (benchmark_contract_onesite.jl, ENV_LOADER), for ANY bond dimensions
# (chi_env, chi_peps, D): a CTMRG3D environment of the Ising PEPO of the PEPO file (stage 1, development/profiling/comparing/ising_model_3D_CE_pepos.jl)
# at β = BETA with D_CE = D, after ENV_ITERS CTMRG3D iterations from a random start (init = :random, so that every bond has its full dimension).
# It is not converged on purpose: the time of the contraction depends on the shapes and the memory layout of the tensors, not on their values, and a
# converged environment would cost an hour or more per set. M = O (the "norm" tensor; an observable-dressed M has the same cost).
#
# The benchmark starts a fresh process for every plan, so the environment is built ONCE per set and saved (JLD2, plain arrays as in the PEPO files)
# in data/contraction_plans/envs/; every later call reads it. The parent process of the benchmark builds the environments of all its sets before
# timing anything. Delete a file in data/contraction_plans/envs/ to build that environment again (for example after a change of the settings below).
#
# It defines onesite_load_env(dims) -> (env, M), env = (; C, E, A) with typed vectors (as in a real CTMRG3D environment); the benchmark calls it.

using TensorKit, TensorOperations, MPSKit, JLD2
using MatrixAlgebraKit: truncrank
using ClusterExpansions

# pepo_file, load_pepos, pack_tensor, unpack_tensor
include(joinpath(@__DIR__, "..", "profiling", "tools", "pepo_files.jl"))

# ---------------- settings ----------------
const ENV_BETA = 0.21            # β of the PEPO (in the PEPO file)
const ENV_DBETA = 0.005          # Δβ, J, g of the PEPO file
const ENV_J = 1.0
const ENV_G = 0.0
const ENV_ITERS = 2              # CTMRG3D iterations from the random start
const ENV_SEED = 1               # seed of the random start
const ENV_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "contraction_plans", "envs"))
# ------------------------------------------

env_cache_file(dims) = joinpath(ENV_DIR, "env_$(dims[1])_$(dims[2])_$(dims[3])_beta$(ENV_BETA)_iters$(ENV_ITERS).jld2")

function onesite_load_env(dims)
    chienv, chipeps, D = Int.(Tuple(dims))
    path = env_cache_file((chienv, chipeps, D))
    if !isfile(path)
        # O[(UP, NORTH, EAST, DOWN, SOUTH, WEST)], all arrows ingoing, as expectation_value gives it to CTMRG3D (see env_loader_example.jl)
        PEPO = only(load_pepos([ENV_BETA], ENV_DBETA, D, ENV_J, ENV_G))
        @tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]
        O = flip(permute(O, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])
        t0 = time()
        env = CTMRG3D(O; trunc_env = truncrank(chienv), trunc_bpeps = truncrank(chipeps), Vbp = ℂ^chipeps, Vv = ℂ^chienv,
                      miniter = ENV_ITERS, maxiter = ENV_ITERS, tol = 0.0, verbosity = 0, init = :random, init_seed = ENV_SEED)
        env, = MPSKit.leading_boundary(env)
        mkpath(ENV_DIR)
        jldsave(path; C = pack_tensor.(env.C), E = pack_tensor.(env.E), A = pack_tensor.(env.A), M = pack_tensor(O))
        println(stderr, "real environment ($chienv, $chipeps, $D): $ENV_ITERS CTMRG3D iterations in $(round(time() - t0; digits = 1)) s, saved to $path")
    end
    f = JLD2.load(path)
    unp(v) = [unpack_tensor(p) for p in v]
    return (; C = unp(f["C"]), E = unp(f["E"]), A = unp(f["A"])), unpack_tensor(f["M"])
end
