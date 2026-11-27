# Does CTMRG3D :svd give the belief propagation (BP) answer also for the transverse-field Ising model at g ≠ 0 and for the Bose-Hubbard model, and do the octant
# variants do better? For one model of model_pepos.jl we run BP and the cases below at its β list (warm, cold near βc) and append both observables per β
# to data/profiling/comparing/models_3D_CE_results.csv.

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using LinearAlgebra
using Printf
using JLD2
using TimerOutputs
const CE = ClusterExpansions
const to = CE.to

include(joinpath(@__DIR__, "..", "tools", "timing_io.jl"))
include(joinpath(@__DIR__, "..", "tools", "compare_tools.jl"))
include(joinpath(@__DIR__, "..", "..", "initial_pepos", "pepo_files.jl"))
include(joinpath(@__DIR__, "..", "..", "initial_pepos", "model_pepos.jl"))

# ======================================================================================================================
# Settings (a Hortense job passes CE_MODEL, CE_COMPARE_DIMS, CE_COMPARE_CASES, CE_COMPARE_RUN, CE_COMPARE_BETAS, CE_COMPARE_MAXITER, CE_COMPARE_BUDGET_S, CE_COMPARE_CORES, CE_COMPARE_RAM)
# ======================================================================================================================
const MODEL = get(ENV, "CE_MODEL", "tfim_g2")
const M = model_settings(MODEL)
const chienv, chipeps, D_CE = compare_dims((8, 4, M.D_CE))
D_CE == M.D_CE || error("D_CE = $D_CE of the dims is not the D_CE $(M.D_CE) of the model $MODEL")
const β_obs = env_override("CE_COMPARE_BETAS", parse_betas, M.betas)
const CORES = parse(Int, get(ENV, "CE_COMPARE_CORES", "4"))
const RUN_BUDGET_S = env_override("CE_COMPARE_BUDGET_S", s -> parse(Float64, s), 64 * 3600.0 / CORES)   # 64 core hours per case
const MAX_OBS_FACTOR = 1.2
const MAX_SECONDS_OBS = MAX_OBS_FACTOR * RUN_BUDGET_S / length(β_obs)
const MIN_SECONDS_OBS = 60.0
const MAXITER = env_override("CE_COMPARE_MAXITER", s -> parse(Int, s), 300)
const MINITER = 3
const TOL = 1e-7
const INIT = :random
const INIT_SEED = 1
const WARM_WINDOW = 0.02          # warm rule: cold start when |β - beta_c of the model| < WARM_WINDOW
const STAGNATION_WINDOW = 20      # stagnation stop of the :svd P = 0 cases (they may not converge)
const RAM_GIB = env_override("CE_COMPARE_RAM", s -> parse(Float64, s), 5.0)   # RAM budget (GiB) of the sliced contraction: TREE_CURRENT with the fewest cuts
const BP_TOL = 1e-12
const BP_MAXITER = 50_000
# (name, ctm_alg kwargs); bp is computed for every β of every job (cheap)
const CASES_ALL = [
    ("svd_P05", (; ctm_alg = CTMRG3D, projectors = :svd, projector_exponent = 0.5, class_projectors = false)),
    ("svd_P0", (; ctm_alg = CTMRG3D, projectors = :svd, projector_exponent = 0.0, class_projectors = false, stagnation_window = STAGNATION_WINDOW)),
    ("svdclass_P0", (; ctm_alg = CTMRG3D, projectors = :svd, projector_exponent = 0.0, class_projectors = true, stagnation_window = STAGNATION_WINDOW)),
    ("octantunweighted", (; ctm_alg = CTMRG3DOctant, face_projector = :unweighted)),
    ("octantsigmaweighted", (; ctm_alg = CTMRG3DOctant, face_projector = :sigma_weighted)),
    ("octantleadweighted", (; ctm_alg = CTMRG3DOctant, face_projector = :lead_weighted)),
]
const CASES = filter(c -> c[1] in compare_cases([c[1] for c in CASES_ALL], [c[1] for c in CASES_ALL]), CASES_ALL)
# ======================================================================================================================

function O_and_Ms(PEPO, obss)
    @tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]
    O = flip(permute(O, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])
    Ms = map(obss) do obs
        @tensor Mo[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 2; -1 -2 -3 -4 -5 -6] * obs[2; 1]
        flip(permute(Mo, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])
    end
    return O, Ms
end

# BP at every β from a random start (a warm start stays in the symmetric fixed point)
function run_bp(Os, obss, names, run)
    for (i, β) in enumerate(β_obs)
        t = @elapsed (vals, _, ϵs) = bp_expectation_value(Os[i], obss; init = :random, init_seed = INIT_SEED, maxiter = BP_MAXITER, tol = BP_TOL, return_messages = true)
        conv = ϵs[end] < BP_TOL
        println("OBS bp β = $β: $(names[1]) = $(real(vals[1])), $(names[2]) = $(real(vals[2])), $(length(ϵs)) iterations, $(conv ? "converged" : "not converged")")
        append_models_row((MODEL, "BP", run, "bp", β, D_CE, 0, 0, names[1], real(vals[1]), names[2], real(vals[2]), length(ϵs), conv, ϵs[end], false, t, 1, conv ? "ok" : "not converged"))
    end
    return nothing
end

function run_case(name, kw, Os, obss, names, run)
    @info "===== $MODEL, case $name, (χenv, χpeps, D_CE) = ($chienv, $chipeps, $D_CE), budget $RUN_BUDGET_S s ====="
    t_case = time()
    env_prev = nothing
    for (i, β) in enumerate(β_obs)
        allowed = obs_seconds_allowed(RUN_BUDGET_S, t_case, length(β_obs) - i + 1; min_s = MIN_SECONDS_OBS, max_s = MAX_SECONDS_OBS)
        warm = env_prev !== nothing && warm_start_at(β, M.beta_c, WARM_WINDOW)
        t0 = time()
        try
            # the environment only (no observable), then O and both M contracted once each
            _, env, ϵs, timed_out = MPSKit.expectation_value(Os[i], [], ℂ^chipeps, ℂ^chienv; return_env = true, env0 = warm ? env_prev : nothing,
                                                              miniter = MINITER, maxiter = MAXITER, tol = TOL, init = INIT, init_seed = INIT_SEED,
                                                              max_seconds = allowed, ram_gib = RAM_GIB, onesite_kwargs = (; machine = "hortense", plans = nothing), kw...)
            O, Ms = O_and_Ms(Os[i], obss)
            kwc = (; machine = "hortense", plans = nothing)
            den = CE.contract_onesite_dispatch(env, O, RAM_GIB; kwc...)
            vals = [CE.contract_onesite_dispatch(env, Mo, RAM_GIB; kwc...) / den for Mo in Ms]
            conv = ϵs[end] < TOL
            check = (conv ? "ok" : timed_out ? "timed out" : length(ϵs) < MAXITER ? "stopped (no progress)" : "not converged") * (warm ? "" : "; cold start")
            dt = time() - t0
            println("OBS $name β = $β: $(names[1]) = $(real(vals[1])), $(names[2]) = $(real(vals[2])), $(length(ϵs)) iterations, ϵ = $(ϵs[end]), ",
                    "$(round(dt; digits = 1)) s, $check; ", memory_note())
            append_models_row((MODEL, "CTMRG3D", run, name, β, D_CE, chipeps, chienv, names[1], real(vals[1]), names[2], real(vals[2]), length(ϵs), conv, ϵs[end],
                        timed_out, dt, CORES, check))
            env_prev = env
        catch err
            @error "OBS $name β = $β failed" exception = (err, catch_backtrace())
            append_models_row((MODEL, "CTMRG3D", run, name, β, D_CE, chipeps, chienv, names[1], NaN, names[2], NaN, 0, false, NaN, false, time() - t0, CORES,
                        "ERROR: " * first(sprint(showerror, err), 120)))
        end
        GC.gc()
        flush(stdout); flush(stderr)
    end
    return nothing
end

function main()
    BLAS.set_num_threads(CORES)
    outdir = compare_run_dir(@__FILE__)
    run = basename(outdir)
    open(joinpath(outdir, "settings.txt"), "a") do io
        println(io, "MODEL = $MODEL ($M), dims = ($chienv, $chipeps, $D_CE), β = $β_obs, cases = $([c[1] for c in CASES]), budget $RUN_BUDGET_S s on $CORES cores, ",
                "MAXITER = $MAXITER, TOL = $TOL, INIT = $INIT, WARM_WINDOW = $WARM_WINDOW, RAM_GIB = $RAM_GIB, Julia $(VERSION), host $(gethostname())")
    end
    obss = model_observables(M)
    names = model_observable_names(M)
    Os = load_model_pepos(MODEL, β_obs)
    @info "$MODEL: PEPOs of $(model_pepo_file(MODEL)), run $run, results $MODELS_RESULTS_PATH"
    # warm-up (compilation) at tiny dimensions
    for (_, kw) in CASES
        MPSKit.expectation_value(Os[1], [], ℂ^2, ℂ^4; return_env = true, miniter = 0, maxiter = 1, init = INIT, kw...)
    end
    run_bp(Os, obss, names, run)
    for (name, kw) in CASES
        run_case(name, kw, Os, obss, names, run)
    end
    return nothing
end

main()
