# The vbPEPS part of the model comparison (models_3D_CE_bp_compare.jl): for one model of model_pepos.jl (vbpeps = true) we run vbPEPS with the fixed
# environment at its β list (cold start at every β; vbpeps_warm: warm, cold near the βc estimate) and append both observables per β to data/profiling/comparing/models_3D_CE_results.csv.
# The settings of the optimization are those of ising_model_3D_CE_vbPEPS_compare.jl; it runs with BetterPEPSKit (environment development/fast_vbpeps) or PEPSKit (the clone).

# Which environment: development/fast_vbpeps (BetterPEPSKit) or the ClusterExpansions environment of the clone
const IN_FAST_ENV = basename(dirname(Base.active_project())) == "fast_vbpeps"

using TensorKit
using TimerOutputs
using Random
using LinearAlgebra
using JLD2
if IN_FAST_ENV
    include(joinpath(@__DIR__, "..", "..", "fast_vbpeps", "fast_bpeps.jl"))
    using BetterPEPSKit
    const to = FastBpeps.to
else
    using TensorKitTensors
    using MPSKit
    using PEPSKit
    using ClusterExpansions
    const to = ClusterExpansions.to
end

include(joinpath(@__DIR__, "..", "tools", "timing_io.jl"))
include(joinpath(@__DIR__, "..", "tools", "compare_tools.jl"))
include(joinpath(@__DIR__, "..", "..", "initial_pepos", "pepo_files.jl"))
include(joinpath(@__DIR__, "..", "..", "initial_pepos", "model_pepos.jl"))

# ======================================================================================================================
# Settings (a Hortense job passes CE_MODEL, CE_COMPARE_DIMS, CE_COMPARE_CASES, CE_COMPARE_RUN, CE_COMPARE_BETAS, CE_COMPARE_MAXITER, CE_COMPARE_BUDGET_S,
# CE_COMPARE_CORES, CE_COMPARE_IMPL)
# ======================================================================================================================
const MODEL = get(ENV, "CE_MODEL", "tfim_g2")
const MS = model_settings(MODEL)
MS.vbpeps || error("$MODEL has vbpeps = false in model_pepos.jl (its O is not cube symmetric, the C4v CTMRG of vbPEPS needs that)")
# dims = (chienv, chipeps, D_CE); chienv ≥ 0.64 chipeps² as in the Ising runs
const chienv, chipeps, D_CE = compare_dims((16, 5, MS.D_CE))
D_CE == MS.D_CE || error("D_CE = $D_CE of the dims is not the D_CE $(MS.D_CE) of the model $MODEL")
const β_obs = env_override("CE_COMPARE_BETAS", parse_betas, MS.betas)
const CORES = parse(Int, get(ENV, "CE_COMPARE_CORES", "4"))
const RUN_BUDGET_S = env_override("CE_COMPARE_BUDGET_S", s -> parse(Float64, s), 64 * 3600.0 / CORES)   # 64 core hours per case
const MAX_OBS_FACTOR = 1.2
const MAX_SECONDS_OBS = MAX_OBS_FACTOR * RUN_BUDGET_S / length(β_obs)
const MIN_OBS_FACTOR = 0.5
const MIN_SECONDS_OBS = max(60.0, MIN_OBS_FACTOR * RUN_BUDGET_S / length(β_obs))
const WARM_WINDOW = 0.02          # warm rule: cold start when |β - beta_c of the model| < WARM_WINDOW
const SEED_BASE = 1000
const VBPEPS_IMPL = Symbol(get(ENV, "CE_COMPARE_IMPL", IN_FAST_ENV ? "betterpepskit" : "pepskit"))
(VBPEPS_IMPL === :betterpepskit) == IN_FAST_ENV || error("CE_COMPARE_IMPL = $VBPEPS_IMPL does not fit the environment $(Base.active_project())")
const FAST_KWARGS = (; sparse = true, truncation_window = 4, truncation_rtol = 1.0e-3, gradient_maxiter = 20, gradient_krylovdim = 30)
const base_kwargs = (;
    boundary_alg_type = C4vCTMRG, symmetrization = RotateReflect(),
    boundary_tol = 1.0e-8, boundary_maxiter = 300, boundary_verbosity = 1,
    gradient_tol = 1.0e-6, gradient_maxiter = 5, gradient_krylovdim = 20,
    optimizer_memory = 64, optimizer_tol = 1.0e-5,
    optimizer_maxiter = env_override("CE_COMPARE_MAXITER", s -> parse(Int, s), 100),
    maxiter_init = 20,            # must be MAXITER_INIT of models_3D_CE_pepos.jl
    optimizer_verbosity = 2,
    differentiate_environment = false,
)
# (name, warm)
const CASES_ALL = [("vbpeps_warm", true), ("vbpeps_cold", false)]
const CASES = filter(c -> c[1] in compare_cases(["vbpeps_cold"], first.(CASES_ALL)), CASES_ALL)
# ======================================================================================================================

# One observation: ((; cost, numfg, timed_out, normgrad, m = [m1, m2]), state for the next warm start)
function observe(name, β, input, obss, cold, state, allowed, kwargs)
    if VBPEPS_IMPL === :pepskit
        # cold: the seed of the start state of the PEPO file, so we get the same start state as :betterpepskit
        cold && Random.seed!(psi0_seed(β, MS.dbeta))
        r = ClusterExpansions.bpeps_optimize(input, ℂ^chipeps, ℂ^chienv; state0 = cold ? nothing : state, max_seconds = allowed, kwargs...)
        ms = ComplexF64.(ClusterExpansions.bpeps_observables(input, obss, r))
        return (; cost = r.cost, numfg = r.numfg, timed_out = r.timed_out, normgrad = r.normgradhistory === nothing ? NaN : last(r.normgradhistory), m = ms),
               (r.psi, r.env2, r.env3)
    end
    fast_kw = (; boundary_c4v = true, symmetric = true, boundary_tol = kwargs.boundary_tol, boundary_maxiter = kwargs.boundary_maxiter,
               boundary_verbosity = kwargs.boundary_verbosity, gradient_tol = kwargs.gradient_tol, optimizer_memory = kwargs.optimizer_memory,
               optimizer_tol = kwargs.optimizer_tol, optimizer_maxiter = kwargs.optimizer_maxiter, optimizer_verbosity = kwargs.optimizer_verbosity,
               differentiate_environment = false, max_seconds = Float64(allowed), FAST_KWARGS...)
    res = FastBpeps.run_observation(name, input.O, input.M, cold ? input.psi0 : nothing, chipeps, chienv; warm = !cold, seed = rand(1:typemax(Int32)), fast_kw...)
    return res, true
end

function run_case(name, warm, inputs, obss, names, run)
    @info "===== $MODEL, vbPEPS case $name, (χenv, χpeps, D_CE) = ($chienv, $chipeps, $D_CE), budget $RUN_BUDGET_S s ($VBPEPS_IMPL) ====="
    IN_FAST_ENV ? FastBpeps.reset_timer() : reset_timer!(to)
    t_case = time()
    state = nothing
    for (i, β) in enumerate(β_obs)
        Random.seed!(SEED_BASE + i)
        allowed = obs_seconds_allowed(RUN_BUDGET_S, t_case, length(β_obs) - i + 1; min_s = MIN_SECONDS_OBS, max_s = MAX_SECONDS_OBS)
        cold = !warm || state === nothing || !warm_start_at(β, MS.beta_c, WARM_WINDOW)
        t0 = time()
        try
            res, state = observe(name, β, inputs[i], obss, cold, state, allowed, base_kwargs)
            dt = time() - t0
            conv = !res.timed_out && res.normgrad <= base_kwargs.optimizer_tol
            check = (conv ? "ok" : res.timed_out ? "timed out" : "not converged") * (warm && cold ? "; cold start" : "")
            println("OBS $name β = $β: $(names[1]) = $(real(res.m[1])), $(names[2]) = $(real(res.m[2])), f = $(res.cost), calls = $(res.numfg), ",
                    "$(round(dt; digits = 1)) s, $check; ", memory_note())
            append_models_row((MODEL, "vbPEPS", run, name, β, D_CE, chipeps, chienv, names[1], real(res.m[1]), names[2], real(res.m[2]), res.numfg, conv,
                               res.normgrad, res.timed_out, dt, CORES, check))
        catch err
            @error "OBS $name β = $β failed" exception = (err, catch_backtrace())
            append_models_row((MODEL, "vbPEPS", run, name, β, D_CE, chipeps, chienv, names[1], NaN, names[2], NaN, 0, false, NaN, false, time() - t0, CORES,
                               "ERROR: " * first(sprint(showerror, err), 120)))
        end
        GC.gc()
        flush(stdout); flush(stderr)
    end
    return nothing
end

function main()
    BLAS.set_num_threads(1)
    outdir = compare_run_dir(@__FILE__)
    run = basename(outdir)
    open(joinpath(outdir, "settings.txt"), "a") do io
        println(io, "MODEL = $MODEL ($MS), vbPEPS $VBPEPS_IMPL, dims = ($chienv, $chipeps, $D_CE), β = $β_obs, cases = $(first.(CASES)), budget $RUN_BUDGET_S s ",
                "on $CORES cores, threads = $(Threads.nthreads()), WARM_WINDOW = $WARM_WINDOW, base_kwargs = $base_kwargs, ",
                IN_FAST_ENV ? "FAST_KWARGS = $FAST_KWARGS, " : "", "Julia $(VERSION), host $(gethostname())")
    end
    names = model_observable_names(MS)
    # the observables: in the packed M of the PEPO file (:betterpepskit) or as operators (:pepskit)
    obss = IN_FAST_ENV ? nothing : model_observables(MS)
    inputs = IN_FAST_ENV ? load_model_bpeps_tensors(MODEL, β_obs, chipeps; maxiter_init = base_kwargs.maxiter_init) : load_model_pepos(MODEL, β_obs)
    @info "$MODEL: PEPOs of $(model_pepo_file(MODEL)), run $run, results $MODELS_RESULTS_PATH"
    # warm-up (compilation): a cold and a warm call with tiny settings
    tiny = (; base_kwargs..., optimizer_maxiter = 1, maxiter_init = 2, boundary_maxiter = 100)
    _, st = observe("warmup", β_obs[1], inputs[1], obss, true, nothing, Inf, tiny)
    observe("warmup", β_obs[1], inputs[1], obss, false, st, Inf, tiny)
    for (name, warm) in CASES
        run_case(name, warm, inputs, obss, names, run)
    end
    return nothing
end

main()
