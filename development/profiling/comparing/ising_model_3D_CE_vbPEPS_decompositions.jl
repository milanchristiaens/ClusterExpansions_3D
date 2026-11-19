# Dense against block-Krylov decompositions in the CTMRG of the fast vbPEPS (BetterPEPSKit), for a broad grid of bond dimensions: for which
# (χenv, χpeps, D_CE) is the dense eigh of the enlarged corner faster than the block-Krylov decomposition? The answer chooses FAST_KWARGS.sparse of
# ising_model_3D_CE_vbPEPS_compare.jl.
#
# Only the CTMRG is timed, not the whole vbPEPS (no LBFGS, no gradient). For every set (χenv, χpeps, D_CE) of the grid (CHIENVS × CHIPEPS × D_CES):
#   ψ = the start state of the vbPEPS for χpeps (power method, from the PEPO file of stage 1 at β = BETA, symmetrized and normalized as in the vbPEPS),
#   O = the 2D PEPO tensor for D_CE (same file); the two networks of the vbPEPS: ⟨ψ|ψ⟩ (bond χpeps²) and ⟨ψ|O|ψ⟩ (bond χpeps²·D_CE).
#   For each network: one start environment (grown from a product state, with the dense decomposition), the same for both methods. Per method
#   (dense = sparse false, block-Krylov = sparse true; the C4v CTMRG and gap cut of the vbPEPS compare): WARM_STEPS CTMRG steps that are not timed,
#   then TIMED_STEPS steps in one leading_boundary call (as one cost-function call of the vbPEPS does), timed: t_step = time / TIMED_STEPS.
#   The contractions of a step are the same for both methods, so the difference is the decomposition. n = χenv · bond is the size of the enlarged
#   corner that is decomposed (dense: cost ∝ n³; block-Krylov: only the χenv largest values, warm-started from the previous step).
#
# Every set is appended to results.csv of a new run folder data/profiling/comparing/ising_model_3D_CE_vbPEPS_decompositions/<date>_<time>/
# immediately, so a crash halfway keeps the earlier sets; the run folder also gets settings.txt and (made afterwards with REPLOT_DIR, see below) the plot
# decompositions_times.png: time per CTMRG step against the sets (sorted by n), left ⟨ψ|ψ⟩, right ⟨ψ|O|ψ⟩, one colour per method. The summary
# prints per network and set the faster method and from which n on block-Krylov is always faster.
#
# Two environments (as the vbPEPS compare):
#   the runs:  the BetterPEPSKit environment (see development/fast_vbpeps/README.md), which has no Plots; at the end we print the summary:
#                  julia -t 4 --project=development/fast_vbpeps development/profiling/comparing/ising_model_3D_CE_vbPEPS_decompositions.jl
#   the plot:  REPLOT_DIR = the run folder, in the ClusterExpansions environment:
#                  julia --project=. development/profiling/comparing/ising_model_3D_CE_vbPEPS_decompositions.jl
# The PEPO file (ising_model_3D_CE_pepos.jl) must have BETA, the D_CE of D_CES and start states for the χpeps of CHIPEPS.
# PROFILE (see the settings) chooses the threads: :laptop (julia -t 4, as the vbPEPS compare on the laptop) or :hpc (Hortense, julia -t 2 on 2 cores,
# as the full vbPEPS run there; development/hpc/job_decompositions.pbs sets CE_DECOMP_PROFILE), so that the result holds for the compare run on that
# machine. On Hortense this benchmark runs after the dimension pilot and before the timing (see the Hortense guide).
# Compilation happens once per process, in a warm-up with the first set (not measured).

ENV["GKSwstype"] = "nul"     # no plot window: the plots are only saved to a file

# Which environment: development/fast_vbpeps (the runs) or the ClusterExpansions environment of the clone (the plot)
const IN_FAST_ENV = basename(dirname(Base.active_project())) == "fast_vbpeps"

using TensorKit
using TimerOutputs
using Random
using LinearAlgebra
using JLD2
if IN_FAST_ENV
    include(joinpath(@__DIR__, "..", "..", "fast_vbpeps", "fast_bpeps.jl"))    # module FastBpeps (bpeps_initial_environment)
    using BetterPEPSKit
else
    using Plots
    using Plots.PlotMeasures
end

# The helper functions of this script, in ../tools/:
#   timing_io.jl:   new_run_dir, existing_run_dir
#   pepo_files.jl:  load_bpeps_tensors, unpack_tensor (the PEPO file of stage 1)
include(joinpath(@__DIR__, "..", "tools", "timing_io.jl"))
include(joinpath(@__DIR__, "..", "tools", "pepo_files.jl"))

# ======================================================================================================================
# Settings
# ======================================================================================================================
# Profile: :laptop (default) or :hpc (Hortense; set by development/hpc/job_decompositions.pbs): the threads of the vbPEPS compare on that machine
const PROFILE = Symbol(get(ENV, "CE_DECOMP_PROFILE", "laptop"))
const PROFILES = Dict(:laptop => (; julia_threads = 4), :hpc => (; julia_threads = 2))
haskey(PROFILES, PROFILE) || error("unknown profile :$PROFILE (CE_DECOMP_PROFILE): use laptop or hpc")
const JULIA_THREADS = PROFILES[PROFILE].julia_threads
const BLAS_THREADS = 1
BLAS.set_num_threads(BLAS_THREADS)
Threads.nthreads() == JULIA_THREADS || @warn "This benchmark expects julia -t $JULIA_THREADS (as the vbPEPS compare), but Threads.nthreads() = $(Threads.nthreads())"

# The grid: every combination of these (the x axis of the plot is sorted by n = χenv · bond of the network)
const CHIENVS = [8, 16, 24, 32, 48]
const CHIPEPS = [2, 3, 4]                # needs start states for these χpeps in the PEPO file (CHIPEPS of ising_model_3D_CE_pepos.jl)
const D_CES = [2, 3]
const BETA = 0.21                        # one β near βc ≈ 0.2217 (as the pilot of the vbPEPS compare)
const WARM_STEPS = 3                     # CTMRG steps before the timing (not timed)
const TIMED_STEPS = 20                   # CTMRG steps in the timed leading_boundary call

const J = 1.0
const g = 0.0
const Δβ_iter = 0.005
const MAXITER_INIT = 20                  # power-method steps of the start states in the PEPO file (MAXITER_INIT of ising_model_3D_CE_pepos.jl)
const TRUNCATION_WINDOW = 4              # the gap cut of the vbPEPS compare (FAST_KWARGS there)
const TRUNCATION_RTOL = 1.0e-3

# The two methods: name, value of sparse, colour in the plot; the two networks: name, marker in the plot
const METHODS = [("dense", false, :darkorange), ("block-Krylov", true, :royalblue)]
const NETWORKS = [("<psi|psi>", :circle), ("<psi|O|psi>", :square)]

# If REPLOT_DIR is the NAME of a run folder (for example "20261118_1800") in data/profiling/comparing/ising_model_3D_CE_vbPEPS_decompositions/,
# we only redo the summary and the plot from its results.csv (in the ClusterExpansions environment). Leave it "" for a normal run.
const REPLOT_DIR = ""

# ======================================================================================================================
# results.csv of a run: one row per (set, network, method)
# ======================================================================================================================
const DECOMP_HEADER = "chi_env,chi_peps,D_CE,beta,network,method,n,steps,t_step_s,t_call_s,check"

function append_decomp_row(path, r)
    open(path, "a") do io
        println(io, join((r.chienv, r.chipeps, r.D_CE, r.β, r.network, r.method, r.n, r.steps, r.t_step, r.t_call, replace(r.check, "," => ";", "\n" => " ")), ","))
    end
    return nothing
end

function read_decomp_results(path)
    rows = NamedTuple[]
    for line in readlines(path)[2:end]
        isempty(strip(line)) && continue
        c = split(line, ","; limit = 11)
        push!(rows, (; chienv = parse(Int, c[1]), chipeps = parse(Int, c[2]), D_CE = parse(Int, c[3]), β = parse(Float64, c[4]), network = String(c[5]),
                     method = String(c[6]), n = parse(Int, c[7]), steps = parse(Int, c[8]), t_step = parse(Float64, c[9]), t_call = parse(Float64, c[10]),
                     check = String(c[11])))
    end
    return rows
end

# ======================================================================================================================
# The CTMRG of one network with one method
# ======================================================================================================================
# The C4v CTMRG of the vbPEPS compare (gap cut; sparse: block-Krylov decompositions) with exactly `steps` steps (tol = 0, miniter = maxiter)
function ctmrg_alg(χenv, sparse, steps)
    trunc = truncgap(χenv; window = TRUNCATION_WINDOW, margin = sparse ? 1 : 0, rtol = TRUNCATION_RTOL)
    decomposition = sparse ? (; decomposition_alg = (; alg = :BlockKrylov)) : (;)
    return C4vCTMRG(; tol = 0.0, maxiter = steps, miniter = steps, verbosity = 0, trunc, decomposition...)
end

# WARM_STEPS steps (not timed), then TIMED_STEPS steps in one timed call; returns (t_step, t_call, check)
function time_ctmrg(network, env0, χenv, sparse)
    env, = leading_boundary(deepcopy(env0), network, ctmrg_alg(χenv, sparse, WARM_STEPS))
    alg = ctmrg_alg(χenv, sparse, TIMED_STEPS)
    t0 = time()
    leading_boundary(env, network, alg)
    t_call = time() - t0
    return t_call / TIMED_STEPS, t_call, "ok"
end

# ======================================================================================================================
# Summary and plot
# ======================================================================================================================
dims_of(r) = (r.chienv, r.chipeps, r.D_CE)
dims_label(d) = "($(d[1]), $(d[2]), $(d[3]))"
fmt(x) = isfinite(x) ? string(round(x; sigdigits = 3)) : "-"

# Per network: per set the time per step of both methods and the faster one, and from which n on block-Krylov is always faster
function print_summary(rows)
    for (network, _) in NETWORKS
        sub = filter(r -> r.network == network, rows)
        isempty(sub) && continue
        println("\n===== $network: time per CTMRG step (s), sets sorted by n = χenv · bond =====")
        println(rpad("(χenv, χpeps, D_CE)", 21), rpad("n", 7), rpad("dense", 12), rpad("block-Krylov", 14), "faster")
        sets = sort(unique(dims_of.(sub)); by = d -> (first(r.n for r in sub if dims_of(r) == d), d))
        bk_wins = Tuple{Int, Bool}[]
        for d in sets
            a = filter(r -> dims_of(r) == d && r.method == "dense", sub)
            b = filter(r -> dims_of(r) == d && r.method == "block-Krylov", sub)
            (isempty(a) || isempty(b)) && continue
            a, b = a[1], b[1]
            ok = isfinite(a.t_step) && isfinite(b.t_step)
            faster = !ok ? "-" : a.t_step < b.t_step ? "dense ×$(round(b.t_step / a.t_step; digits = 2))" : "block-Krylov ×$(round(a.t_step / b.t_step; digits = 2))"
            ok && push!(bk_wins, (a.n, b.t_step < a.t_step))
            println(rpad(dims_label(d), 21), rpad(a.n, 7), rpad(fmt(a.t_step), 12), rpad(fmt(b.t_step), 14), faster)
        end
        # the smallest n from which on block-Krylov wins at every larger n
        sort!(bk_wins)
        i = findlast(w -> !w[2], bk_wins)
        cross = i === nothing ? "at every measured n" : i == length(bk_wins) ? "never (dense is faster at the largest measured n)" : "from n = $(bk_wins[i + 1][1]) on"
        println("block-Krylov is faster at every larger n: ", cross)
    end
    return nothing
end

# Ticks of a logarithmic time axis at 1, 2 and 5 times a power of 10, and the limits, for the times ys (s)
function time_axis(ys)
    lo, hi = extrema(ys)
    ticks = filter(t -> lo / 1.5 <= t <= hi * 1.5, [m * 10.0^k for k in floor(Int, log10(lo)):ceil(Int, log10(hi)) for m in (1, 2, 5)])
    labels = [t >= 1 ? string(round(Int, t)) : string(round(t; sigdigits = 1)) for t in ticks]
    return (; ylims = (lo / 1.5, hi * 1.5), yticks = (ticks, labels))
end

# One panel per network: time per step against the sets sorted by n, one colour per method
function plot_network(rows, network, marker)
    sub = filter(r -> r.network == network && isfinite(r.t_step) && r.t_step > 0, rows)
    isempty(sub) && return plot(; title = "$network: no data")
    sets = sort(unique(dims_of.(sub)); by = d -> (first(r.n for r in sub if dims_of(r) == d), d))
    xpos = Dict(d => i for (i, d) in enumerate(sets))
    nset = length(sets)
    labels = ["$(dims_label(d)) n=$(first(r.n for r in sub if dims_of(r) == d))" for d in sets]
    p = plot(; xlabel = "(χenv, χpeps, D_CE), n = χenv · bond (sorted by n)", ylabel = "time per CTMRG step [s]", title = network, yscale = :log10,
             legend = :topleft, xticks = (1:nset, labels), xrotation = 60, xlims = (0.5, nset + 0.5), minorgrid = true, gridalpha = 0.25, minorgridalpha = 0.12,
             time_axis([r.t_step for r in sub])...)
    for (k, (method, _, colour)) in enumerate(METHODS)
        pts = sort(filter(r -> r.method == method, sub); by = r -> xpos[dims_of(r)])
        isempty(pts) && continue
        plot!(p, [xpos[dims_of(r)] + (k - 1.5) * 0.15 for r in pts], [r.t_step for r in pts]; color = colour, marker, markersize = 5, markerstrokewidth = 0,
              linealpha = 0.4, label = method)
    end
    return p
end

function make_plot(rows, outdir)
    plt = plot([plot_network(rows, network, marker) for (network, marker) in NETWORKS]...; layout = (1, 2), size = (1800, 650), margin = 6mm,
               bottom_margin = 22mm, left_margin = 10mm,
               plot_title = "fast vbPEPS: dense against block-Krylov per CTMRG step, β = $(rows[1].β), g = $g, C4v CTMRG, $(rows[1].steps) timed steps per set")
    file = joinpath(outdir, "decompositions_times.png")
    savefig(plt, file)
    @info "Plot saved: $file"
    return nothing
end

# ======================================================================================================================
# Main
# ======================================================================================================================
function main()
    if !isempty(REPLOT_DIR)
        IN_FAST_ENV && error("REPLOT_DIR needs Plots: run this script in the ClusterExpansions environment (julia --project=. <this file>)")
        outdir = existing_run_dir(@__FILE__, REPLOT_DIR)
        rows = read_decomp_results(joinpath(outdir, "results.csv"))
        print_summary(rows)
        make_plot(rows, outdir)
        return nothing
    end

    IN_FAST_ENV || error("The runs need the BetterPEPSKit environment: julia -t 4 --project=development/fast_vbpeps <this file> (the ClusterExpansions " *
                         "environment only makes the plot, with REPLOT_DIR)")
    t_run = time()
    outdir = new_run_dir(@__FILE__)
    resultspath = joinpath(outdir, "results.csv")
    write(resultspath, DECOMP_HEADER * "\n")
    open(joinpath(outdir, "settings.txt"), "w") do io
        println(io, "PROFILE = $PROFILE, Julia $(VERSION), threads = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads()), host = $(gethostname())")
        println(io, "CHIENVS = $CHIENVS, CHIPEPS = $CHIPEPS, D_CES = $D_CES, BETA = $BETA, WARM_STEPS = $WARM_STEPS, TIMED_STEPS = $TIMED_STEPS")
        println(io, "J = $J, g = $g, Δβ_iter = $Δβ_iter, MAXITER_INIT = $MAXITER_INIT, TRUNCATION_WINDOW = $TRUNCATION_WINDOW, TRUNCATION_RTOL = $TRUNCATION_RTOL")
    end
    @info "Output directory: $outdir"

    # The two networks of the vbPEPS for (χpeps, D_CE): ψ = the start state of the PEPO file (symmetrized and normalized as in FastBpeps.bpeps_optimize)
    function networks(chipeps, D_CE)
        inp = only(load_bpeps_tensors([BETA], Δβ_iter, D_CE, J, g, chipeps; maxiter_init = MAXITER_INIT))
        ψ = BetterPEPSKit.peps_normalize(symmetrize!(InfinitePEPS(unpack_tensor(inp.psi0)), RotateReflect()))
        T = InfinitePEPO(unpack_tensor(inp.O))
        return [InfiniteSquareNetwork(ψ), InfiniteSquareNetwork(ψ, T)], [chipeps^2, chipeps^2 * D_CE]
    end
    # One set: both networks, both methods; rows appended to results.csv
    function run_set(chienv, chipeps, D_CE; record = true)
        nets, bonds = networks(chipeps, D_CE)
        for (k, (network, _)) in enumerate(NETWORKS)
            env0 = FastBpeps.bpeps_initial_environment(ctmrg_alg(chienv, false, 1), nets[k], chienv; window = TRUNCATION_WINDOW)
            for (method, sparse, _) in METHODS
                r = try
                    t_step, t_call, check = time_ctmrg(nets[k], env0, chienv, sparse)
                    (; t_step, t_call, check)
                catch err
                    @error "($chienv, $chipeps, $D_CE) $network $method failed" exception = (err, catch_backtrace())
                    (; t_step = NaN, t_call = NaN, check = "ERROR: " * first(sprint(showerror, err), 150))
                end
                record || continue
                row = (; chienv, chipeps, D_CE, β = BETA, network, method, n = chienv * bonds[k], steps = TIMED_STEPS, r...)
                append_decomp_row(resultspath, row)
                println("SET ($chienv, $chipeps, $D_CE) $network $method: n = $(row.n), t_step = $(fmt(r.t_step)) s, $(r.check)")
                flush(stdout); flush(stderr)
            end
        end
        return nothing
    end

    # Warm-up: compile both methods and both networks with the smallest set (not measured)
    t0 = time()
    run_set(minimum(CHIENVS), minimum(CHIPEPS), minimum(D_CES); record = false)
    @info "Warm-up took $(round(time() - t0; digits = 1)) s"

    for D_CE in D_CES, chipeps in CHIPEPS, chienv in CHIENVS
        run_set(chienv, chipeps, D_CE)
    end

    rows = read_decomp_results(resultspath)
    print_summary(rows)
    @info "Total time: $(round((time() - t_run) / 60; digits = 1)) min, peak memory of the process $(round(Sys.maxrss() / 2^30; digits = 2)) GiB"
    @info "The plot: set REPLOT_DIR = \"$(basename(outdir))\" and run this script in the ClusterExpansions environment (julia --project=. <this file>)"
    return nothing
end

main()
