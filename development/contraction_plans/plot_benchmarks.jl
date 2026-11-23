# plot_benchmarks.jl -- the figures of the contraction benchmark (benchmark_contract_onesite.jl) again, from the data only (23 Nov):
#   1. per run folder data/contraction_plans/benchmark_contract_onesite/<run>/: every plot of BENCH_PLOTS (tools/benchmark_plots.jl), in that folder
#      (benchmark_times_reuse.png, benchmark_times_manual.png, benchmark_allocators.png, benchmark_ram.png and the new benchmark_slices.png);
#   2. ONE figure of the time against the number of slices with the plans of ALL selected run folders together (all dimension sets, plans for RAM_GIB),
#      data/contraction_plans/benchmark_contract_onesite/all_runs_slices_<MACHINE>.png, and on screen per set the slope of log(time) against log(slices).
#      A plan that was timed in several runs (same dimensions, tree, cuts and budget) counts once, with its fastest time.
# Nothing is timed (unlike RUN_DIR in benchmark_contract_onesite.jl, which also times the plans that a run did not finish).
# Times of different machines must not be mixed: we only take the run folders whose settings.txt has MACHINE = MACHINE below.
# Run in the ClusterExpansions environment (Plots from the default environment), after copying the run folders from Hortense (Hortense guide, section 21):
#   PowerShell, clone folder:  julia +1.12 --project=. development/contraction_plans/plot_benchmarks.jl

ENV["GKSwstype"] = "nul"     # no plot window: the plots are only saved to a file
using Printf
include(joinpath(@__DIR__, "tools", "benchmark_plots.jl"))   # read_bench_results, make_all_benchmark_plots, make_slices_figure, slices_slopes

# ======================================================================================================================
# Settings
# ======================================================================================================================
const RUNS = nothing          # nothing = every run folder; or a list of names, for example ["20261121_0230", "20261121_1700"]
const MACHINE = "hortense"    # only run folders measured on this machine ("hortense" or "laptop"; the line MACHINE = ... of their settings.txt)
const RAM_GIB = 3             # the RAM budget of the plans in the combined figure (3 GiB: the budget of the compare runs)
const ALLOCATOR = "reuse"     # the allocator of the plans in the combined figure (reuse: the default of contract_onesite_auto)
const ENVIRONMENT = "mock"    # "mock" or "real": the environment of the plans in the combined figure (beginning of the column env of results.csv)
const PER_RUN = true          # false = only the combined figure
# ======================================================================================================================

const BASE = joinpath(BENCH_DATA_DIR, "benchmark_contract_onesite")

# The machine of a run folder, from its settings.txt ("" if there is none)
function run_machine(dir)
    f = joinpath(dir, "settings.txt")
    isfile(f) || return ""
    m = match(r"MACHINE = (\w+)", read(f, String))
    return m === nothing ? "" : String(m.captures[1])
end

function main()
    isdir(BASE) || error("no folder $BASE: copy the run folders of the benchmark first")
    names = RUNS === nothing ? sort(filter(d -> isfile(joinpath(BASE, d, "results.csv")), readdir(BASE))) : RUNS
    pooled = NamedTuple[]
    for name in names
        dir = joinpath(BASE, name)
        mach = run_machine(dir)
        if mach != MACHINE
            @info "run $name: machine \"$mach\", not \"$MACHINE\": skipped"
            continue
        end
        rows = read_bench_results(joinpath(dir, "results.csv"))
        @info "run $name: $(count(r -> isfinite(r.tmin), rows)) timed plans"
        PER_RUN && make_all_benchmark_plots(rows, dir)
        append!(pooled, [r for r in rows if isfinite(r.tmin) && r.alloc == ALLOCATOR && startswith(r.env, ENVIRONMENT) && r.ram_gib == RAM_GIB])
    end
    isempty(pooled) && return @warn "no timed plan for the combined figure (MACHINE = $MACHINE, RAM_GIB = $RAM_GIB, ALLOCATOR = $ALLOCATOR, ENVIRONMENT = $ENVIRONMENT)"

    # one row per plan: the fastest of the runs that timed it
    best = Dict{Any, NamedTuple}()
    for r in pooled
        k = plan_id(r)
        (!haskey(best, k) || r.tmin < best[k].tmin) && (best[k] = r)
    end
    rows = collect(values(best))
    fig = make_slices_figure(rows; ram = RAM_GIB)
    out = joinpath(BASE, "all_runs_slices_$(MACHINE).png")
    savefig(fig, out)
    @info "Plot saved: $out ($(length(rows)) plans of $(length(unique(bench_dims.(rows)))) dimension sets)"

    println("\nslope of log(time) against log(slices) per dimension set (plans for $RAM_GIB GiB, $ALLOCATOR, $ENVIRONMENT, $MACHINE):")
    println(rpad("(χenv, χpeps, D)", 18), rpad("plans", 7), rpad("slices", 16), "slope")
    for s in slices_slopes(rows; ram = RAM_GIB)
        println(rpad("($(dims_label(s.dims)))", 18), rpad(s.n, 7), rpad("$(Int(s.slices[1]))-$(Int(s.slices[2]))", 16), @sprintf("%.2f", s.slope))
    end
    return nothing
end

main()
