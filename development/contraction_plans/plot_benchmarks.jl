# The figures of the contraction benchmark (benchmark_contract_onesite.jl) made again from the data only: every plot of BENCH_PLOTS per run folder, and figures with the plans of all selected run folders together.
# Nothing is timed, and times of different machines are not mixed (only run folders whose settings.txt has MACHINE = MACHINE below).

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
    everything = NamedTuple[]     # all timed rows of the selected run folders (every allocator, environment and budget), for part 3
    nruns = 0                     # number of run folders of this machine
    for name in names
        dir = joinpath(BASE, name)
        mach = run_machine(dir)
        if mach != MACHINE
            @info "run $name: machine \"$mach\", not \"$MACHINE\": skipped"
            continue
        end
        rows = read_bench_results(joinpath(dir, "results.csv"))
        nruns += 1
        @info "run $name: $(count(r -> isfinite(r.tmin), rows)) timed plans"
        PER_RUN && make_all_benchmark_plots(rows, dir)
        append!(pooled, [r for r in rows if isfinite(r.tmin) && r.alloc == ALLOCATOR && startswith(r.env, ENVIRONMENT) && r.ram_gib == RAM_GIB])
        append!(everything, [r for r in rows if isfinite(r.tmin)])
    end
    # part 3: one row per (plan, allocator, environment kind), the fastest of the runs that timed it
    envkind(r) = startswith(r.env, "mock") ? "mock" : "real"
    best3 = Dict{Any, NamedTuple}()
    for r in everything
        k = (plan_id(r), r.alloc, envkind(r))
        (!haskey(best3, k) || r.tmin < best3[k].tmin) && (best3[k] = r)
    end
    rows3 = collect(values(best3))
    for (file, fig) in (("all_runs_allocators_$(MACHINE).png", () -> make_allocator_figure([r for r in rows3 if envkind(r) == ENVIRONMENT])),
                        ("all_runs_mock_vs_real_$(MACHINE).png", () -> make_env_figure(rows3)))
        try
            plt = fig()
            plt === nothing ? @info("$file not made: no plan was timed both ways") : (savefig(plt, joinpath(BASE, file)); @info "Plot saved: $(joinpath(BASE, file))")
        catch err
            @error "$file failed" exception = (err, catch_backtrace())
        end
    end
    # the times figure of the plans for RAM_GIB with ALLOCATOR and ENVIRONMENT, with all repetitions of all runs
    file = "all_runs_times_$(ALLOCATOR)_$(MACHINE).png"
    try
        sub = pool_repetitions(rows_of([r for r in everything if r.ram_gib == RAM_GIB && startswith(r.env, ENVIRONMENT)], ALLOCATOR))
        if isempty(sub)
            @info "$file not made: no plan with allocator $ALLOCATOR"
        else
            savefig(make_times_figure(sub; nruns), joinpath(BASE, file))
            @info "Plot saved: $(joinpath(BASE, file)) ($(length(sub)) plans)"
        end
    catch err
        @error "$file failed" exception = (err, catch_backtrace())
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