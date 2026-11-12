# This file times the plans of the plan file src/utility/onesite_plans.csv (written by development/contraction_plans/find_orders_ome.jl): the sliced contraction
# `contract_onesite` with a given contraction tree and given cut bonds.
# Each plan runs in a FRESH child julia process, so that `Sys.maxrss()` (peak resident memory) belongs to that plan only and no allocator pool survives.
#
# USAGE (PowerShell, nothing else running; -O1 keeps the just-in-time compilation of the 27-tensor kernels shorter than the default -O2, -t 6 = 6 threads):
#   julia +1.12 -O1 -t 6 --project=. development/contraction_plans/benchmark_contract_onesite.jl
#
# WHAT IT DOES
#   The plans of the plan file that were made for a RAM budget of at most RAM_BUDGET_GIB (column ram_gib) and for the dimensions in DIMS (nothing = all dimensions of the file)
#   are timed, one after the other, grouped by dimensions (chi_env, chi_peps, D), the cheapest dimensions first. Every plan is timed REPS times after the warm-up.
#   - ENV_LOADER = nothing (default): the contraction runs on a MOCK environment, random dense tensors with the dimensions of the plan.
#     ALL plans of the same dimensions get EXACTLY the same mock environment (every child builds it with the same MOCK_SEED; the benchmark checks this, see below).
#     The time goes into the column time_s of the plan file.
#   - ENV_LOADER = "development/contraction_plans/env_loader_example.jl": the contraction runs on the REAL environment that this file builds (see env_loader_example.jl). Only the plans
#     for the dimensions of that environment are timed (the others are skipped). The time goes into the column time_real_s.
#     contract_onesite_auto uses time_real_s if it is filled, otherwise time_s.
#   The seven columns allocator, rss_gib, pool_gib, sys_s, minflt, check, note belong to the real run if there is one, otherwise to the mock run.
#
#   WHERE THE RESULTS GO
#   - data/contraction_plans/benchmark_contract_onesite/<date>_<time>/   (a new run folder for every run, see tools/benchmark_plots.jl):
#       results.csv   one row per plan with ALL repetitions (written after every plan, so nothing is lost if the run stops)
#       settings.txt  the settings of the run
#       the plots     made at the end of the run by tools/benchmark_plots.jl (BENCH_PLOTS); a new plot is a new function + one entry there
#   - the plan file: after every plan the FASTEST repetition is written into the time column of its row (CE.onesite_update_plans), also if the row had a time already.
#
#   STOPPING AND CONTINUING (RUN_DIR)
#   - A run that stops (Ctrl-C, switched off, ...) keeps everything that was done. Set RUN_DIR to the name of its run folder (for example "20261111_2230") and start again:
#     the plans that are in its results.csv are skipped, the rest is timed, and the plots are made from ALL rows of the folder. If nothing is left to time, only the plots are made again.
#   - MAX_HOURS: no group of dimensions is STARTED if the time that is still left is predicted to be too short for ALL its plans (the prediction is made from the plans that are
#     done: the first group always runs). A group that is skipped is printed. Plans of one group are never split, so that the mean of a group (right panel of the plot) is complete.
#   A plan that needs more memory than is free (model estimate > 85 % of the free memory) is not run (unless FORCE): it gets a `note` and stays without a time.
#
#   A time is only valid for the machine, the allocator and the number of (BLAS) threads that it was measured with: on another machine use another plan file
#   (PLAN_FILE here and in find_orders_ome.jl) and time it there. A plan that was made for a large RAM budget (column ram_gib) is only worth timing on a machine
#   that has that much memory.

using Printf, LinearAlgebra
using TensorKit
using ClusterExpansions         # Import this full package we are working on
const CE = ClusterExpansions

# ======================================================================================================================
# SETTINGS:
# ======================================================================================================================
const PLAN_FILE  = CE.ONESITE_PLAN_FILE   # plan file that is read and updated (the same constant is in find_orders_ome.jl)
const ENV_LOADER = nothing                # nothing = mock environment (time_s); "development/contraction_plans/env_loader_example.jl" = real environment (time_real_s)
const MOCK_SEED  = 1                      # seed of the random mock environment: the same seed gives the same environment for all plans of the same dimensions
const REPS       = 2                      # timed repetitions of the contraction after the warm-up (the fastest is used; all are in results.csv)
const ALLOCATOR  = "reuse"                # "reuse" (the allocator of contract_onesite_auto) or "manual"
const FORCE      = true                   # true = run also if the estimated memory exceeds 85 % of the free memory
const RAM_BUDGET_GIB = 4                  # only plans made for a RAM budget (column ram_gib of the plan file) of at most this many GiB are timed (4 = laptop)
const DIMS       = nothing                # nothing = all dimensions of the plan file; or a list such as [(8, 4, 3), (12, 4, 3)] for (chi_env, chi_peps, D)
const MAX_HOURS  = 10                     # no new group of dimensions is started if it is predicted not to finish within this many hours after the start (see above)
const RUN_DIR    = "20261111_2104"        # "" = new run folder; the NAME of an existing run folder (for example "20261111_2230") = continue that run / make its plots again
# ======================================================================================================================

# The plots and the results.csv functions are only needed by the parent process (loading Plots takes long: the children do not do it)
any(==("--child"), ARGS) || include(joinpath(@__DIR__, "tools", "benchmark_plots.jl"))

# WHAT IS MEASURED (the plan file gets: time, allocator, rss_gib, pool_gib, sys_s, minflt, check, note; results.csv has all repetitions and more)
#   time         wall time of the whole sliced contraction (fastest of REPS, GC.gc() before each); NOT including compilation
#   model_*      cost model (dense; src/utility/contract_onesite_slicing.jl): flops relative to current unsliced, peak of the intermediates in GiB (screen only)
#   tracked_gib  peak bytes of ALL temporaries of one slice (TrackingAllocator; includes TensorKit's permutation copies) (screen only)
#   rss_gib      peak resident memory of the whole child process (Sys.maxrss) -- includes Julia/TensorKit (~1-2 GiB), the environment
#                tensors, garbage not yet collected, and (reuse) the pool
#   pool_gib     reuse only: what the ReuseAllocator asked the OS for at its high-water mark
#   sys_s, minflt  Linux only (/proc/self/stat): system CPU seconds and minor page faults per repetition (empty on Windows)
#   check        relative deviation of the value from the first plan of the same dimensions in this run (so 0 for that first plan; should be ~1e-10 or smaller)
#   env_fp       fingerprint of the environment (sum of the norms of the 27 tensors + M): must be the same number for all plans of the same dimensions
# NOTE on the repetitions: the allocator is created once per plan, so repetition 1 also pays for growing the pool of the reuse allocator, and later repetitions reuse it.

# The absolute path of the plan file (also used as the key of the plan cache, so the update and the reads agree)
const PLAN_PATH = abspath(PLAN_FILE)
# Convert bytes to GiB
GiB(x) = x / 2^30
# SEP = tab
const SEP = '\t'


# Turn the command-line arguments (--key=value) into a Dict of strings. Only the child processes get arguments (--child --plan=NAME --dims=a,b,c).
function parse_cli(args)
    o = Dict{String,String}()
    for a in args
        # Loop over all arguments, and skip anything that doesn't start with --
        startswith(a, "--") || continue

        # Drop the leading -- and split at the first = only
        kv = split(a[3:end], '='; limit = 2)

        # Add this to Dic: key -> value (if no =, like in --child, then it gets value "true")
        o[String(kv[1])] = length(kv) == 2 ? String(kv[2]) : "true"
    end
    return o
end

# return (env, M), either from the real env (ENV_LOADER) or a random mock env with dims = (chi_env, chi_peps, D)
function load_env(dims)
    # if a real env was given:
    if ENV_LOADER !== nothing
        # Now we evaluate this file inside Main
        Base.include(Main, abspath(ENV_LOADER))
        # And we run it in Main, and retrieve the answer (the binding is looked up AND called with invokelatest: Julia 1.12 warns about
        # a binding that was defined after this function started running)
        f = Base.invokelatest(getproperty, Main, :onesite_load_env)
        return Base.invokelatest(f)
    end
    # If no real env was given, we build a random mock env (seeded: the same dims and the same MOCK_SEED always give the same tensors)
    return CE.onesite_mock_env(ℂ^dims[1], ℂ^dims[2], ℂ^dims[3]; seed = MOCK_SEED)
end

# A number that identifies the environment: the sum of the norms of all 27 tensors and M. Plans of the same dimensions must give exactly the same number.
env_fingerprint(env, M) = sum(norm, env.C) + sum(norm, env.E) + sum(norm, env.A) + norm(M)

# The ram_gib column (column 11 of the plan file) of every data row, in the order in which CE.onesite_read_plans returns the rows
# (the same lines are skipped: empty lines, comments, the header; CE.onesite_read_plans does not return this column)
function plan_ram_gibs(path)
    out = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#") || startswith(s, "chi_env")) && continue
        f = split(line, ';')
        push!(out, length(f) >= 11 && !isempty(strip(f[11])) ? parse(Float64, strip(f[11])) : NaN)
    end
    return out
end

# The rows of the plan file that were made for the bond dimensions d (the 54 numbers of CE.onesite_dims), as a Vector of (name, tree, cuts, source, ram_gib, time_s, time_real_s).
# The name is SOURCE#N (N = number of the row among the rows for these dims, whatever their RAM budget), so parent and child process find the same row.
function plan_rows(d)
    all = CE.onesite_read_plans(PLAN_PATH)
    ram = plan_ram_gibs(PLAN_PATH)
    keep = [i for (i, r) in enumerate(all) if CE.onesite_dims(r.chi_env, r.chi_peps, r.D) == d]
    return [(name = string(all[i].source, "#", n), tree = all[i].tree, cuts = all[i].cuts, source = all[i].source, ram_gib = ram[i],
             time_s = all[i].time_s, time_real_s = all[i].time_real_s) for (n, i) in enumerate(keep)]
end

# LINUX ONLY: Read the process's minor page faults and CPU times, on windows this returns nothing
function procstat()
    isfile("/proc/self/stat") || return nothing
    p = split(split(read("/proc/self/stat", String), ") ")[end])
    return (minflt = parse(Int, p[8]), user = parse(Int, p[12]) / 100, sys = parse(Int, p[13]) / 100)    # 100 clock ticks/s assumed
end

# Print which BLAS library is in use (OpenBLAS or MKL)
# GEMM speed depends on it
function blas_name()
    try
        return basename(BLAS.get_config().loaded_libs[1].libname)
    catch
        return "BLAS ?"
    end
end

# Create a fresh allocator from the string ALLOCATOR
make_alloc(kind) = kind == "reuse" ? CE.ReuseAllocator() : CE.ManualAllocator()


# CHILD: runs ONE plan in a fresh processor and prints a result line on stdout
# o = the options of the child: plan = NAME of the plan (SOURCE#N), dims = chi_env,chi_peps,D (for the mock environment; the real environment brings its own)
function run_child(o)
    name = o["plan"]
    dims = parse.(Int, split(o["dims"], ','))
    akind = ALLOCATOR

    # Build the mock/real environment
    env, M = load_env(dims)
    # The fingerprint of the environment (the parent checks that all plans of the same dimensions have the same one)
    fp = env_fingerprint(env, M)
    # define the vector of 54 bond dimensions, read from the actual tensors
    d = CE.onesite_dims(env, M)

    # The plan: its tree, re-ordered such that the peak of the sliced network is minimal, and its cuts
    orders = plan_rows(d)
    i = findfirst(r -> r.name == name, orders)
    i === nothing && error("no plan named $name for these dimensions in $PLAN_PATH")
    cuts = orders[i].cuts
    kernel = CE.onesite_reorder(orders[i].tree, d; cuts)          # a generated kernel: the model tree is the kernel's tree

    # Turn each cut label into label=>dimension: e.g. [:L5 => 8, :L4 => 5, :A8 => 5]
    cutp = CE.onesite_full_cuts(env, M, cuts)
    # Define the resulting number of slices: products of piece count
    ns = CE.onesite_nslices(env, M, cutp)

    # Define the model cost of this plan (flops, peak elements, slices)
    # And define base as the model cost of the current tree unsliced
    cost = CE.onesite_cost(kernel, d; cuts); base = CE.onesite_cost(CE.TREE_CURRENT, d)

    # Define the element size: 16 bytes for a single ComplexF64 element
    esize = sizeof(scalartype(M))
    # Now transform the cost.peak to a number in GiB
    mpeak = GiB(cost.peak * esize)

    # Now we use the same rule of thumb as contract_onesite_auto (CE.ONESITE_RAM_RULES) for calculating
    # wether or not we have currently enough free RAM: predicted size of the process = baseline_gib + factor * model peak,
    # with factor = rules.reuse or rules.manual
    est = CE.ONESITE_RAM_RULES.baseline_gib + CE.ONESITE_RAM_RULES[Symbol(akind)] * mpeak # empirical rule of thumb; not a guarantee

    # Compare with currently free RAM
    free = GiB(Sys.free_memory())

    # Already construct the first six output columns
    base_cols = (name, join(cuts, ","), akind, round(Int, ns), cost.flops / base.flops, mpeak)

    # Memory guard: if we don't have enough ram
    if est > 0.85 * free && !FORCE
        # Then it just prints a 7-field line and stops
        println(join((base_cols..., "skipped: estimated $(round(est, digits = 1)) GiB > 85% of free $(round(free, digits = 1)) GiB (set FORCE = true)"), SEP))
        return
    end

    # Resolve the kernel: generate and compile the contraction function of this tree ONCE (it compiles on its first call)
    kern = CE.onesite_make_kernel(kernel)

    # every run goes through a TrackingAllocator (a few hundred dictionary operations per slice: negligible against seconds per slice)
    # so that the kernel is compiled only once per plan.
    # warm-up: compile everything.  mock env: tiny env (same Vector{Any} specialisation); real env: one slice of the real thing

    # Now we print that the long compile phase has started
    println(stderr, "[$name / $akind] compiling (first call) ..."); flush(stderr)

    # Warm up (not timed yet)
    if ENV_LOADER !== nothing
        # If we use a real env: then we run one slice of the real thing (using typed tensor vectors)
        CE.contract_onesite_sliced(env, M, cutp; kernel = kern, allocator = CE.TrackingAllocator(make_alloc(akind)), max_slices = 1)
    else
        # If we use a mock env: create the mock env, and run the whole sliced contraction on a tiny (2, 2, 2) mock
        # (mock stores tensors in Vector{any} so compiled code is same as for big mocks)
        # (this re-seeds the random number generator, but the big environment above is already built)
        e2, M2 = CE.onesite_mock_env(ℂ^2, ℂ^2, ℂ^2)
        CE.contract_onesite_sliced(e2, M2, CE.onesite_full_cuts(e2, M2, cuts); kernel = kern, allocator = CE.TrackingAllocator(make_alloc(akind)))
    end

    # Before value of memory: Retrieve the peak resident memory so far (Julia, TensorKit, environment compilation)
    rss0 = GiB(Sys.maxrss())

    # Create the fresh allocator for the measurement (if reuse allocator: pools start empty)
    ta = CE.TrackingAllocator(make_alloc(akind))
    #  z collects answer,  ts collects the timings
    z = nothing; ts = Float64[]
    # ps0: LINUX only: page-faults and CPU snapshots. WINDOWS: nothing
    ps0 = procstat()

    # For all repetitions that were asked
    for r in 1:REPS
        # Collect garbage first
        GC.gc()
        # Time the full sliced contraction, z keeps the contraction value (for correctness check)
        push!(ts, @elapsed z = CE.contract_onesite_sliced(env, M, cutp; kernel = kern, allocator = ta))
        # Print some info
        println(stderr, "[$name / $akind] rep $r: $(round(ts[end], digits = 2)) s"); flush(stderr)
    end

    # Take a snapshot after the loop and the peak RSS after it
    ps1 = procstat()
    rss1 = GiB(Sys.maxrss())

    # The peak of all live temporaries (any slice, all reps)
    tracked = GiB(ta.peak_bytes)
    # The amount of memory that the reuse allocator requested from OS (also counting pooled memory)
    pool = ta.inner isa CE.ReuseAllocator ? GiB(ta.inner.peak_os_bytes) : NaN
    # free the pool (if reuse)
    ta.inner isa CE.ReuseAllocator && CE.release!(ta.inner)

    # tmin = best timing for single rep
    tmin = minimum(ts)

    # LINUX only, WINDOWS: NAN
    rf = ps0 === nothing ? NaN : (ps1.minflt - ps0.minflt) / REPS
    rs = ps0 === nothing ? NaN : (ps1.sys - ps0.sys) / REPS

    # Print the final result line as 18 tab separated fields: six base columns, time (fastest), time per slice, tracked, rss, rss before, pool, sys, minflt, re, im,
    # the times of ALL repetitions (separated by a space) and the fingerprint of the environment.
    println(join((base_cols..., tmin, tmin / ns, tracked, rss1, rss0, pool, rs, rf, real(z), imag(z), join(ts, " "), fp), SEP))
end


# PARENT: runs one plan in a fresh julia process (run_child) and returns its answer as a named tuple.
# msg == "" means a complete result; otherwise the plan was skipped by the child (memory) or the child failed, and msg says why.
function bench_row(name, g)
    # Build the command line that will start a new Julia process
    # it uses same: Julia, project, thread count as us.
    # it adds --child, so now it will get filtered out above
    # it also adds: the plan name and the dimensions
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$(Threads.nthreads()) $(abspath(PROGRAM_FILE)) --child --plan=$name --dims=$(g[1]),$(g[2]),$(g[3])`

    # Now we start the worker, wait until it is done and return what is printed
    out = try
        strip(read(cmd, String))
    # If the worker crashes (e.g. running out of memory)
    catch e
        # Then we store: "failed: <error>" and the benchmark can continue
        "failed: " * sprint(showerror, e)
    end

    # We only keep the last printed line of the worker's answer (can also be an error message or the memory skip message)
    line = String(last(split(out, '\n')))
    # We split it (SEP = tab) into separate fields f
    f = split(line, SEP)

    nan = NaN
    empty_row(msg) = (name = name, cuts = "", alloc = ALLOCATOR, slices = "", flops_rel = nan, peak = nan, time = nan, per_slice = nan, tracked = nan,
                      rss = nan, rss0 = nan, pool = nan, sys = nan, minflt = nan, re = nan, im = nan, times = Float64[], fp = nan, msg = msg)
    # A full result has 18 fields
    if length(f) == 18
        v = parse.(Float64, f[5:16])
        return (name = name, cuts = String(f[2]), alloc = String(f[3]), slices = String(f[4]), flops_rel = v[1], peak = v[2], time = v[3], per_slice = v[4],
                tracked = v[5], rss = v[6], rss0 = v[7], pool = v[8], sys = v[9], minflt = v[10], re = v[11], im = v[12],
                times = parse.(Float64, split(f[17])), fp = parse(Float64, f[18]), msg = "")
    # Exactly 7 fields: the child skipped the plan because of memory
    elseif length(f) == 7
        return merge(empty_row(String(f[7])), (cuts = String(f[2]), slices = String(f[4])))
    end
    # Otherwise the child crashed (we keep the last line it printed as the message)
    return empty_row(line)
end

# Print the table of the rows of this run on screen
function print_table(rows)
    # Print the table header
    @printf("\n%-8s %-22s %-24s %6s %10s %10s %9s %9s %9s %8s\n", "dims", "plan", "cuts", "slices", "model flops", "time s", "reps s", "rss GiB", "pool GiB", "check")

    # Loop over all rows
    for r in rows
        dims = string(r.chi_env, ",", r.chi_peps, ",", r.D)
        # If the row crashed/was skipped than there is no time; we only print the plan, cuts and the note
        if !isfinite(r.tmin)
            @printf("%-8s %-22s %-24s %s\n", dims, r.name, r.cuts[1:min(end, 24)], r.note); continue
        end

        # Now we print the finished row with the check value
        @printf("%-8s %-22s %-24s %6.0f %10.3e %10.2f %9s %9.3f %9.3f %8.1e\n", dims, r.name, r.cuts[1:min(end, 24)], r.slices, r.flops, r.tmin,
                join(round.(r.times; digits = 1), "/"), r.rss, r.pool, r.check)
    end
end

# Print an environment header
function print_system_info()
    println("Julia ", VERSION, ", threads ", Threads.nthreads(), ", BLAS threads ", BLAS.get_num_threads(), " (", blas_name(), ")",
            ", total memory ", round(GiB(Sys.total_memory()), digits = 1), " GiB, free ", round(GiB(Sys.free_memory()), digits = 1), " GiB")
    println("TensorKit ", pkgversion(TensorKit), ", TensorOperations ", pkgversion(CE.TensorOperations),
            "   environment: ", ENV_LOADER === nothing ? "mock (dense, random, seed $MOCK_SEED)" : "real (" * ENV_LOADER * ")", ", allocator ", ALLOCATOR, ", reps ", REPS)
end

# Round to 6 significant digits (NaN and 0 stay as they are), for the plan file
rnd(x) = (isfinite(x) && x != 0) ? round(x; sigdigits = 6) : x

# The settings of this run, written (appended) to settings.txt of the run folder
function write_settings(outdir)
    open(joinpath(outdir, "settings.txt"), "a") do io
        println(io, "start: ", Libc.strftime("%Y-%m-%d %H:%M:%S", time()))
        println(io, "Julia ", VERSION, ", threads ", Threads.nthreads(), ", BLAS threads ", BLAS.get_num_threads(), " (", blas_name(), ")")
        println(io, "memory GiB: total ", round(GiB(Sys.total_memory()), digits = 1), ", free at start ", round(GiB(Sys.free_memory()), digits = 1))
        println(io, "PLAN_FILE = ", PLAN_PATH)
        println(io, "ENV_LOADER = ", ENV_LOADER, ", MOCK_SEED = ", MOCK_SEED, ", REPS = ", REPS, ", ALLOCATOR = ", ALLOCATOR, ", FORCE = ", FORCE)
        println(io, "RAM_BUDGET_GIB = ", RAM_BUDGET_GIB, ", DIMS = ", DIMS, ", MAX_HOURS = ", MAX_HOURS, ", RUN_DIR = \"", RUN_DIR, "\"")
        println(io)
    end
end


# Main function, if in child mode: runs one plan
# Otherwise it is the parent: times all selected plans that are not in results.csv of the run folder yet, writes the measurements back, and makes the plots
function main()
    # Define Dict o: holding the command-line options of the current process; only a child has them
    o = parse_cli(ARGS)
    # Check if the process is a child: this only happens if it contains --child, this only gets added by the script itself
    # This function is ran first as parent, then once for every plan as a child
    haskey(o, "child") && return run_child(o)

    envkind = ENV_LOADER === nothing ? :mock : :real
    timecol = envkind === :mock ? "time_s" : "time_real_s"
    plans = CE.onesite_read_plans(PLAN_PATH)
    isempty(plans) && error("no plans in $PLAN_PATH: make them first with development/contraction_plans/find_orders_ome.jl")

    # The groups of dimensions that are timed: all of them for the mock environment (or those of DIMS); for the real environment only the one with its dimensions
    groups = unique((r.chi_env, r.chi_peps, r.D) for r in plans)
    DIMS === nothing || (groups = filter(g -> g in DIMS, groups))
    if ENV_LOADER !== nothing
        env, M = load_env(nothing)
        d_env = CE.onesite_dims(env, M)
        groups = filter(g -> CE.onesite_dims(g...) == d_env, groups)
        if isempty(groups)
            println("no plan in $PLAN_PATH has the bond dimensions of the environment of $ENV_LOADER: make plans for these dimensions with development/contraction_plans/find_orders_ome.jl")
            return
        end
    end
    isempty(groups) && (println("no plan in $PLAN_PATH has dimensions in DIMS = $DIMS"); return)
    # Cheapest dimensions first (model flops of the current tree, unsliced): if the time runs out, the most expensive dimensions are the ones that are left out
    sort!(groups; by = g -> CE.onesite_cost(CE.TREE_CURRENT, CE.onesite_dims(g...)).flops)

    # The run folder, and the rows of results.csv that are already in it (empty for a new run)
    outdir = isempty(RUN_DIR) ? bench_run_dir(@__FILE__) : existing_bench_run_dir(@__FILE__, RUN_DIR)
    resultspath = joinpath(outdir, "results.csv")
    isfile(resultspath) || open(io -> println(io, BENCH_HEADER), resultspath, "w")
    write_settings(outdir)
    rows = read_bench_results(resultspath)                      # ALL rows of the run folder (the plots use these)
    nold = length(rows)

    # reference value and fingerprint of the first plan of every group of dimensions (also from the rows of an earlier part of this run)
    refs = Dict{Any,ComplexF64}(); fps = Dict{Any,Float64}()
    for r in rows
        isfinite(r.tmin) || continue
        g = (r.chi_env, r.chi_peps, r.D)
        haskey(refs, g) || (refs[g] = complex(r.re, r.im); fps[g] = r.fp)
    end
    isdone(g, name) = any(r -> (r.chi_env, r.chi_peps, r.D) == g && r.name == name && isfinite(r.tmin), rows)

    println("plan file: ", PLAN_PATH, "  (", length(plans), " plans)")
    println("run folder: ", outdir, "  (", nold, " results in it already)")
    println("selected: plans for RAM budget <= ", RAM_BUDGET_GIB, " GiB, ", length(groups), " dimension sets, cheapest first: ", join(("(" * join(g, ",") * ")" for g in groups), " "))
    print_system_info()

    # Prediction of the time of a plan from the plans that are done: fixed overhead (compilation, warm-up, building the environment) + REPS * flops / (flops per second of the repetitions)
    # The factor 1.3 is a safety margin: the speed differs between trees (the OPT_TRUE tree has skinnier matrices).
    nfin = 0; over = 0.0; work = 0.0; reptime = 0.0
    predicted(flops) = nfin == 0 ? 0.0 : over / nfin + 1.3 * REPS * flops / (work / reptime)

    t_start = time()
    nrun = 0
    # This is the main loop for the parent process: goes through the groups of dimensions, and in each group through the plans that need timing one by one
    # For each plan it does 1 of 3: skip it, run it, record a failure.
    for g in groups
        d = CE.onesite_dims(g...)
        # The plans of this group for the RAM budget, with their model cost
        todo = NamedTuple[]
        for r in plan_rows(d)
            (isfinite(r.ram_gib) && r.ram_gib <= RAM_BUDGET_GIB) || continue
            isdone(g, r.name) && continue
            cost = CE.onesite_cost(CE.onesite_reorder(r.tree, d; cuts = r.cuts), d; cuts = r.cuts)
            push!(todo, (plan = r, cost = cost))
        end
        isempty(todo) && continue

        # Time left? (the first group always runs)
        left = 3600 * MAX_HOURS - (time() - t_start)
        pred = sum(predicted(j.cost.flops) for j in todo)
        if nfin > 0 && pred > left
            println(stderr, "\n($(join(g, ","))) NOT started: predicted $(round(pred / 60, digits = 1)) min for its $(length(todo)) plans, but only $(round(max(left, 0) / 60, digits = 1)) min of MAX_HOURS are left")
            continue
        end

        for (n, j) in enumerate(todo)
            r = j.plan; cost = j.cost
            nrun += 1
            # Show progress
            println(stderr, "\n[($(join(g, ","))) plan $n/$(length(todo)), $(round((time() - t_start) / 60, digits = 1)) min since the start] $(r.name), $(length(r.cuts)) cuts"); flush(stderr)
            # Identify the row in the CSV corresponding to current job
            key = (chi_env = g[1], chi_peps = g[2], D = g[3], tree = r.tree, cuts = r.cuts)
            # What is the same for every outcome of this plan
            ident = (chi_env = g[1], chi_peps = g[2], D = g[3], name = r.name, source = r.source, ram_gib = r.ram_gib, cuts = join(r.cuts, " "),
                     slices = cost.slices, flops = cost.flops, peak = GiB(cost.peak * sizeof(ComplexF64)), env = ENV_LOADER === nothing ? "mock seed $MOCK_SEED" : "real",
                     blas = BLAS.get_num_threads())
            # A row without a time (skipped or failed)
            norow(msg) = merge(ident, (times = Float64[], tmin = NaN, tracked = NaN, rss = NaN, pool = NaN, alloc = ALLOCATOR, fp = NaN, re = NaN, im = NaN, check = NaN, note = msg))
            function addrow(row)
                push!(rows, row)
                try
                    append_bench_row(resultspath, row)
                catch e
                    @warn "could not write results.csv (the row is kept in memory, the plots still work)" exception = e
                end
            end

            # Skip a plan that needs more memory than is free, without starting a process (the child has the same guard)
            # We assume 16 bytes per element (ComplexF64), child checks explicitely again for real scalar size
            est = CE.ONESITE_RAM_RULES.baseline_gib + CE.ONESITE_RAM_RULES[Symbol(ALLOCATOR)] * GiB(cost.peak * sizeof(ComplexF64))
            # Compare estimated needed ram with currently free ram
            free = GiB(Sys.free_memory())
            if est > 0.85 * free && !FORCE
                # Add message and NaN everywhere else
                msg = "skipped: estimated $(round(est, digits = 1)) GiB > 85% of free $(round(free, digits = 1)) GiB (set FORCE = true)"
                addrow(norow(msg))
                println(stderr, "    ", msg)

                # Only write the note (= msg) to the .csv file
                planfile_update(key, (note = msg,), envkind)
                continue
            end

            # Run the contraction!
            t_wall = @elapsed res = bench_row(r.name, g)

            # If there is a message, this means that child skipped the plan or crashed
            if res.msg != ""
                addrow(norow(res.msg))
                # failed or skipped by the child: the note says why
                planfile_update(key, (note = res.msg,), envkind)
                continue
            end

            # check: relative deviation of the value from the first result of the same dimensions
            z = complex(res.re, res.im)
            c = 0.0
            # reminder: g is dimension triplet (which we loop over)
            haskey(refs, g) ? (c = abs(z - refs[g]) / abs(refs[g])) : (refs[g] = z)
            # the environment must be the same one as for the first plan of these dimensions
            note = ""
            if haskey(fps, g)
                if res.fp != fps[g]
                    note = "ENVIRONMENT DIFFERS from the first plan of these dimensions (fingerprint $(res.fp) against $(fps[g]))"
                    println(stderr, "    WARNING: ", note)
                end
            else
                fps[g] = res.fp
            end

            addrow(merge(ident, (times = res.times, tmin = minimum(res.times), tracked = res.tracked, rss = res.rss, pool = res.pool, alloc = res.alloc, fp = res.fp,
                                 re = res.re, im = res.im, check = c, note = note)))

            # for the prediction of the next groups
            nfin += 1; reptime += sum(res.times); work += length(res.times) * cost.flops; over += t_wall - sum(res.times)

            # Write the measurements back into THE row of the plan file (time = fastest repetition, allocator, rss_gib, pool_gib, sys_s, minflt, check, note)
            planfile_update(key, (time_s = rnd(minimum(res.times)), allocator = res.alloc, rss_gib = rnd(res.rss), pool_gib = rnd(res.pool),
                                  sys_s = rnd(res.sys), minflt = rnd(res.minflt), check = rnd(c), note = note), envkind)
            println(stderr, "    ", timecol, " = ", rnd(minimum(res.times)), " (reps ", join(round.(res.times; digits = 2), ", "), "), written into the plan file and results.csv")
        end
    end

    # Summary on screen
    new_rows = rows[(nold + 1):end]
    isempty(new_rows) ? println("\nnothing to time in this run (all selected plans are in results.csv, or the time did not suffice)") : print_table(new_rows)
    ok = filter(r -> isfinite(r.tmin), rows)
    println("\nenvironment check (same environment for all plans of the same dimensions):")
    for g in unique((r.chi_env, r.chi_peps, r.D) for r in ok)
        sel = [r for r in ok if (r.chi_env, r.chi_peps, r.D) == g]
        @printf("  (%s): %d plans, same environment: %s, largest check %.1e\n", join(g, ","), length(sel), length(unique(r.fp for r in sel)) == 1 ? "yes" : "NO", maximum(r.check for r in sel))
    end
    println("\ntotal time of this run: ", round((time() - t_start) / 60, digits = 1), " min")
    println("results: ", resultspath)

    # The plots (from ALL rows of the run folder); each plot in its own try
    make_all_benchmark_plots(rows, outdir)
end

# Write into the plan file; if that fails (on Windows, a file in a OneDrive folder can be locked for a moment) the run goes on: results.csv has the data anyway
function planfile_update(key, vals, envkind)
    try
        CE.onesite_update_plans(PLAN_PATH, [merge(key, vals)]; env = envkind)
    catch e
        @warn "could not update the plan file (the result is in results.csv)" exception = e
    end
    return nothing
end

# This starts the whole thing
main()