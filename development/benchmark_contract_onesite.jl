# This file times the plans of the plan file src/utility/onesite_plans.csv (written by development/find_orders_ome.jl): the sliced contraction
# `contract_onesite` with a given contraction tree and given cut bonds.
# Each plan runs in a FRESH child julia process, so that `Sys.maxrss()` (peak resident memory) belongs to that plan only and no allocator pool survives.
#
# USAGE:
#   -O1 keeps the just-in-time compilation of the 27-tensor kernels shorter than the default -O2, -t 8 = 8 threads):
#   julia +1.12 -O1 -t 8 --project=. development/benchmark_contract_onesite.jl
#
# WHAT IT DOES
#   Every plan of the plan file that has no time yet is timed, one after the other, grouped by dimensions (chi_env, chi_peps, D).
#   After each plan the measurements are written back into the same row of the plan file (CE.onesite_update_plans). Nothing else is written.
#   - ENV_LOADER = nothing (default): the contraction runs on a MOCK environment, random dense tensors with the dimensions of the plan, for ALL plans of the file.
#     The time goes into the column time_s.
#   - ENV_LOADER = "development/env_loader_example.jl": the contraction runs on the REAL environment that this file builds (see env_loader_example.jl). Only the plans
#     for the dimensions of that environment are timed (the others are skipped). The time goes into the column time_real_s.
#     contract_onesite_auto uses time_real_s if it is filled, otherwise time_s.
#   The seven columns allocator, rss_gib, pool_gib, sys_s, minflt, check, note belong to the real run if there is one, otherwise to the mock run.
#   A plan that has a time in the column of this run is skipped, so an interrupted run can simply be started again. To time a plan again, empty its time cell in the csv.
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
const ENV_LOADER = nothing                # nothing = mock environment (time_s); "development/env_loader_example.jl" = real environment (time_real_s)
const REPS       = 1                      # timed repetitions of the contraction after the warm-up (the minimum is written)
const ALLOCATOR  = "reuse"                # "reuse" (the allocator of contract_onesite_auto) or "manual"
const FORCE      = false                  # true = run also if the estimated memory exceeds 85 % of the free memory
# ======================================================================================================================

# WHAT IS MEASURED (the plan file gets: time, allocator, rss_gib, pool_gib, sys_s, minflt, check, note)
#   time         wall time of the whole sliced contraction (min over REPS, GC.gc() before each); NOT including compilation
#   model_*      cost model (dense; src/utility/contract_onesite_slicing.jl): flops relative to current unsliced, peak of the intermediates in GiB (screen only)
#   tracked_gib  peak bytes of ALL temporaries of one slice (TrackingAllocator; includes TensorKit's permutation copies) (screen only)
#   rss_gib      peak resident memory of the whole child process (Sys.maxrss) -- includes Julia/TensorKit (~1-2 GiB), the environment
#                tensors, garbage not yet collected, and (reuse) the pool
#   pool_gib     reuse only: what the ReuseAllocator asked the OS for at its high-water mark
#   sys_s, minflt  Linux only (/proc/self/stat): system CPU seconds and minor page faults per repetition (empty on Windows)
#   check        relative deviation of the value from the first plan of the same dimensions in this run (so 0 for that first plan; should be ~1e-10 or smaller)

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
    # If no real env was given, we build a random mock env
    return CE.onesite_mock_env(ℂ^dims[1], ℂ^dims[2], ℂ^dims[3])
end

# The rows of the plan file that were made for the bond dimensions d (the 54 numbers of CE.onesite_dims), as a Vector of (name, tree, cuts, time_s, time_real_s).
# The name is SOURCE#N (N = number of the row among the rows for these dims), so parent and child process find the same row.
function plan_rows(d)
    rows = CE.onesite_plans_for(d, PLAN_PATH)
    return [(name = string(r.source, "#", i), tree = r.tree, cuts = r.cuts, time_s = r.time_s, time_real_s = r.time_real_s) for (i, r) in enumerate(rows)]
end

# The time of a plan row in the column that this run fills
run_time(r) = ENV_LOADER === nothing ? r.time_s : r.time_real_s

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

    # Print the final result line as 16 tab separated fields: six base columns, time, time per slice, tracked, rss, rss before, pool, sys, minflt, re, im.
    println(join((base_cols..., tmin, tmin / ns, tracked, rss1, rss0, pool, rs, rf, real(z), imag(z)), SEP))
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
                      rss = nan, rss0 = nan, pool = nan, sys = nan, minflt = nan, re = nan, im = nan, msg = msg)
    # A full result has 16 fields
    if length(f) == 16
        v = parse.(Float64, f[5:16])
        return (name = name, cuts = String(f[2]), alloc = String(f[3]), slices = String(f[4]), flops_rel = v[1], peak = v[2], time = v[3], per_slice = v[4],
                tracked = v[5], rss = v[6], rss0 = v[7], pool = v[8], sys = v[9], minflt = v[10], re = v[11], im = v[12], msg = "")
    # Exactly 7 fields: the child skipped the plan because of memory
    elseif length(f) == 7
        return merge(empty_row(String(f[7])), (cuts = String(f[2]), slices = String(f[4])))
    end
    # Otherwise the child crashed (we keep the last line it printed as the message)
    return empty_row(line)
end

# Print the table of rows on screen; chk[i] = relative deviation of the value of row i from the reference (NaN: no check)
function print_table(rows, chk)
    # Print the table header
    @printf("\n%-24s %-24s %6s %9s %9s %10s %9s %9s %9s %8s\n", "plan", "cuts", "slices", "flops/cur", "model GiB", "time s", "tracked", "rss GiB", "pool GiB", "check")

    # Loop over all rows
    for (i, r) in enumerate(rows)
        # If the row crashed/was skipped than there is no time; we only print the plan, cuts and the message
        if r.msg != ""
            @printf("%-24s %-24s %s\n", r.name, r.cuts[1:min(end, 24)], r.msg); continue
        end

        # Now we print the finished row with the check value
        @printf("%-24s %-24s %6s %9.3f %9.3f %10.2f %9.3f %9.3f %9.3f %8.1e\n", r.name, r.cuts[1:min(end, 24)], r.slices, r.flops_rel, r.peak, r.time, r.tracked, r.rss, r.pool, chk[i])
    end
end

# Print an environment header
function print_system_info()
    println("Julia ", VERSION, ", threads ", Threads.nthreads(), ", BLAS threads ", BLAS.get_num_threads(), " (", blas_name(), ")",
            ", total memory ", round(GiB(Sys.total_memory()), digits = 1), " GiB, free ", round(GiB(Sys.free_memory()), digits = 1), " GiB")
    println("TensorKit ", pkgversion(TensorKit), ", TensorOperations ", pkgversion(CE.TensorOperations),
            "   environment: ", ENV_LOADER === nothing ? "mock (dense, random)" : "real (" * ENV_LOADER * ")", ", allocator ", ALLOCATOR, ", reps ", REPS)
end

# Round to 6 significant digits (NaN and 0 stay as they are), for the plan file
rnd(x) = (isfinite(x) && x != 0) ? round(x; sigdigits = 6) : x


# Main function, if in child mode: runs one plan
# Otherwise it is the parent: times all plans of the plan file that have no time yet, and writes the measurements back
function main()
    # Define Dict o: holding the command-line options of the current process; only a child has them
    o = parse_cli(ARGS)
    # Check if the process is a child: this only happens if it contains --child, this only gets added by the script itself
    # This function is ran first as parent, then once for every plan as a child
    haskey(o, "child") && return run_child(o)

    envkind = ENV_LOADER === nothing ? :mock : :real
    timecol = envkind === :mock ? "time_s" : "time_real_s"
    plans = CE.onesite_read_plans(PLAN_PATH)
    isempty(plans) && error("no plans in $PLAN_PATH: make them first with development/find_orders_ome.jl")

    # The groups of dimensions that are timed: all of them for the mock environment; for the real environment only the one with its dimensions
    groups = unique((r.chi_env, r.chi_peps, r.D) for r in plans)
    if ENV_LOADER !== nothing
        env, M = load_env(nothing)
        d_env = CE.onesite_dims(env, M)
        groups = filter(g -> CE.onesite_dims(g...) == d_env, groups)
        if isempty(groups)
            println("no plan in $PLAN_PATH has the bond dimensions of the environment of $ENV_LOADER: make plans for these dimensions with development/find_orders_ome.jl")
            return
        end
    end

    # The jobs: the plans of these groups that have no time in the column of this run
    jobs = NamedTuple[]
    for g in groups, r in plan_rows(CE.onesite_dims(g...))
        isfinite(run_time(r)) && continue
        push!(jobs, (dims = g, name = r.name, plan = r))
    end
    println("plan file: ", PLAN_PATH, "  (", length(plans), " plans, ", length(jobs), " to time: those without ", timecol, ")")
    print_system_info()
    isempty(jobs) && (println("nothing to do (all selected plans have a $timecol; to time a plan again, empty its $timecol cell in the csv)"); return)

    # row record one result per plan, chk holds one number per row
    rows = NamedTuple[]; chk = Float64[]
    # ref maps dim triple to contraction value of the first plan that finished for those dims
    refs = Dict{Any,ComplexF64}()   
    # This is the main loop for the parent process: goes through the plans that needs timing one by one
    # For each plan it does 1 of 3: skip it, run it, record a failure.
    for (n, j) in enumerate(jobs)
        # g = dim triple, r = plan row
        g = j.dims; r = j.plan
        # Show progress
        println(stderr, "\n[$n/$(length(jobs))] (chi_env, chi_peps, D) = $g, $(j.name), $(length(r.cuts)) cuts"); flush(stderr)
        # Identify the row in the CSV corresponding to current job
        key = (chi_env = g[1], chi_peps = g[2], D = g[3], tree = r.tree, cuts = r.cuts)

        # Skip a plan that needs more memory than is free, without starting a process (the child has the same guard)
        d = CE.onesite_dims(g...)
        cost = CE.onesite_cost(CE.onesite_reorder(r.tree, d; cuts = r.cuts), d; cuts = r.cuts)
        # We assume 16 bytes per element (ComplexF64), child checks explicitely again for real scalar size
        est = CE.ONESITE_RAM_RULES.baseline_gib + CE.ONESITE_RAM_RULES[Symbol(ALLOCATOR)] * GiB(cost.peak * sizeof(ComplexF64))
        # Compare estimated needed ram with currently free ram
        free = GiB(Sys.free_memory())
        if est > 0.85 * free && !FORCE
            # Add message and NaN everywhere else
            msg = "skipped: estimated $(round(est, digits = 1)) GiB > 85% of free $(round(free, digits = 1)) GiB (set FORCE = true)"
            push!(rows, (name = j.name, cuts = join(r.cuts, ","), alloc = ALLOCATOR, slices = string(round(Int, cost.slices)), msg = msg,
                         flops_rel = NaN, peak = NaN, time = NaN, per_slice = NaN, tracked = NaN, rss = NaN, rss0 = NaN, pool = NaN, sys = NaN, minflt = NaN, re = NaN, im = NaN))
            push!(chk, NaN)
            println(stderr, "    ", msg)

            # Only write the note (= msg) to the .csv file
            CE.onesite_update_plans(PLAN_PATH, [merge(key, (note = msg,))]; env = envkind)
            continue
        end

        # Run the contraction!
        row = bench_row(j.name, g)
        # Add result to rows
        push!(rows, row)

        # If there is a message, this means that child skipped the plan or crashed 
        if row.msg != ""
            push!(chk, NaN)
            # failed or skipped by the child: the note says why
            CE.onesite_update_plans(PLAN_PATH, [merge(key, (note = row.msg,))]; env = envkind)
            continue
        end

        # check: relative deviation of the value from the first result of the same dimensions in this run
        z = complex(row.re, row.im)
        c = 0.0
        # reminder: g is dimension triplet (which we loop over)
        haskey(refs, g) ? (c = abs(z - refs[g]) / abs(refs[g])) : (refs[g] = z)
        push!(chk, c)

        # Write the measurements back into THE row of the plan file (time, allocator, rss_gib, pool_gib, sys_s, minflt, check, note)
        CE.onesite_update_plans(PLAN_PATH, [merge(key, (time_s = rnd(row.time), allocator = row.alloc, rss_gib = rnd(row.rss), pool_gib = rnd(row.pool),
                                                        sys_s = rnd(row.sys), minflt = rnd(row.minflt), check = rnd(c), note = ""))]; env = envkind)
        println(stderr, "    ", timecol, " = ", rnd(row.time), ", written into the plan file")
    end

    print_table(rows, chk)
    println("\nthe results were written into ", PLAN_PATH)
end

# This starts the whole thing
main()