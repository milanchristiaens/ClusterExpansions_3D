# Times the plans of the plan file src/utility/onesite_plans.csv (written by find_orders_ome.jl): the sliced contraction `contract_onesite` with a given contraction tree and given cut bonds, each plan in a fresh child julia process.
# The times go into the plan file and, with all repetitions, into results.csv of a new run folder in data/contraction_plans/benchmark_contract_onesite/, together with the plots.

using Printf, LinearAlgebra
using TensorKit
using ClusterExpansions         # Import this full package we are working on
const CE = ClusterExpansions

# ======================================================================================================================
# SETTINGS:
# ======================================================================================================================
# PROFILE: :laptop (default), :hpc or :validate (the Hortense job scripts set CE_BENCH_PROFILE)
#   :laptop    the (11, 5, 3) plans for RAM budgets up to 3 GiB, both allocators (machine "laptop")
#   :hpc       Hortense (4 cores), the 3 GiB plans of the candidates of the CTMRG3D dimension pilot, reuse allocator (the default of contract_onesite_auto)
#   :hpc29     the same sets on a quarter node (16 cores), the 29 GiB plans
#   :validate  Hortense (4 cores), three sets (low, medium, high), only the current tree and the ome-sc tree, both allocators; run once with the mock and once with the real environment (CE_BENCH_ENV=real)
#   sources: the tree kinds that are timed ("current", "opt-true", "ome-flops", "ome-sc"); nothing = all
# The time of a row goes into the plan file together with the machine (column machine), so the times of the two machines do not overwrite each other
const PROFILE = Symbol(get(ENV, "CE_BENCH_PROFILE", "laptop"))
const PROFILES = Dict(
    :laptop   => (; machine = "laptop",   ram_min = 0, ram_max = 3, dims = [(11, 5, 3)],                         allocators = ["reuse", "manual"], max_hours = 10,
                  sources = nothing, run_dir = ""),
    :hpc      => (; machine = "hortense", ram_min = 3, ram_max = 3, dims = [(8, 4, 3), (10, 5, 3), (12, 4, 3), (12, 5, 3), (16, 4, 3), (16, 5, 3), (16, 6, 3), (18, 6, 3)],
                  allocators = ["reuse"], max_hours = 16,
                  sources = nothing, run_dir = ""),
    :hpc29    => (; machine = "hortense", ram_min = 29, ram_max = 29, dims = [(8, 4, 3), (10, 5, 3), (12, 4, 3), (12, 5, 3), (16, 4, 3), (16, 5, 3), (16, 6, 3), (18, 6, 3)],
                  allocators = ["reuse"], max_hours = 16,
                  sources = nothing, run_dir = ""),
    :validate => (; machine = "hortense", ram_min = 3, ram_max = 3, dims = [(8, 4, 3), (11, 5, 3), (18, 6, 3)],  allocators = ["reuse", "manual"], max_hours = 13,
                  sources = ["current", "ome-sc"], run_dir = ""),
)
haskey(PROFILES, PROFILE) || error("unknown profile :$PROFILE (CE_BENCH_PROFILE): use laptop, hpc, hpc29 or validate")
const P = PROFILES[PROFILE]
const MACHINE    = P.machine                  # written into the column machine of every row that gets a time
const PLAN_FILE  = CE.ONESITE_PLAN_FILE   # plan file that is read and updated (the same constant is in find_orders_ome.jl)
# nothing = mock environment (time_s); env_loader_example.jl or env_loader_pepo.jl in development/contraction_plans = real environment (time_real_s)
const ENV_LOADER = get(ENV, "CE_BENCH_ENV", "mock") == "real" ? "development/contraction_plans/env_loader_pepo.jl" : nothing
const MOCK_SEED  = 1                      # seed of the random mock environment: the same seed gives the same environment for all plans of the same dimensions
const T_MOCK     = Float64                # element type of the mock environment (the CE Ising PEPOs are Float64); only the plans made for it (column eltype) are timed
const REPS       = 2                      # timed repetitions of the contraction after the warm-up (the fastest is used; all are in results.csv)
const ALLOCATORS = P.allocators           # the allocators whose rows are timed (missing rows are added first, see above): ["reuse"], ["manual"] or both
const FORCE      = true                   # true = run also if the estimated memory exceeds 85 % of the free memory
const RAM_MIN_GIB = P.ram_min             # only plans made for a RAM budget (column ram_gib of the plan file) of at least RAM_MIN_GIB ...
const RAM_BUDGET_GIB = P.ram_max          # ... and at most RAM_BUDGET_GIB are timed (laptop 0-3, Hortense 3)
const DIMS = P.dims                       # nothing = all dimensions of the plan file; or a list such as [(8, 4, 3), (12, 4, 3)] for (chi_env, chi_peps, D)
const SOURCES = P.sources                 # nothing = all tree kinds; or a list such as ["current", "ome-sc"] (tree_kind of tools/benchmark_plots.jl)
const MAX_FLOPS  = 1.0e15                 # plans whose model flops (all slices together) exceed this are not timed (Inf = no limit)
const MAX_SLICES = 20_000                 # plans with more slices than this are not timed (Inf = no limit)
const MAX_HOURS  = P.max_hours            # no new group of dimensions is started if it is predicted not to finish within this many hours after the start (see above)
const RUN_DIR    = P.run_dir              # "" = new run folder; the NAME of an existing run folder (for example "20261111_2230") = continue that run / make its plots again
# ======================================================================================================================

# The plots and the results.csv functions are only needed by the parent process (loading Plots takes long: the children do not do it)
any(==("--child"), ARGS) || include(joinpath(@__DIR__, "tools", "benchmark_plots.jl"))

# WHAT IS MEASURED (the plan file gets: time, machine, rss_gib, pool_gib, sys_s, minflt, check, note; results.csv has all repetitions and more)
#   time: wall time of the whole sliced contraction (fastest of REPS, GC.gc() before each), without compilation
#   model_*, tracked_gib: cost model and peak bytes of the temporaries of one slice (screen only)
#   rss_gib: peak resident memory of the whole child process; pool_gib: what the ReuseAllocator asked the OS for at its high-water mark
#   sys_s, minflt: Linux only, system CPU seconds and minor page faults per repetition
#   check: relative deviation of the value from the first plan of the same dimensions; env_fp: fingerprint of the environment (the same for all plans of the same dimensions)
# The allocator is created once per plan, so repetition 1 also pays for growing the pool of the reuse allocator

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
        # a loader for any dimensions (env_loader_pepo.jl) takes them; a loader of one environment (env_loader_example.jl) takes nothing
        return loader_takes_dims(f) ? Base.invokelatest(f, Tuple(dims)) : Base.invokelatest(f)
    end
    # If no real env was given, we build a random mock env (seeded: the same dims and the same MOCK_SEED always give the same tensors)
    return CE.onesite_mock_env(ℂ^dims[1], ℂ^dims[2], ℂ^dims[3]; T = T_MOCK, seed = MOCK_SEED)
end

# Does the loader take the dimensions (onesite_load_env(dims))?
loader_takes_dims(f) = Base.invokelatest(applicable, f, (1, 1, 1))

# A number that identifies the environment: the sum of the norms of all 27 tensors and M. Plans of the same dimensions must give exactly the same number.
env_fingerprint(env, M) = sum(norm, env.C) + sum(norm, env.E) + sum(norm, env.A) + norm(M)

# The rows of the plan file that were made for the bond dimensions d (the 54 numbers of CE.onesite_dims) and the element type T, as a Vector of
# (name, allocator, tree, cuts, source, ram_gib, time_s, time_real_s). The name is SOURCE#N (N = number of the row among the rows for these dims, this element type
# and the same allocator, whatever their RAM budget), so parent and child process find the same row (with the name AND the allocator).
function plan_rows(d, T)
    all = CE.onesite_read_plans(PLAN_PATH)
    out = NamedTuple[]

    # For all allocator that occur in the file ("reuse" and "manual")
    for a in unique(r.allocator for r in all)
        # Collect the indices i of the rows which satisfy: that they have dimensions d, element type T and allocator a
        keep = [i for (i, r) in enumerate(all) if CE.onesite_dims(r.chi_env, r.chi_peps, r.D) == d && r.eltype == string(T) && r.allocator == a]
        append!(out, [(name = string(all[i].source, "#", n), allocator = a, tree = all[i].tree, cuts = all[i].cuts, source = all[i].source, ram_gib = all[i].ram_gib,
                       time_s = all[i].time_s, time_real_s = all[i].time_real_s, line = i) for (n, i) in enumerate(keep)])
    end
    # in the order of the file (so the reuse row and the manual copy right below it are timed one after the other)
    return sort!(out; by = r -> r.line)
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

# Create a fresh allocator from its name ("reuse" or "manual")
make_alloc(kind) = kind == "reuse" ? CE.ReuseAllocator() : CE.ManualAllocator()


# CHILD: runs ONE plan in a fresh processor and prints a result line on stdout
# o = the options of the child: plan = NAME of the plan (SOURCE#N), alloc = its allocator, dims = chi_env,chi_peps,D (for the mock environment; the real environment brings its own)
function run_child(o)
    name = o["plan"]
    dims = parse.(Int, split(o["dims"], ','))
    akind = o["alloc"]

    # Build the mock/real environment
    env, M = load_env(dims)
    # The fingerprint of the environment (the parent checks that all plans of the same dimensions have the same one)
    fp = env_fingerprint(env, M)
    # define the vector of 54 bond dimensions and the element type, read from the actual tensors
    d = CE.onesite_dims(env, M)
    T = CE.onesite_eltype(env, M)

    # The plan: its tree, re-ordered such that the peak of the sliced network is minimal, and its cuts
    orders = plan_rows(d, T)
    i = findfirst(r -> r.name == name && r.allocator == akind, orders)
    i === nothing && error("no plan named $name with allocator $akind for these dimensions in $PLAN_PATH")
    cuts = orders[i].cuts
    kernel = CE.onesite_reorder(orders[i].tree, d; cuts)          # a generated kernel: the model tree is the kernel's tree

    # Turn each cut label into label=>dimension: e.g. [:L5 => 8, :L4 => 5, :A8 => 5]
    cutp = CE.onesite_full_cuts(env, M, cuts)
    # Define the resulting number of slices: products of piece count
    ns = CE.onesite_nslices(env, M, cutp)

    # Define the model cost of this plan (flops, peak elements, slices)
    # And define base as the model cost of the current tree unsliced
    cost = CE.onesite_cost(kernel, d; cuts); base = CE.onesite_cost(CE.TREE_CURRENT, d)

    # Now transform the cost.peak to a number in GiB (sizeof(T) bytes per element)
    mpeak = GiB(cost.peak * sizeof(T))

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
        e2, M2 = CE.onesite_mock_env(ℂ^2, ℂ^2, ℂ^2; T = T_MOCK)
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
function bench_row(name, alloc, g)
    # Build the command line that will start a new Julia process
    # it uses same: Julia, project, thread count as us.
    # it adds --child, so now it will get filtered out above
    # it also adds: the plan name and the dimensions
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$(Threads.nthreads()) $(abspath(PROGRAM_FILE)) --child --plan=$name --alloc=$alloc --dims=$(g[1]),$(g[2]),$(g[3])`

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
    empty_row(msg) = (name = name, cuts = "", alloc = alloc, slices = "", flops_rel = nan, peak = nan, time = nan, per_slice = nan, tracked = nan,
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
    @printf("\n%-8s %-22s %-7s %-24s %6s %10s %10s %9s %9s %9s %8s\n", "dims", "plan", "alloc", "cuts", "slices", "model flops", "time s", "reps s", "rss GiB", "pool GiB", "check")

    # Loop over all rows
    for r in rows
        dims = string(r.chi_env, ",", r.chi_peps, ",", r.D)
        # If the row crashed/was skipped than there is no time; we only print the plan, cuts and the note
        if !isfinite(r.tmin)
            @printf("%-8s %-22s %-7s %-24s %s\n", dims, r.name, r.alloc, r.cuts[1:min(end, 24)], r.note); continue
        end

        # Now we print the finished row with the check value
        @printf("%-8s %-22s %-7s %-24s %6.0f %10.3e %10.2f %9s %9.3f %9.3f %8.1e\n", dims, r.name, r.alloc, r.cuts[1:min(end, 24)], r.slices, r.flops, r.tmin,
                join(round.(r.times; digits = 1), "/"), r.rss, r.pool, r.check)
    end
end

# Print an environment header
function print_system_info()
    println("Julia ", VERSION, ", threads ", Threads.nthreads(), ", BLAS threads ", BLAS.get_num_threads(), " (", blas_name(), ")",
            ", total memory ", round(GiB(Sys.total_memory()), digits = 1), " GiB, free ", round(GiB(Sys.free_memory()), digits = 1), " GiB")
    println("TensorKit ", pkgversion(TensorKit), ", TensorOperations ", pkgversion(CE.TensorOperations),
            "   environment: ", ENV_LOADER === nothing ? "mock (dense, random, seed $MOCK_SEED, $T_MOCK)" : "real (" * ENV_LOADER * ")", ", allocators ", join(ALLOCATORS, "/"), ", reps ", REPS)
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
        println(io, "ENV_LOADER = ", ENV_LOADER, ", MOCK_SEED = ", MOCK_SEED, ", T_MOCK = ", T_MOCK, ", REPS = ", REPS, ", ALLOCATORS = ", ALLOCATORS, ", FORCE = ", FORCE)
        println(io, "PROFILE = ", PROFILE, ", MACHINE = ", MACHINE, ", host = ", gethostname())
        println(io, "RAM_MIN_GIB = ", RAM_MIN_GIB, ", RAM_BUDGET_GIB = ", RAM_BUDGET_GIB, ", DIMS = ", DIMS, ", MAX_HOURS = ", MAX_HOURS, ", RUN_DIR = \"", RUN_DIR, "\"")
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
    all(a -> a in ("reuse", "manual"), ALLOCATORS) || error("ALLOCATORS may only contain \"reuse\" and \"manual\", got $ALLOCATORS")
    
    # Every plan gets a row for every allocator of ALLOCATORS (same tree and cuts, no time yet), see onesite_add_allocator_copies()
    for a in ALLOCATORS
        n = CE.onesite_add_allocator_copies(PLAN_PATH, Symbol(a))
        n > 0 && println("added $n rows with allocator $a to the plan file (copies of the plans that had none)")
    end
    plans = CE.onesite_read_plans(PLAN_PATH)
    isempty(plans) && error("no plans in $PLAN_PATH: make them first with development/contraction_plans/find_orders_ome.jl")

    # The element type of the environment: T_MOCK, or the one of the real environment; only plans made for it are timed
    per_dims = false     # true: a real environment for every set (env_loader_pepo.jl)
    if ENV_LOADER === nothing
        T = T_MOCK
    else
        Base.include(Main, abspath(ENV_LOADER))
        per_dims = loader_takes_dims(Base.invokelatest(getproperty, Main, :onesite_load_env))
        per_dims && DIMS === nothing && error("$ENV_LOADER builds an environment for every set: set DIMS (the profile) to the sets")
        # the environment of the first set (per_dims) or the one environment: its element type (and its dimensions)
        env, M = load_env(per_dims ? DIMS[1] : nothing)
        T = CE.onesite_eltype(env, M)
        d_env = CE.onesite_dims(env, M)
    end
    # The environment as written into results.csv; only the rows of results.csv with this environment count as done
    envtag = ENV_LOADER === nothing ? "mock seed $MOCK_SEED $T" : "real $T"

    # The groups of dimensions that are timed: all of them for the mock environment (or those of DIMS); for the real environment only the one with its dimensions
    groups = unique((r.chi_env, r.chi_peps, r.D) for r in plans if r.eltype == string(T))
    DIMS === nothing || (groups = filter(g -> g in DIMS, groups))
    if ENV_LOADER !== nothing && !per_dims
        groups = filter(g -> CE.onesite_dims(g...) == d_env, groups)
        if isempty(groups)
            println("no plan in $PLAN_PATH has the bond dimensions and the element type ($T) of the environment of $ENV_LOADER: make plans for them with development/contraction_plans/find_orders_ome.jl")
            return
        end
    end
    isempty(groups) && (println("no plan in $PLAN_PATH for element type $T has dimensions in DIMS = $DIMS"); return)
    # Cheapest dimensions first (model flops of the current tree, unsliced): if the time runs out, the most expensive dimensions are the ones that are left out
    sort!(groups; by = g -> CE.onesite_cost(CE.TREE_CURRENT, CE.onesite_dims(g...)).flops)
    # A real environment for every set (env_loader_pepo.jl): build them all now (each is saved once and read by every plan of its set)
    if per_dims
        for g in groups
            load_env(g)
        end
        env = M = nothing; GC.gc()
    end

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
        (isfinite(r.tmin) && r.env == envtag) || continue
        g = (r.chi_env, r.chi_peps, r.D)
        haskey(refs, g) || (refs[g] = complex(r.re, r.im); fps[g] = r.fp)
    end
    # Define a Dict which given row info as key, returns Boolean which tells you if it was already timed
    isdone(g, name, alloc) = any(r -> (r.chi_env, r.chi_peps, r.D) == g && r.name == name && r.alloc == alloc && r.env == envtag && isfinite(r.tmin), rows)

    println("plan file: ", PLAN_PATH, "  (", length(plans), " plans)")
    println("run folder: ", outdir, "  (", nold, " results in it already)")
    println("selected: plans for RAM budget ", RAM_MIN_GIB, "-", RAM_BUDGET_GIB, " GiB and element type ", T, " (machine ", MACHINE, "), ", length(groups), " dimension sets, cheapest first: ", join(("(" * join(g, ",") * ")" for g in groups), " "))
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
        for r in plan_rows(d, T)
            (isfinite(r.ram_gib) && RAM_MIN_GIB <= r.ram_gib <= RAM_BUDGET_GIB) || continue
            r.allocator in ALLOCATORS || continue
            (SOURCES === nothing || tree_kind(r.source) in SOURCES) || continue
            isdone(g, r.name, r.allocator) && continue
            cost = CE.onesite_cost(CE.onesite_reorder(r.tree, d; cuts = r.cuts), d; cuts = r.cuts)
            # we skip plans that are predicted to take far too long (for example an OME slicing with very many slices)
            if cost.flops > MAX_FLOPS || cost.slices > MAX_SLICES
                println(stderr, "($(join(g, ","))) $(r.name) ($(r.allocator)) not timed: $(round(Int, cost.slices)) slices, $(round(cost.flops, sigdigits = 3)) flops (MAX_SLICES, MAX_FLOPS)")
                continue
            end
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
            println(stderr, "\n[($(join(g, ","))) plan $n/$(length(todo)), $(round((time() - t_start) / 60, digits = 1)) min since the start] $(r.name) ($(r.allocator)), $(length(r.cuts)) cuts"); flush(stderr)
            # Identify the row in the CSV corresponding to current job
            key = (chi_env = g[1], chi_peps = g[2], D = g[3], eltype = string(T), tree = r.tree, cuts = r.cuts, allocator = r.allocator)
            # What is the same for every outcome of this plan
            ident = (chi_env = g[1], chi_peps = g[2], D = g[3], name = r.name, source = r.source, ram_gib = r.ram_gib, cuts = join(r.cuts, " "),
                     slices = cost.slices, flops = cost.flops, peak = GiB(cost.peak * sizeof(T)), env = envtag,
                     blas = BLAS.get_num_threads())
            # A row without a time (skipped or failed)
            norow(msg) = merge(ident, (times = Float64[], tmin = NaN, tracked = NaN, rss = NaN, pool = NaN, alloc = r.allocator, fp = NaN, re = NaN, im = NaN, check = NaN, note = msg))
            function addrow(row)
                push!(rows, row)
                try
                    append_bench_row(resultspath, row)
                catch e
                    @warn "could not write results.csv (the row is kept in memory, the plots still work)" exception = e
                end
            end

            # Skip a plan that needs more memory than is free, without starting a process (the child has the same guard)
            est = CE.ONESITE_RAM_RULES.baseline_gib + CE.ONESITE_RAM_RULES[Symbol(r.allocator)] * GiB(cost.peak * sizeof(T))
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
            t_wall = @elapsed res = bench_row(r.name, r.allocator, g)

            # If there is a message, this means that child skipped the plan or crashed
            if res.msg != ""
                addrow(norow(res.msg))
                # failed or skipped by the child: the note says why
                planfile_update(key, (note = res.msg,), envkind)
                continue
            end
            # a complete line without any timing : we keep it as a failed row
            if isempty(res.times)
                msg = "failed: the child returned no timings"
                addrow(norow(msg))
                planfile_update(key, (note = msg,), envkind)
                println(stderr, "    ", msg)
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

            # Write the measurements back into THE row of the plan file (time = fastest repetition, rss_gib, pool_gib, sys_s, minflt, check, note)
            planfile_update(key, (time_s = rnd(minimum(res.times)), rss_gib = rnd(res.rss), pool_gib = rnd(res.pool),
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
        CE.onesite_update_plans(PLAN_PATH, [merge(key, vals, (; machine = MACHINE))]; env = envkind)
    catch e
        @warn "could not update the plan file (the result is in results.csv)" exception = e
    end
    return nothing
end

# This starts the whole thing
main()