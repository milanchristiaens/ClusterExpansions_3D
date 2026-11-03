# This file benchmarks time and memory usage of `contract_onesite` variants: current bracketing vs. other contraction trees, 
# with 0-3 (or more) sliced bonds, with the default allocator (ManualAllocator) or the ReuseAllocator.  Each case runs in a FRESH child
# julia process, so that `Sys.maxrss()` (peak resident memory) belongs to that case only and no allocator pool survives.
#
# USAGE; (-O1 (-02=default = 'more optimization' for compiling the code just in time) keeps the compilation of the 27-tensor kernels shorter (-02 takes longer than -01 in this case)
# , -t 8 = 8 number of cores):
#   julia +1.12 -O1 -t 8 --project=. development/benchmark_contract_onesite.jl --list --dims=8,5,5
#       -> only prints the cost-model prediction of every case (flops, peak memory, slices, estimated RSS); contracts nothing
#   julia +1.12 -O1 -t 8 --project=. development/benchmark_contract_onesite.jl --dims=6,4,4
#       -> 'default' (= current:2, current:3 and fast:3, each with both allocators manual/reuse) cases on a mock environment of random dense tensors
#   julia +1.12 -O1 -t 8 --project=. development/benchmark_contract_onesite.jl --dims=6,4,4 --orders=development/orders_6_4_4.tsv --select="slice\+reconf/T2[46]"
#       -> Use same mock environment, but this time the different cases come from contengra's output:
#          --orders=... reads the different rows of the TSV written by contengra, --select=REGEX
#          now we only keep rows that match with REGEX: in this case we have REGEX = "slice\+reconf/T2[46]", this corresponds with:
#          technique "slicing with reoptimalisation", with target for largest intermediate = T2[46] = between 2^24 and 2^26 (ComplexF64: 0.25-1 GiB)
#   julia +1.12 -O1 -t 8 --project=. development/benchmark_contract_onesite.jl --env-loader=development/env_loader_example.jl --cases=current:3
#       -> Use the real environment via loader file and contract using current contraction + 3 slices

# OPTIONS (all --key=value)
#   --dims=6,4,4           chi_env,chi_peps,D of the mock environment (dense ℂ^χ, ComplexF64)           [default 6,4,4]
#   --env-loader=FILE      instead of the mock environment use a real one: a file defining `onesite_load_env() -> (env, M)`           (see env_loader_example.jl)
#   --cases=LIST           comma list of  current:K  fast:K  (K = 0..3: amount of cuts, specific cuts correspond to defaults)   [default current:2,current:3,fast:3]
#                          current:minpeak:K  fast:minpeak:K  (K cuts chosen by exhaustive search for minimal model peak)
#                          orders:NAME        a row of the --orders file;   all = current:0..3 and fast:0..3
#   --orders=FILE --select=REGEX   TSV from find_orders_cotengra.py; rows whose name matches REGEX are added as cases
#   --allocator=manual|reuse|both                                                                          [default both]
#   --reps=N               timed repetitions of contraction after the warm-up (min is reported)                           [default 2]
#   --slices=N             time only the first N slices and extrapolate to all (flagged 'extrapolated'; no value check)
#   --blas-threads=N       BLAS threads (default: unchanged)
#   --csv=FILE             results file                                                                    [default benchmark_contract_onesite.csv]
#   --force                run even if the estimated memory exceeds 85 % of the free memory

# WHAT IS MEASURED
#   time_s       wall time of the whole sliced contraction (min over reps, GC.gc() before each); NOT including compilation
#   model_*      cost model (dense; src/utility/contract_onesite_orders.jl): flops relative to current unsliced, peak of the intermediates in GiB
#   tracked_gib  peak bytes of ALL temporaries of one slice (TrackingAllocator; includes TensorKit's permutation copies)
#   rss_gib      peak resident memory of the whole child process (Sys.maxrss) -- includes Julia/TensorKit (~1-2 GiB), the environment
#                tensors, garbage not yet collected, and (reuse) the pool
#   pool_gib     reuse only: what the ReuseAllocator asked the OS for at its high-water mark
#   sys_s, minflt  Linux only (/proc/self/stat): system CPU seconds and minor page faults during the timed reps
#   check        relative deviation of the value from the first full-slice case (should be ~1e-10 or smaller; not for --slices)


using Printf, LinearAlgebra, Random
using TensorKit
using ClusterExpansions         # Import this full package we are working on
const CE = ClusterExpansions

# Turn the command-line arguments (--key=value) into a Dict of strings
function parse_cli(args)
    o = Dict{String,String}()
    for a in args
        # Loop over all arguments, and skip anything that doesn't start with --
        startswith(a, "--") || continue

        # Drop the leading -- and split at the first = only
        kv = split(a[3:end], '='; limit = 2)

        # Add this to Dic: key -> value (if no =, like in --list, then list gets value=True)
        o[String(kv[1])] = length(kv) == 2 ? String(kv[2]) : "true"
    end
    return o
end

# Convert bytes to GiB
GiB(x) = x / 2^30
# SEP = tab
const SEP = '\t'

# return (env, M), either from real env or from random mock env
function load_env(o)
    # If in settings dict o, there is (env-loader -> FILE) = as input "--env-loader=FILE"
    if haskey(o, "env-loader")
        # Now we evaluate this file inside Main
        Base.include(Main, abspath(o["env-loader"]))
        # And we run it in Main, and retrieve the answer
        return Base.invokelatest(getproperty(Main, :onesite_load_env))
    end

    # If no real env was given, then we read --dims (default = 6, 4, 4)
    # And retrieve the 3 integeres
    χe, χp, D = parse.(Int, split(get(o, "dims", "6,4,4"), ','))

    # Now we can build a random env with this dimensions
    return CE.onesite_mock_env(ℂ^χe, ℂ^χp, ℂ^D)
end


# returns (kernel, model_tree, cut labels) from specs
function resolve_case(spec::AbstractString, d, orders)
    # spec = string such as "current:2"
    # d = vector of 54 bond dimensions
    # orders = list of rows read from TSV
    
    # Split the specs: e.g. "fast:minpeak:3" → ["fast","minpeak","3"]
    parts = split(spec, ':')

    # If the first part of spec refers to a reference tree:
    if parts[1] in ("current", "fast")
        # Choose which one
        tree = parts[1] == "current" ? CE.TREE_CURRENT : CE.TREE_FAST

        # If form is "current/fast:K"
        if length(parts) == 2
            # In 'src/utility/contract_onesite_orders.jl' we defined some default cut orders (in case of 8, 5, 5) for the current and fast tree
            # for 0-3 cuts: e.g.     current = [Symbol[], [:L5], [:L5, :L4], [:L5, :L4, :A8]], fast = ...
            # Now we have to find the correct one, so we take the (K+1)th element of the list
            # cuts will be e.g. ":L5"
            cuts = getfield(CE.ONESITE_TESTED_CUTS, Symbol(parts[1]))[parse(Int, parts[2]) + 1]
        
        # If form is current:minpeak:K
        elseif length(parts) == 3 && parts[2] == "minpeak"
            # Extract k
            k = parse(Int, parts[3])
            # if k=0 there are no cuts, otherwise run the exhhaustive search over all k-subsets of 54 bonds
            # given the limit on the model peak, and then it takes the .cuts field
            cuts = k == 0 ? Symbol[] : CE.onesite_best_cuts(tree, d, k).cuts
        
        # Any other shape is an error
        else
            error("bad case $spec")
        end

        # Return: :current/:fast, tree for cost model, the cuts as a vector
        return Symbol(parts[1]), tree, collect(Symbol, cuts)
    
    # If form is orders:NAME
    elseif parts[1] == "orders"
        # join everything after "order" back together (if : was in NAME)
        name = join(parts[2:end], ':')

        # Search the TSV row (in orders) with that name
        i = findfirst(r -> r.name == name, orders)
        # if no row found return error
        i === nothing && error("no row named $name in the orders file")
        # Otherwise return the full TSV row
        r = orders[i]
        
        # Now we reorder the children of each pair so that the peak of the sliced network is minimal
        # And we return the tree (in form of a tuple), nothing, and the cuts 
        return CE.onesite_reorder(r.tree, d; cuts = r.cuts), nothing, r.cuts      # model tree = the re-ordered tree itself
    end

    # If unknown prefix
    error("bad case $spec")
end


# LINUX ONLY: Read the process's minor page faults and CPU times, on windows tthis returns nothing
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

# Create a fresh allocator from the string given by --alocator
make_alloc(kind) = kind == "reuse" ? CE.ReuseAllocator() : CE.ManualAllocator()

# This function runs ONE case with ONE allocator in a fresh processor
# prints a results line on stdout
function run_child(o)
    # spec is the case name: e.g. current:2. akind is "manual" or "reuse"
    spec, akind = o["case"], o["allocator"]
    # Reps = number of timed repetitions (default = 2)
    reps = parse(Int, get(o, "reps", "2"))
    # max number of slices to run (default = all slices), 
    maxs = haskey(o, "slices") ? parse(Int, o["slices"]) : typemax(Int)

    # Set the BLAS thread count only if requested
    haskey(o, "blas-threads") && BLAS.set_num_threads(parse(Int, o["blas-threads"]))
    
    # Read the contengra TSV (name, cuts, tree) or an empty list if there is none\
    # And return a vector of named tuples: Vector{NamedTuple}:  orders = (name = "current/unsliced", cuts = Symbol[], tree = (1, (2, (3, ...))), ctg_flops = 2.272612e12, ctg_peak = 2.602e9, slices = 1)
    orders = haskey(o, "orders") ? CE.onesite_read_orders(o["orders"]) : NamedTuple[]
    
    # Build the mock/real environment
    env, M = load_env(o)
    # define the vector of 54 bond dimensions, read from the actual tensors
    d = CE.onesite_dims(env, M)

    # Decode the case name
    kernel, mtree, cuts = resolve_case(spec, d, orders)
    # if orders: than we put mtree to nothing, and then we use the reorderd kernel as the model 
    if mtree === nothing; mtree = kernel; end                      # generated kernel: the model tree is the kernel's tree
    
    # Turn each cut label into label=>dimension: e.g. [:L5 => 8, :L4 => 5, :A8 => 5]
    cutp = CE.onesite_full_cuts(env, M, cuts)
    # Define the resulting number of slices: products of piece count
    ns = CE.onesite_nslices(env, M, cutp)

    # Define the model cost of this case (flops, peak elements, slices)
    # And define base as the model cost of the current tree unsliced
    cost = CE.onesite_cost(mtree, d; cuts); base = CE.onesite_cost(CE.TREE_CURRENT, d)
    
    # Define the element size: 16 bytes for a single ComplexF64 element
    esize = sizeof(scalartype(M))
    # Now transform the cost.peak to a number in GiB
    mpeak = GiB(cost.peak * esize)

    # Now we define a rule of thumb, for calculating wether or not we have currently enough free random
    # rule of thumb: we need 1.6 x model peak (2.0x model peak if reuse) plus approx 1.6GiB for Julia/TensorKit
    est = (akind == "reuse" ? 2.0 : 1.6) * mpeak + 1.6                       # rule of thumb from my sandbox runs; not a guarantee
    # Compare with currently free RAM
    free = GiB(Sys.free_memory())

    # Already construct the first six output columns
    base_cols = (spec, join(cuts, ","), akind, round(Int, ns), cost.flops / base.flops, mpeak)
    
    # Memory guard: if we don't have enough ram
    if est > 0.85 * free && !haskey(o, "force")
        # Then it just prints a 7-field line and stops
        println(join((base_cols..., "skipped: estimated $(round(est, digits = 1)) GiB > 85% of free $(round(free, digits = 1)) GiB (use --force)"), SEP))
        return
    end

    # Resolve whether we have to make a new contraction tree/kernel (if kernel isa tuple), or if we can just use :current or :fast
    kern = kernel isa Tuple ? CE.onesite_make_kernel(kernel) : kernel          # resolve ONCE (a generated kernel compiles on its first call)


    # every run goes through a TrackingAllocator (a few hundred dictionary operations per slice: negligible against seconds per slice)
    # so that the kernel is compiled only once per case.
    # warm-up: compile everything.  mock env: tiny env (same Vector{Any} specialisation); real env: one slice of the real thing
    
    # Now we print that the long compile phase has started
    println(stderr, "[$spec / $akind] compiling (first call) ..."); flush(stderr)
    
    # Warm up (not timed yet)
    if haskey(o, "env-loader")
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

    # For all reperations that were asked (default=2)
    for r in 1:reps
        # Collect garbage first
        GC.gc()
        # Time the full sliced contraction, z keeps the contraction value (for correctness check)
        push!(ts, @elapsed z = CE.contract_onesite_sliced(env, M, cutp; kernel = kern, allocator = ta, max_slices = maxs))
        # Print some info
        println(stderr, "[$spec / $akind] rep $r: $(round(ts[end], digits = 2)) s"); flush(stderr)
    end

    # Take a snapchot after the loop and the peak RSS after it
    ps1 = procstat()
    rss1 = GiB(Sys.maxrss())

    # The peak of all live temporaries (any slice, all reps)
    tracked = GiB(ta.peak_bytes)
    # The amount of memory that the reuse allocator requested from OS (also counting pooled memory)
    pool = ta.inner isa CE.ReuseAllocator ? GiB(ta.inner.peak_os_bytes) : NaN
    # free the pool (if reuse)
    ta.inner isa CE.ReuseAllocator && CE.release!(ta.inner)
    
    # fill = True if all slices ran
    full = maxs >= ns
    # tmin = best timing for single rep
    tmin = minimum(ts)
    # if only the first maxs slices ran, then we extrapolate tmin to ns slices
    tt = full ? tmin : tmin / min(maxs, ns) * ns

    # LINUX only, WINDOW: NAN
    rf = ps0 === nothing ? NaN : (ps1.minflt - ps0.minflt) / reps
    rs = ps0 === nothing ? NaN : (ps1.sys - ps0.sys) / reps
    
    # Print the final result line as 17 tab seperated fields, same order as CSV header:
    # six base columns, time, time per slice, tracked, rss, rss before, pool, sys, minflt, flag, re, im.
    println(join((base_cols..., tt, tt / ns, tracked, rss1, rss0, pool, rs, rf, full ? "" : "extrapolated",
                  real(z), imag(z)), SEP))
end

# Turns --cases and --select into a flat list of case names
# e.g. --cases=current:0,fast:1 -> ["current:0", "fast:1"]
# e.g. --orders=development/orders_6_4_4.tsv --select="slice\+reconf/T26" -> ["orders:current/slice+reconf/T26", "orders:ctg-flops/slice+reconf/T26", "orders:myfast/slice+reconf/T26", ...]
# e.g. --cases=current:2 --orders=... --select="slice\+reconf/T26" -> ["current:2", "orders:current/slice+reconf/T26", "orders:ctg-flops/slice+reconf/T26", ...]
function expand_cases(o, orders)
    specs = String[]
    # For all --cases (if no take default=urrent:2,current:3,fast:3, EXCEPT if --select is given then default=empty)
    #  keepempty = false drops empty pieces
    for c in split(get(o, "cases", haskey(o, "select") ? "" : "current:2,current:3,fast:3"), ','; keepempty = false)
        # if c==all then append all 8 built-in cases
        # if not, then add string(c) to specs
        c == "all" ? append!(specs, ["current:$k" for k in 0:3], ["fast:$k" for k in 0:3]) : push!(specs, String(c))
    end

    # If --select
    if haskey(o, "select")
        # Check every TSV row, to see whose name matches with regex
        rx = Regex(o["select"])
        # add to specs: orders:NAME
        append!(specs, ["orders:" * r.name for r in orders if occursin(rx, r.name)])
    end
    return specs
end


# Main function, if in child mode: runs one case
# Otherwise it is parent: prints the system info, model tablee, spawns one child per (case,allocator)
# collects the lines, writes the CSV and table
function main()
    # Define Dict o: holding the command-line options of the current process, input is ARGS
    # e.g. julia +1.12 -O1 -t 8 --project=. development/benchmark_contract_onesite.jl --dims=6,4,4 --reps=3 --list
    # ARGS = ["--dims=6,4,4", "--reps=3", "--list"], o = Dict("dims" => "6,4,4", "reps" => "3", "list" => "true")
    o = parse_cli(ARGS)
    
    # Check if the process is a child: this only happens if it contains --child, this only gets added by the script itself
    # This function is ran first as parent, then once for every (case,allocator) as a child
    # if child, then we run run_child(o)
    haskey(o, "child") && return run_child(o)

    # if there are orders, read the order TSV and build the case list
    orders = haskey(o, "orders") ? CE.onesite_read_orders(o["orders"]) : NamedTuple[]
    # Add these cases to specs
    specs = expand_cases(o, orders)
    # Define the used allocators (default = both)
    akinds = get(o, "allocator", "both") == "both" ? ["manual", "reuse"] : [o["allocator"]]
    
    # Print an environment header
    println("Julia ", VERSION, ", threads ", Threads.nthreads(), ", BLAS threads ", BLAS.get_num_threads(), " (", blas_name(), ")",
            ", total memory ", round(GiB(Sys.total_memory()), digits = 1), " GiB, free ", round(GiB(Sys.free_memory()), digits = 1), " GiB")
    println("TensorKit ", pkgversion(TensorKit), ", TensorOperations ", pkgversion(CE.TensorOperations), "   cases: ", join(specs, ", "))
    
    # if --list: only prints the cost-model prediction of every case (flops, peak memory, slices, estimated RSS); contracts nothing
    if haskey(o, "list")
        # Build the env only to get dimensions d, and calculate its BASE cost given d and current tree
        env, M = load_env(o); d = CE.onesite_dims(env, M); base = CE.onesite_cost(CE.TREE_CURRENT, d)
        
        # Print a header
        @printf("%-34s %-28s %8s %10s %12s\n", "case", "cuts", "slices", "flops/cur", "model peak GiB")
        # Print one model row per case: slices, relative flops, model peak, estimated RSS '1.6x + 1.6 Gb rule)
        for s in specs
            kernel, mt, cuts = resolve_case(s, d, orders); mt === nothing && (mt = kernel)
            c = CE.onesite_cost(mt, d; cuts)
            @printf("%-34s %-28s %8d %10.4f %12.4f   est. RSS %.1f GiB\n", s, join(cuts, ",")[1:min(end, 28)], round(Int, c.slices), c.flops / base.flops,
                    GiB(c.peak * sizeof(scalartype(M))), 1.6 * GiB(c.peak * sizeof(scalartype(M))) + 1.6)
        end
        return
    end

    # preparing the child calls
    # pass goes from Dict o back to the string we inputted in terminal (but we skip some options, which we decide ourselves)
    pass = ["--$k=$v" for (k, v) in o if !(k in ("case", "allocator", "child", "cases", "select", "list"))]
    # Small check if --force exists
    haskey(o, "force") && !any(startswith("--force"), pass) && push!(pass, "--force")
    
    # Create a results file: on --csv= choice or on default
    csv = get(o, "csv", "benchmark_contract_onesite.csv")
    # Define the header/list of 17 columns
    hdr = ["case", "cuts", "allocator", "slices", "model_flops_rel", "model_peak_gib", "time_s", "time_per_slice_s", "tracked_gib",
           "rss_gib", "rss_before_gib", "pool_gib", "sys_s", "minflt", "extrapolated", "re", "im"]
    # Defien an  empty list that will collect all results 
    rows = Vector{Vector{String}}()

    # Running every child in its own worker
    # We loop over every case and allocator
    for s in specs, a in akinds
        # Build the command line that will start a new Julia process
        # it uses same: Julia, project, thread count as us.
        # it adds --child, so now it will get filtered out above
        # it also add: case, allocator and forwarded options 'pass'
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$(Threads.nthreads()) $(abspath(PROGRAM_FILE)) --child --case=$s --allocator=$a $pass`
        
        # Print which case is starting
        println(stderr, "==> ", s, " / ", a); flush(stderr)

        # Now we start the worker, wait untill it is done and return what is printed
        out = try
            strip(read(cmd, String))
        # If the worker crashes (e.g. running out of memory)
        catch e
            # Then we store: "failed: <error>" and the benchmark can continue
            join(("failed: " * sprint(showerror, e),), SEP)
        end

        # We only keep the last printed line of worker's answer (can also be error message or memory skip message)
        line = last(split(out, '\n'))
        # We split it (SEP=tab seperated) into seperate fields f
        f = split(line, SEP)

        # If we have less than 6 fields: the worker crashed
        if length(f) < 6        
            # We store only case name, allocator and error text
            push!(rows, [s, "", a, "", "", "", "", "", "", "", "", "", "", "", line, "", ""])
        # If exactly 7: then the worker skipped the case, because of memory
        elseif length(f) == 7
            # We keep the first six fields and the "skipped: ..." message                     
            push!(rows, [f[1], f[2], f[3], f[4], f[5], f[6], "", "", "", "", "", "", "", "", f[7], "", ""])
        # Otherwise we have a full result
        else
            push!(rows, String.(f))
        end
    end

    # Now we write the CSV file, using the rows
    open(csv, "w") do io
        # First we write the title line hdr
        println(io, join(hdr, ","))
        # Then for each row we put the answers their (we also replace , with ; in the answers to not break the CSV)
        # e.g. cut list (L5, L4) to (L5; L4)
        foreach(r -> println(io, join(replace.(r, "," => ";"), ",")), rows)
    end

    # Now we print the table on screen
    # ref will hold the reference balue (will be first finished full run)
    ref = nothing
    # Print the table header
    @printf("\n%-28s %-24s %-6s %6s %9s %9s %10s %9s %9s %9s %8s\n", "case", "cuts", "alloc", "slices", "flops/cur", "model GiB", "time s", "tracked", "rss GiB", "pool GiB", "check")
    
    # Loop over all roows
    for r in rows
        # If the row crashed/was skipped than the time column is also emptu
        if length(r) < 17 || r[7] == ""
            #" In that case we only print the case, cuts, allocator and message
            @printf("%-28s %-24s %-6s %s\n", r[1], r[2][1:min(end, 24)], r[3], r[15]); continue
        end

        # If row has full results:
        # Now we recreate the contraction number as real and imaginary parts
        z = complex(parse(Float64, r[16]), parse(Float64, r[17]))
        
        # r[15] contains the extrapolated result (in case of not all slices ran)
        # full = True means, case ran fully
        full = r[15] == ""
        
        # Now we perform a check: see if all answers for z correspond to each other
        # we only use full results, and we define ref as the result of the first finished run
        # Then we calculate the relative difference from the reference (this should be tiny)
        chk = full ? (ref === nothing ? (ref = z; 0.0) : abs(z - ref) / abs(ref)) : NaN
        
        # Now we prin the finished row with the check value and if applicable 'extrapolated'
        @printf("%-28s %-24s %-6s %6s %9.3f %9.3f %10.2f %9.3f %9.3f %9.3f %8.1e %s\n", r[1], r[2][1:min(end, 24)], r[3], r[4], parse(Float64, r[5]), parse(Float64, r[6]),
                parse(Float64, r[7]), parse(Float64, r[9]), parse(Float64, r[10]), parse(Float64, r[12]), chk, r[15])
    end

    # If this prints, then the csv file was written
    println("\nwrote ", csv)
end

# This starts the whole thing
main()
