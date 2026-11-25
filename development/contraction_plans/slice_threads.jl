# Looks at why the sliced contraction keeps only about one core busy, for one set of dimensions and TREE_CURRENT.
# Part 1 times the first slices with 1 or 4 Julia threads and 1 or 4 BLAS threads and profiles where the time goes; part 2 compares the
# garbage-collection rules (gc_every, gc_gib); part 3 runs the slices in parallel tasks with one BLAS thread each. Every run is a fresh child process.

using Printf, LinearAlgebra, Profile
using TensorKit
using ClusterExpansions
const CE = ClusterExpansions

# ======================================================================================================================
# SETTINGS (a Hortense job may pass CE_ST_DIMS, CE_ST_HOURS, CE_ST_SUMMARY)
# ======================================================================================================================
const DIMS = Tuple(parse.(Int, split(get(ENV, "CE_ST_DIMS", "12,5,3"), ',')))   # (χenv, χpeps, D)
const MAX_HOURS = parse(Float64, get(ENV, "CE_ST_HOURS", "2.5"))                # no new child is started after this many hours
const SUMMARY_ONLY = get(ENV, "CE_ST_SUMMARY", "")       # name of an existing run folder: only remake its summary
const RAM_GIB = 3                     # RAM budget that decides k_min (as the 3 GiB plans)
const T_MOCK = Float64
const MOCK_SEED = 1
const PART1_SLICES = 48               # part 1 only runs the first slices of the k_min plan
const THREAD_CONFIGS = [(1, 1), (1, 4), (4, 1), (4, 4)]   # (Julia threads, BLAS threads) of part 1; we profile the last one
const GC_MODES = ["every=1", "every=10", "every=0", "gib=0.25"]   # part 2: gc_every = 1, 10, never, and gc_gib = 0.25 GiB
const PAR_GC_GIB = 0.25               # garbage-collection rule of part 3 (gc_gib)
const PAR_RUNS = [(:kmin, 1, 4), (:kmin, 2, 1), (:kmin, 4, 1), (:kmin1, 1, 4), (:kmin1, 2, 1), (:kmin1, 4, 1)]   # (plan, tasks, BLAS threads)
# ======================================================================================================================

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "contraction_plans", "slice_threads"))
const SEP = '\t'
const HEADER = join(("part", "config", "plan", "k", "slices_run", "time_s", "per_slice_s", "busy_cores", "gc_s", "alloc_gib", "rss0_gib", "rss_gib",
                     "pool_gib", "value", "profile", "note"), SEP)
GiB(x) = x / 2^30

function parse_cli(args)
    o = Dict{String,String}()
    for a in args
        startswith(a, "--") || continue
        kv = split(a[3:end], '='; limit = 2)
        o[String(kv[1])] = length(kv) == 2 ? String(kv[2]) : "true"
    end
    return o
end

# CPU seconds (user + system, all threads of this process); Linux only
cpu_seconds() = isfile("/proc/self/stat") ? (p = split(split(read("/proc/self/stat", String), ") ")[end]); (parse(Int, p[12]) + parse(Int, p[13])) / 100) : NaN

# The two plans: k_min (fewest cuts that fit RAM_GIB) and k_min + 1 (fewest flops with at most half the peak)
function plans()
    d = CE.onesite_dims(DIMS...)
    base = CE.onesite_choose_cuts(d; ram_gib = RAM_GIB, T = T_MOCK, allocator = :reuse)
    peak = CE.onesite_cost(CE.onesite_reorder(CE.TREE_CURRENT, d; cuts = base.cuts), d; cuts = base.cuts).peak
    r = CE.onesite_best_cuts(CE.TREE_CURRENT, d, base.k + 1; cap = peak / 2, order = :best)
    r === nothing && (r = CE.onesite_best_cuts(CE.TREE_CURRENT, d, base.k + 1; order = :best))
    return Dict(:kmin => base.cuts, :kmin1 => r.cuts)
end

# The tensors of one slice (as in contract_onesite_sliced)
function slice_tensors(env, M, bcs, js)
    C, E, A, Mj = copy(env.C), copy(env.E), copy(env.A), M
    for (b, j) in zip(bcs, js), e in 1:2
        kind, i, leg = b.ends[e]
        ι = b.ιs[e][j]
        if kind === :C
            C[i] = CE._os_slice_leg(C[i], leg, ι)
        elseif kind === :E
            E[i] = CE._os_slice_leg(E[i], leg, ι)
        elseif kind === :A
            A[i] = CE._os_slice_leg(A[i], leg, ι)
        else
            Mj = CE._os_slice_leg(Mj, leg, ι)
        end
    end
    return C, E, A, Mj
end

# Sum over the slices in nw tasks, every task with its own ReuseAllocator; the garbage collection follows the gc_gib rule for all tasks together
function sliced_parallel(env, M, cuts, kern, nw; gc_gib = PAR_GC_GIB)
    bcs = [CE.BondCut(env, M, lab, n) for (lab, n) in cuts]
    jss = vec(collect(Iterators.product((1:CE.npieces(b) for b in bcs)...)))
    accs = zeros(scalartype(M), nw)
    allocs = [CE.ReuseAllocator() for _ in 1:nw]
    lk = ReentrantLock()
    last = Ref(Base.gc_bytes())
    ngc = Ref(0)
    @sync for w in 1:nw
        Threads.@spawn for n in w:nw:length(jss)
            C, E, A, Mj = slice_tensors(env, M, bcs, jss[n])
            accs[w] += kern(C, E, A, Mj; allocator = allocs[w])
            C = E = A = Mj = nothing
            lock(lk) do
                if Base.gc_bytes() - last[] > gc_gib * 2^30
                    ngc[] += 1
                    GC.gc(ngc[] % 10 == 0)
                    last[] = Base.gc_bytes()
                end
            end
        end
    end
    pool = sum(a.peak_os_bytes for a in allocs)
    foreach(CE.release!, allocs)
    return sum(accs), pool, length(jss)
end

# Share of the profile samples per kind of work (leaf side first: garbage collection, idle thread, BLAS, permutes/copies, slicing, other)
function classify(sample, lidict)
    names = String[]
    for ip in sample, fr in get(lidict, ip, StackTraces.StackFrame[])
        push!(names, lowercase(string(fr.func, " ", fr.file)))
    end
    has(keys...) = any(s -> any(k -> occursin(k, s), keys), names)
    has("jl_gc_", "gc_collect", "safepoint_wait_gc") && return "gc"
    has("poptask", "task_get_next", "jl_task_wait", "uv_run") && return "idle"
    has("gemm", "openblas", "blas.jl") && return "blas"
    has("strided", "tensoradd", "permute", "add_transpose", "_mapreduce", "copyto") && return "permute"
    has("_os_slice_leg", "bondcut") && return "slicing"
    return "other"
end

function profile_breakdown(f)
    Profile.clear()
    Profile.init(n = 10^7, delay = 0.002)
    @profile f()
    data = Profile.fetch(include_meta = false)
    lidict = Profile.getdict(data)
    counts = Dict{String,Int}()
    sample = UInt64[]
    for ip in data
        if ip == 0
            isempty(sample) || (c = classify(sample, lidict); counts[c] = get(counts, c, 0) + 1)
            empty!(sample)
        else
            push!(sample, ip)
        end
    end
    tot = sum(values(counts); init = 0)
    return join(("$k=$(round(100 * get(counts, k, 0) / max(tot, 1); digits = 1))%" for k in ("blas", "permute", "slicing", "gc", "other", "idle")), " ")
end

# CHILD: one run, prints one result line
function run_child(o)
    part = o["part"]
    cuts = o["cuts"] == "none" ? Symbol[] : Symbol.(split(o["cuts"], ','))
    blas = parse(Int, o["blas"])
    BLAS.set_num_threads(blas)
    env, M = CE.onesite_mock_env(ℂ^DIMS[1], ℂ^DIMS[2], ℂ^DIMS[3]; T = T_MOCK, seed = MOCK_SEED)
    d = CE.onesite_dims(env, M)
    kern = CE.onesite_make_kernel(CE.onesite_reorder(CE.TREE_CURRENT, d; cuts))
    cutp = CE.onesite_full_cuts(env, M, cuts)

    # We warm up on a tiny mock environment: this compiles everything without using the memory of the big one
    e2, M2 = CE.onesite_mock_env(ℂ^2, ℂ^2, ℂ^2; T = T_MOCK)
    cut2 = CE.onesite_full_cuts(e2, M2, cuts)
    CE.contract_onesite_sliced(e2, M2, cut2; kernel = kern, allocator = CE.ReuseAllocator(), gc_every = 0)
    part == "parallel" && sliced_parallel(e2, M2, cut2, kern, parse(Int, o["tasks"]))
    rss0 = GiB(Sys.maxrss())

    GC.gc()
    prof = "-"
    c0 = cpu_seconds()
    if part == "threads"
        a = CE.ReuseAllocator()
        st = @timed CE.contract_onesite_sliced(env, M, cutp; kernel = kern, allocator = a, max_slices = PART1_SLICES, gc_every = 0)
        nrun, pool = min(PART1_SLICES, CE.onesite_nslices(env, M, cutp)), a.peak_os_bytes
    elseif part == "gc"
        mode, val = split(o["mode"], '=')
        a = CE.ReuseAllocator()
        kw = mode == "gib" ? (gc_gib = parse(Float64, val),) : (gc_every = parse(Int, val),)
        st = @timed CE.contract_onesite_sliced(env, M, cutp; kernel = kern, allocator = a, kw...)
        nrun, pool = CE.onesite_nslices(env, M, cutp), a.peak_os_bytes
    else
        st0 = @timed sliced_parallel(env, M, cutp, kern, parse(Int, o["tasks"]))
        st = (time = st0.time, gctime = st0.gctime, bytes = st0.bytes, value = st0.value[1])
        nrun, pool = st0.value[3], st0.value[2]
    end
    busy = (cpu_seconds() - c0) / st.time
    rss = GiB(Sys.maxrss())
    if part == "threads" && get(o, "profile", "false") == "true"
        prof = try
            profile_breakdown(() -> CE.contract_onesite_sliced(env, M, cutp; kernel = kern, allocator = CE.ReuseAllocator(), max_slices = 16, gc_every = 0))
        catch e
            "profile failed: " * replace(sprint(showerror, e), SEP => ' ', '\n' => ' ')
        end
    end
    println(join((nrun, st.time, st.time / nrun, busy, st.gctime, GiB(st.bytes), rss0, rss, GiB(pool), real(st.value), prof), SEP))
end

# PARENT: start one child and turn its last line into the measured columns (or a note when it failed)
function run_child_process(args, jthreads)
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$jthreads $(abspath(PROGRAM_FILE)) --child $args`
    out = try
        strip(read(cmd, String))
    catch e
        "failed: " * sprint(showerror, e)
    end
    l = String(last(split(out, '\n')))
    f = split(l, SEP)
    length(f) == 11 && return (f, "")
    return (fill("NaN", 11), replace(l, SEP => ' '))
end

function read_results(path)
    lines = filter(!isempty, readlines(path))
    names = Symbol.(split(lines[1], SEP))
    rows = NamedTuple[]
    for l in lines[2:end]
        v = Dict(zip(names, split(l, SEP)))
        num(s) = parse(Float64, v[s])
        push!(rows, (part = String(v[:part]), config = String(v[:config]), plan = String(v[:plan]), k = parse(Int, v[:k]), slices = num(:slices_run),
                     time = num(:time_s), per_slice = num(:per_slice_s), busy = num(:busy_cores), gc = num(:gc_s), rss = num(:rss_gib),
                     pool = num(:pool_gib), value = num(:value), profile = String(v[:profile]), note = String(v[:note])))
    end
    return rows
end

function summary(dir)
    allrows = read_results(joinpath(dir, "results.tsv"))
    rows = filter(r -> isfinite(r.time), allrows)
    io = IOBuffer()
    println(io, "Threads, garbage collection and parallel slices: TREE_CURRENT at ($(join(DIMS, ","))), mock $T_MOCK seed $MOCK_SEED, one run each\n")

    println(io, "PART 1: first $PART1_SLICES slices of the k_min plan, no garbage collection in the loop (varied: Julia and BLAS threads)")
    @printf(io, "%-22s %12s %10s %12s\n", "threads (Julia, BLAS)", "s per slice", "speed-up", "busy cores")
    r1 = filter(r -> r.part == "threads", rows)
    ref = findfirst(r -> r.config == "jt=1 blas=1", r1)
    for r in r1
        @printf(io, "%-22s %12.3f %10.2f %12.2f\n", r.config, r.per_slice, ref === nothing ? NaN : r1[ref].per_slice / r.per_slice, r.busy)
    end
    for r in r1
        r.profile == "-" || println(io, "profile ($(r.config), share of the samples of the Julia threads): ", r.profile)
    end

    println(io, "\nPART 2: full contraction of the k_min plan, 4 Julia + 4 BLAS threads (varied: garbage-collection rule)")
    @printf(io, "%-12s %10s %8s %10s %9s %9s\n", "rule", "time s", "gc s", "time/ref", "rss GiB", "busy")
    r2 = filter(r -> r.part == "gc", rows)
    ref = findfirst(r -> r.config == "every=1", r2)
    for r in r2
        @printf(io, "%-12s %10.1f %8.1f %10.2f %9.2f %9.2f\n", r.config, r.time, r.gc, ref === nothing ? NaN : r.time / r2[ref].time, r.rss, r.busy)
    end

    println(io, "\nPART 3: slices in parallel tasks (gc_gib = $PAR_GC_GIB; varied: plan, tasks, BLAS threads; 4 Julia threads)")
    @printf(io, "%-6s %3s %7s %-18s %10s %10s %9s %9s %9s %10s\n", "plan", "k", "slices", "config", "time s", "speed-up", "busy", "rss GiB", "pool GiB", "rel. diff")
    r3 = filter(r -> r.part == "parallel", rows)
    vref = isempty(r2) ? NaN : r2[1].value
    for r in r3
        i = findfirst(x -> x.plan == r.plan && startswith(x.config, "tasks=1 "), r3)
        @printf(io, "%-6s %3d %7.0f %-18s %10.1f %10.2f %9.2f %9.2f %9.2f %10.1e\n", r.plan, r.k, r.slices, r.config, r.time,
                i === nothing ? NaN : r3[i].time / r.time, r.busy, r.rss, r.pool, abs(r.value - vref) / abs(vref))
    end
    println(io, "(speed-up against tasks=1 with 4 BLAS threads of the same plan; rel. diff against the value of part 2)")

    failed = filter(r -> !isempty(r.note), allrows)
    isempty(failed) || println(io, "\nnot measured: ", join(("$(r.part) $(r.config) $(r.plan): $(r.note)" for r in failed), "; "))
    s = String(take!(io))
    write(joinpath(dir, "summary.txt"), s)
    print(s)
end

function main()
    o = parse_cli(ARGS)
    haskey(o, "child") && return run_child(o)
    isempty(SUMMARY_ONLY) || return summary(joinpath(DATA_DIR, SUMMARY_ONLY))

    dir = joinpath(DATA_DIR, Libc.strftime("%Y%m%d_%H%M", time()))
    mkpath(dir)
    path = joinpath(dir, "results.tsv")
    open(io -> println(io, HEADER), path, "w")
    ps = plans()
    open(joinpath(dir, "settings.txt"), "w") do io
        println(io, "Julia ", VERSION, ", threads ", Threads.nthreads(), ", host ", gethostname())
        println(io, "DIMS = ", DIMS, ", RAM_GIB = ", RAM_GIB, ", k_min cuts = ", ps[:kmin], ", k_min + 1 cuts = ", ps[:kmin1], ", PART1_SLICES = ", PART1_SLICES,
                ", GC_MODES = ", GC_MODES, ", PAR_GC_GIB = ", PAR_GC_GIB)
    end
    println("run folder: ", dir)

    runs = Tuple{String,String,Symbol,Vector{String},Int}[]
    for (jt, b) in THREAD_CONFIGS
        push!(runs, ("threads", "jt=$jt blas=$b", :kmin, ["--part=threads", "--blas=$b", "--profile=$((jt, b) == THREAD_CONFIGS[end])"], jt))
    end
    for m in GC_MODES
        push!(runs, ("gc", m, :kmin, ["--part=gc", "--blas=4", "--mode=$m"], 4))
    end
    for (p, nw, b) in PAR_RUNS
        push!(runs, ("parallel", "tasks=$nw blas=$b", p, ["--part=parallel", "--blas=$b", "--tasks=$nw"], 4))
    end

    t0 = time()
    for (part, config, p, args, jt) in runs
        cuts = ps[p]
        line = (meas, note) -> join((part, config, p, length(cuts), meas..., note), SEP)
        if time() - t0 > 3600 * MAX_HOURS
            open(io -> println(io, line(fill("NaN", 11), "skipped: MAX_HOURS reached")), path, "a")
            continue
        end
        println(stderr, "\n[$(round((time() - t0) / 60; digits = 1)) min] $part, $config, plan $p (cuts $(join(cuts, " ")))")
        flush(stderr)
        meas, note = run_child_process(Cmd([args; "--cuts=$(isempty(cuts) ? "none" : join(cuts, ","))"]), jt)
        isempty(note) || println(stderr, "    ", note)
        open(io -> println(io, line(meas, note)), path, "a")
    end
    summary(dir)
end

main()
