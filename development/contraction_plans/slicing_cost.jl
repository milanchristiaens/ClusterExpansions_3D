# Measures what slicing itself costs: for every set of dimensions we time one tree (TREE_CURRENT, OPT_TRUE or an OME tree) with the fewest cuts that fit the RAM budget (k_min),
# with one cut fewer and with one and two cuts more, each with two garbage-collection intervals, every plan in a fresh child process.
# From the peak resident memory of these runs we also fit the memory rule (baseline + factor * model peak) and show which plans a fitted rule would allow.

using Printf, LinearAlgebra
using TensorKit
using ClusterExpansions
const CE = ClusterExpansions

# ======================================================================================================================
# SETTINGS (a Hortense job may pass CE_SLICING_DIMS, CE_SLICING_GC, CE_SLICING_HOURS, CE_SLICING_SUMMARY, CE_SLICING_TREE, CE_SLICING_RAM)
# ======================================================================================================================
parse_dims(s) = [Tuple(parse.(Int, split(g, ','))) for g in split(s, ';') if !isempty(strip(g))]
const DIMS = parse_dims(get(ENV, "CE_SLICING_DIMS", "8,4,3;12,4,3;10,5,3;12,5,3;16,4,3"))  # (χenv, χpeps, D), cheapest first
const GC_EVERYS = parse.(Int, split(get(ENV, "CE_SLICING_GC", "1,10"), ','))                # gc_every of contract_onesite_sliced
const MAX_HOURS = parse(Float64, get(ENV, "CE_SLICING_HOURS", "8"))                         # no new child is started after this many hours
const SUMMARY_ONLY = get(ENV, "CE_SLICING_SUMMARY", "")      # name of an existing run folder: only remake its summary
const RAM_GIB = parse(Float64, get(ENV, "CE_SLICING_RAM", "3"))   # RAM budget that decides k_min (3: the 3 GiB plans of the benchmark, 29: a quarter node)
# The tree: current (TREE_CURRENT), opt_true (OPT_TRUE), ome_flops or ome_scN (the OME tree of the ome-flops-mincuts or ome-scN-mincuts plan of the plan file for these
# dimensions; ome_sc takes the sc tree with the fewest model flops; find_orders_ome.jl makes them with OME_SLICING = :mincuts or :both)
const TREE = Symbol(get(ENV, "CE_SLICING_TREE", "current"))
const T_MOCK = Float64                # element type of the mock environment
const MOCK_SEED = 1                   # same seed as the benchmark: the same environment for all plans of a set
const REPS = 2                        # timed repetitions per child (the fastest counts)
const K_STEPS = (-1, 0, 1, 2)         # number of cuts relative to k_min
# The plan with k_min - 1 cuts does not fit RAM_GIB, but the job itself has more memory (JOB_GIB): it only runs if its model peak is below MAX_PEAK_GIB,
# so that its predicted memory (rule 1.4 + 2.7 × peak) stays below 90 % of JOB_GIB (3 GiB budget on 4 cores: 1.5 GiB)
const JOB_GIB = parse(Float64, get(ENV, "CE_SLICING_JOB_GIB", string(1.86 * Threads.nthreads())))
const MAX_PEAK_GIB = min(0.5 * RAM_GIB, (0.9 * JOB_GIB - 1.4) / 2.7)
const MAX_SLICES = 20_000
const MARGIN = 0.8                    # as in onesite_choose_cuts
# ======================================================================================================================

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "contraction_plans", "slicing_cost"))
const SEP = '\t'
const HEADER = join(("chi_env", "chi_peps", "D", "k", "k_min", "cuts", "gc_every", "slices", "flops", "peak_gib", "time_s", "times_s", "gc_s",
                     "alloc_gib", "rss0_gib", "rss_gib", "tracked_gib", "pool_gib", "sys_s", "minflt", "value", "note"), SEP)
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

# Linux only: minor page faults and system CPU seconds of this process
function procstat()
    isfile("/proc/self/stat") || return nothing
    p = split(split(read("/proc/self/stat", String), ") ")[end])
    return (minflt = parse(Int, p[8]), sys = parse(Int, p[13]) / 100)
end

# The tree of the set g (see TREE); parent and child find the same tree
function tree_of(g)
    TREE === :current && return CE.TREE_CURRENT
    TREE === :opt_true && return CE.OPT_TRUE
    m = match(r"^ome_(flops|sc\d*)$", String(TREE))
    m === nothing && error("CE_SLICING_TREE = $TREE: use current, opt_true, ome_flops, ome_sc or ome_scN")
    kind = "ome-" * m.captures[1]
    # the tree is the same for every budget (the cuts differ), so any -mincuts row of these dimensions will do
    rows = [r for r in CE.onesite_read_plans(CE.ONESITE_PLAN_FILE) if (r.chi_env, r.chi_peps, r.D) == Tuple(g) && endswith(r.source, "-mincuts") &&
            (kind == "ome-sc" ? occursin(r"^ome-sc\d+-mincuts$", r.source) : r.source == kind * "-mincuts") && r.eltype == string(T_MOCK) && r.allocator == "reuse"]
    isempty(rows) && error("no $kind-mincuts plan for $g in $(CE.ONESITE_PLAN_FILE): run find_orders_ome.jl first (OME_SLICING = :mincuts or :both)")
    d = CE.onesite_dims(g...)
    return rows[argmin([CE.onesite_cost(r.tree, d).flops for r in rows])].tree
end

# The plans of one set: the tree with k_min cuts (the plan of onesite_choose_cuts), k_min - 1 cuts (smallest peak) and
# k_min + 1, k_min + 2 cuts (fewest flops with at most half the peak of the plan before, the plan of a about twice smaller budget).
function plans_of(g)
    d = CE.onesite_dims(g...)
    tree = tree_of(g)
    base = CE.onesite_choose_cuts(d; ram_gib = RAM_GIB, T = T_MOCK, allocator = :reuse, tree)
    kmin = base.k
    plans = Dict{Int,Vector{Symbol}}(kmin => base.cuts)
    for s in sort([s for s in K_STEPS if s < 0]; rev = true)
        k = kmin + s
        k >= 0 || continue
        plans[k] = k == 0 ? Symbol[] : CE.onesite_best_cuts(tree, d, k; order = :best).cuts
    end
    peak(cuts) = CE.onesite_cost(CE.onesite_reorder(tree, d; cuts), d; cuts).peak
    for s in sort([s for s in K_STEPS if s > 0])
        k = kmin + s
        (k <= 5 && haskey(plans, k - 1)) || continue
        r = CE.onesite_best_cuts(tree, d, k; cap = peak(plans[k - 1]) / 2, order = :best)
        r === nothing && (r = CE.onesite_best_cuts(tree, d, k; order = :best))
        plans[k] = r.cuts
    end
    out = NamedTuple[]
    for k in sort(collect(keys(plans)))
        cuts = plans[k]
        c = CE.onesite_cost(CE.onesite_reorder(tree, d; cuts), d; cuts)
        push!(out, (k = k, kmin = kmin, cuts = cuts, slices = c.slices, flops = c.flops, peak_gib = GiB(c.peak * sizeof(T_MOCK))))
    end
    return out
end

# CHILD: one plan with one gc_every, prints one result line
function run_child(o)
    dims = Tuple(parse.(Int, split(o["dims"], ',')))
    cuts = o["cuts"] == "none" ? Symbol[] : Symbol.(split(o["cuts"], ','))
    ge = parse(Int, o["gc"])
    env, M = CE.onesite_mock_env(ℂ^dims[1], ℂ^dims[2], ℂ^dims[3]; T = T_MOCK, seed = MOCK_SEED)
    d = CE.onesite_dims(env, M)
    kern = CE.onesite_make_kernel(CE.onesite_reorder(tree_of(dims), d; cuts))
    cutp = CE.onesite_full_cuts(env, M, cuts)

    # We warm up on a tiny mock environment: this compiles the kernel without using the memory of the big one
    e2, M2 = CE.onesite_mock_env(ℂ^2, ℂ^2, ℂ^2; T = T_MOCK)
    CE.contract_onesite_sliced(e2, M2, CE.onesite_full_cuts(e2, M2, cuts); kernel = kern, allocator = CE.TrackingAllocator(CE.ReuseAllocator()), gc_every = ge)
    rss0 = GiB(Sys.maxrss())

    ta = CE.TrackingAllocator(CE.ReuseAllocator())
    ps0 = procstat()
    stats = []
    for r in 1:REPS
        GC.gc()
        push!(stats, @timed CE.contract_onesite_sliced(env, M, cutp; kernel = kern, allocator = ta, gc_every = ge))
        println(stderr, "    rep $r: $(round(stats[end].time; digits = 2)) s (gc $(round(stats[end].gctime; digits = 2)) s)"); flush(stderr)
    end
    ps1 = procstat()
    rss = GiB(Sys.maxrss())
    best = stats[argmin([s.time for s in stats])]
    sys = ps0 === nothing ? NaN : (ps1.sys - ps0.sys) / REPS
    minflt = ps0 === nothing ? NaN : (ps1.minflt - ps0.minflt) / REPS
    println(join((best.time, join([s.time for s in stats], " "), best.gctime, GiB(best.bytes), rss0, rss, GiB(ta.peak_bytes),
                  GiB(ta.inner.peak_os_bytes), sys, minflt, real(best.value)), SEP))
end

# PARENT: start one child and turn its last line into the measured columns (or a note when it failed)
function run_plan(g, p, ge)
    cuts = isempty(p.cuts) ? "none" : join(p.cuts, ",")
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$(Threads.nthreads()) $(abspath(PROGRAM_FILE)) --child --dims=$(join(g, ",")) --cuts=$cuts --gc=$ge`
    out = try
        strip(read(cmd, String))
    catch e
        "failed: " * sprint(showerror, e)
    end
    f = split(String(last(split(out, '\n'))), SEP)
    length(f) == 11 && return (f, "")
    return (fill("NaN", 11), replace(String(last(split(out, '\n'))), SEP => ' ', '\n' => ' '))
end

function read_results(path)
    lines = filter(!isempty, readlines(path))
    names = Symbol.(split(lines[1], SEP))
    rows = NamedTuple[]
    for l in lines[2:end]
        f = split(l, SEP)
        v = Dict(zip(names, f))
        num(s) = parse(Float64, v[s])
        push!(rows, (dims = (parse(Int, v[:chi_env]), parse(Int, v[:chi_peps]), parse(Int, v[:D])), k = parse(Int, v[:k]), kmin = parse(Int, v[:k_min]),
                     cuts = String(v[:cuts]), gc = parse(Int, v[:gc_every]), slices = num(:slices), flops = num(:flops), peak = num(:peak_gib),
                     time = num(:time_s), gc_s = num(:gc_s), rss0 = num(:rss0_gib), rss = num(:rss_gib), note = String(v[:note])))
    end
    return rows
end

# Least squares y ≈ X β
lsq(X, y) = X \ y

function summary(dir)
    rows = filter(r -> isfinite(r.time), read_results(joinpath(dir, "results.tsv")))
    io = IOBuffer()
    rules = CE.ONESITE_RAM_RULES
    println(io, "Slicing cost of the tree $TREE at $RAM_GIB GiB (mock, $T_MOCK, seed $MOCK_SEED, reuse allocator, fastest of $REPS repetitions)")
    println(io, "varied: number of cuts k, gc_every; fixed: tree ($TREE), environment, threads, allocator\n")

    println(io, "TEST 2: time against slices on the same tree")
    for g in unique(r.dims for r in rows)
        rg = sort(filter(r -> r.dims == g, rows); by = r -> (r.gc, r.k))
        @printf(io, "\n(%d,%d,%d)\n%3s %-16s %6s %8s %9s %8s %5s %9s %8s %9s %10s\n", g[1], g[2], g[3], "k", "cuts", "gc", "slices", "flops/ref", "peak GiB", "",
                "time s", "gc s", "time/ref", "Δt/Δslice")
        for r in rg
            ref = findfirst(x -> x.k == x.kmin && x.gc == r.gc, rg)
            rr = ref === nothing ? nothing : rg[ref]
            fr = rr === nothing ? NaN : r.flops / rr.flops
            tr = rr === nothing ? NaN : r.time / rr.time
            ds = rr === nothing || r.slices == rr.slices ? NaN : (r.time - rr.time) / (r.slices - rr.slices)
            @printf(io, "%3d %-16s %6d %8.0f %9.3f %8.3f %5s %9.1f %8.1f %9.2f %10.3f\n", r.k, r.cuts, r.gc, r.slices, fr, r.peak,
                    r.k == r.kmin ? "k_min" : "", r.time, r.gc_s, tr, ds)
        end
        # t ≈ a·flops + c·slices over the runs with the first gc_every
        r1 = filter(r -> r.gc == GC_EVERYS[1], rg)
        if length(r1) >= 3
            β = lsq(hcat([r.flops for r in r1], [r.slices for r in r1]), [r.time for r in r1])
            @printf(io, "fit t = flops/R + slices·t0 (gc_every %d): R = %.1f GFLOP/s, t0 = %.3f s per slice%s\n", GC_EVERYS[1], 1e-9 / β[1], β[2],
                    (β[1] <= 0 || β[2] < 0) ? "   (not meaningful: a negative coefficient)" : "")
        end
    end

    println(io, "\nTEST 3: memory rule (peak resident memory of the child against the model peak)")
    println(io, "current rule: baseline $(rules.baseline_gib) GiB + $(rules.reuse) × model peak (reuse)")
    @printf(io, "\n%-10s %3s %6s %9s %9s %9s %10s\n", "dims", "k", "gc", "peak GiB", "rss0 GiB", "rss GiB", "rule GiB")
    for r in sort(rows; by = r -> (r.gc, r.dims, r.k))
        @printf(io, "%-10s %3d %6d %9.3f %9.2f %9.2f %10.2f\n", join(r.dims, ","), r.k, r.gc, r.peak, r.rss0, r.rss, rules.baseline_gib + rules.reuse * r.peak)
    end
    for ge in GC_EVERYS
        rr = filter(r -> r.gc == ge && r.peak >= 0.05, rows)
        length(rr) >= 2 || continue
        β = lsq(hcat(ones(length(rr)), [r.peak for r in rr]), [r.rss for r in rr])
        b = maximum(r.rss0 for r in rr)
        f = maximum((r.rss - r.rss0) / r.peak for r in rr)
        @printf(io, "\ngc_every %d: least squares rss = %.2f + %.2f × peak; upper envelope: baseline %.2f GiB (largest rss0), factor %.2f (largest (rss - rss0)/peak)\n",
                ge, β[1], β[2], b, f)
        newrules = (baseline_gib = b, reuse = f, manual = rules.manual)
        println(io, "plans for $RAM_GIB GiB (margin $MARGIN) with the current rule and with the upper envelope:")
        for g in unique(r.dims for r in rows)
            d = CE.onesite_dims(g...)
            p0 = CE.onesite_choose_cuts(d; ram_gib = RAM_GIB, T = T_MOCK, allocator = :reuse, tree = tree_of(g))
            p1 = CE.onesite_choose_cuts(d; ram_gib = RAM_GIB, T = T_MOCK, allocator = :reuse, rules = newrules, tree = tree_of(g))
            @printf(io, "  (%d,%d,%d): current k %d, %d slices (%s)   envelope k %d, %d slices (%s)%s\n", g[1], g[2], g[3], p0.k, round(Int, p0.slices), join(p0.cuts, " "),
                    p1.k, round(Int, p1.slices), join(p1.cuts, " "), p1.fits ? "" : "  does not fit")
        end
    end
    failed = filter(r -> !isempty(r.note), read_results(joinpath(dir, "results.tsv")))
    isempty(failed) || println(io, "\nnot measured: ", join(("($(join(r.dims, ","))) k $(r.k) gc $(r.gc): $(r.note)" for r in failed), "; "))
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
    open(joinpath(dir, "settings.txt"), "w") do io
        println(io, "Julia ", VERSION, ", threads ", Threads.nthreads(), ", BLAS threads ", BLAS.get_num_threads(), ", host ", gethostname())
        println(io, "TREE = ", TREE, ", DIMS = ", DIMS, ", GC_EVERYS = ", GC_EVERYS, ", RAM_GIB = ", RAM_GIB, ", REPS = ", REPS, ", K_STEPS = ", K_STEPS,
                ", MAX_PEAK_GIB = ", MAX_PEAK_GIB, ", T_MOCK = ", T_MOCK, ", MOCK_SEED = ", MOCK_SEED)
    end
    println("run folder: ", dir)
    t0 = time()
    for g in DIMS
        ps = plans_of(g)
        for ge in GC_EVERYS, p in ps
            line = (meas, note) -> join((g..., p.k, p.kmin, isempty(p.cuts) ? "none" : join(p.cuts, " "), ge, p.slices, p.flops, p.peak_gib, meas..., note), SEP)
            if p.k < p.kmin && p.peak_gib > MAX_PEAK_GIB
                open(io -> println(io, line(fill("NaN", 11), "skipped: model peak $(round(p.peak_gib; digits = 2)) GiB > MAX_PEAK_GIB")), path, "a")
                continue
            end
            if p.slices > MAX_SLICES || time() - t0 > 3600 * MAX_HOURS
                open(io -> println(io, line(fill("NaN", 11), p.slices > MAX_SLICES ? "skipped: more than MAX_SLICES" : "skipped: MAX_HOURS reached")), path, "a")
                continue
            end
            println(stderr, "\n[$(round((time() - t0) / 60; digits = 1)) min] ($(join(g, ","))) k $(p.k) (k_min $(p.kmin)), cuts $(join(p.cuts, " ")), $(round(Int, p.slices)) slices, gc_every $ge")
            flush(stderr)
            meas, note = run_plan(g, p, ge)
            isempty(note) || println(stderr, "    ", note)
            open(io -> println(io, line(meas, note)), path, "a")
        end
    end
    summary(dir)
end

main()
