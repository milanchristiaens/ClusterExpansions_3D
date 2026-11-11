# ======================================================================================================================
# contract_onesite_slicing.jl -- ONE file for everything around the 27-tensor contraction `contract_onesite` of CTMRG3D.
#
# MAIN ENTRY:   contract_onesite_auto(env, M; ram_gib, ...)
#   computes the same number as contract_onesite(env, M), but it picks (a) a contraction order (tree) and (b) a few bonds to "cut"
#   (= slice) such that the predicted memory fits in `ram_gib`, then runs the slices one after the other.
#   `contract_onesite_dispatch(env, M, ram_gib)` is what `expectation_value` (ctmrg_cubic.jl) calls: ram_gib = nothing -> plain contract_onesite.
#
# IDEA OF SLICING (Gray & Kourtis, Quantum 5, 410 (2021), Sec. 4.7.1): the closed 27-tensor network is a sum over every bond.
#   Pick a few bonds, split each bond space into n pieces, contract the network once per combination of pieces (each contraction
#   sees the cut bonds with the small piece space) and add the scalars.  The peak memory of the big intermediates drops by about the
#   product of the n's; the flops stay close to the original.  The slices run one after the other (serial, single process).
#
# WHAT HAPPENS INSIDE contract_onesite_auto
#   1. d = the 54 bond dimensions of the real tensors                                         (onesite_dims)
#   2. default: the fixed tree TREE_CURRENT + BRUTE FORCE search of the best cuts (fewest flops among the cut sets predicted to fit in the RAM)
#                                                                                          (onesite_choose_cuts, onesite_best_cuts)
#      or a ready-made plan (tree + cuts) from the plan file `onesite_plans.csv`: only plans for exactly these bond dimensions, that are
#      predicted to fit, and that you MEASURED (column time_real_s, or time_s if there is no real time), the fastest one is used (onesite_choose_plan)                                (onesite_choose_plan)
#   3. the tree is turned into a Julia function with the same @tensor syntax as contract_onesite (cached)               (onesite_make_kernel)
#   4. the slices are contracted with an allocator that re-uses its temporaries                                         (contract_onesite_sliced, ReuseAllocator)
#
# WHERE THE PLAN FILE COMES FROM (OMEinsumContractionOrders.jl)
#   development/find_orders_ome.jl (settings at the top of that file: dimensions, RAM budgets, kmax, ...) calls onesite_find_plans, which asks
#   OMEinsumContractionOrders.jl (TreeSA) for several contraction trees, finds the best cuts of every tree by brute force, and appends the winners
#   per RAM budget to onesite_plans.csv (or to another file: PLAN_FILE in that file).
#   Predicted flops are only a proxy for the run time (a tree with fewer flops can run slower), so the rows are timed by running
#   development/benchmark_contract_onesite.jl: it times every row that has no time yet and writes the measured seconds back into the same csv
#   (onesite_update_plans): into time_s when it runs on the dense MOCK environment, into time_real_s when it runs on a REAL environment (ENV_LOADER in
#   that file). The other measurements (allocator, rss_gib, pool_gib, sys_s, minflt, check, note) belong to the real run if there is one, otherwise to the mock run.
#   By default (select = :measured) contract_onesite_auto only uses rows that have a time (time_real_s if filled, otherwise time_s, see onesite_plan_time).
#   A time is only valid for the machine, the allocator and the number of threads it was measured with: use one plan file per machine (PLAN_FILE in both development files).
#   contract_onesite_auto never needs to wait for OMEinsumContractionOrders: it only READS the csv.
#
# Slicing and the contraction order only change speed and memory, not the value of the contraction (up to floating point rounding).
#
# FILE MAP
#    1. MAIN ENTRY            ONESITE_RAM_RULES, onesite_plan_of, onesite_choose_cuts, onesite_choose_plan, contract_onesite_auto, contract_onesite_dispatch
#    2. PLAN FILE + OMEco     onesite_read_plans / onesite_write_plans, onesite_find_trees, onesite_find_plans
#    3. THE NETWORK           table of the 27 tensors, 54 bonds, dimensions, TREE_CURRENT, TREE_FAST
#    4. COST MODEL            onesite_cost (flops, peak memory, slices), onesite_reorder
#    5. TREES AS TEXT         onesite_parse_tree
#    6. BRUTE-FORCE CUTS      onesite_best_cuts
#    7. CODE GENERATION       onesite_kernel_source, contract_onesite_kernel(_fast), onesite_make_kernel
#    8. SLICING MACHINERY     BondCut, _os_slice_leg
#    9. ALLOCATORS            ReuseAllocator, TrackingAllocator
#   10. SLICED CONTRACTION    contract_onesite_sliced, check_slicing
#   11. MOCK ENVIRONMENT      onesite_mock_env (random tensors with the right leg structure, for tests and benchmarks)
#   12. TWO-SITE OBSERVABLES  (only a description, nothing is implemented)
#
# Names of the bonds are the index names of the @tensor expression, e.g. :L5 is `env.C[6][B3 L5 L2]`-`env.E[10][L5 B6; L10 L7]`.
# ======================================================================================================================

# inside the ClusterExpansions module these are already loaded, they are repeated here so that the file can also be tested standalone
using TensorKit, TensorOperations, Random
using TensorOperations: ManualAllocator
import OMEinsumContractionOrders as OME


# ======================================================================================================================
# 1. MAIN ENTRY
# ======================================================================================================================

# Now we define an emperical memory rule used to turn the model peak onesite_cost(...).peak (which only counts intermediate tensors) into a predicted process size
# predicted GiB = baseline_gib + factor * (model peak in GiB)
# baseline_gib: resident size of Julia + TensorKit + compiled kernels
# factor = reuse (ReuseAllocator) or manual (ManualAllocator)
# These values were found from small tests on my laptop, and can differ if using HPC
const ONESITE_RAM_RULES = (baseline_gib = 1.4, reuse = 2.7, manual = 1.6)


# Only accepts :reuse and :manual
function _os_check_alloc(allocator::Symbol)
    allocator === :reuse || allocator === :manual || error("allocator must be :reuse or :manual, got :$allocator")
    return nothing
end


# Given a tree and a cut set, evaluate it: compute its cost with and without the cuts.
# reorder = true: first rewrite the order of the two children in every product of the tree such that the peak memory of THIS cut set is as small
# as possible (see onesite_reorder), the plan then contains the rewritten tree.
# Return some data in a named tuple, `tree` is the tree that has to be contracted.
function onesite_plan_of(d::Vector{Float64}, cuts; ram_gib::Real, bytes::Integer = 16, allocator::Symbol = :reuse,
                         margin::Real = 0.8, rules = ONESITE_RAM_RULES, tree = TREE_CURRENT, reorder::Bool = true,
                         source::AbstractString = "")
    # bytes depends on what type of number: ComplexF64 is 16, Float64 is 8
    # d is vector of 54 bond dimensions, cuts is one combination of bond labels to cut
    # margin = fraction of total RAM we allow to be taken by this contraction
    # source = free text telling where this plan comes from (shown with verbosity > 1)

    # Check if allocator is one of reuse/manual
    _os_check_alloc(allocator)

    # Make a Vector of the cuts
    cs = Symbol[Symbol(c) for c in cuts]
    # Possibly rewrite the order of the children, this does not change the flops, only the peak
    t = reorder ? onesite_reorder(tree, d; cuts = cs) : tree
    # Calculate the cost of the unsliced network
    c0 = onesite_cost(t, d)
    # Calculate the cost of the cutted network
    c = onesite_cost(t, d; cuts = cs)

    # Calculate the theoretical expected max peak in GiB
    peak_gib = c.peak * bytes / 2^30

    # Calculate the estimated peak in GiB
    est = rules.baseline_gib + rules[allocator] * peak_gib

    # Return a Tuple: tree, cuts, amount of cuts, product of cut dimensions, total flops, relative needed flops (w.r.t. same tree without cuts),
    # fits=true if est <= margin * ram_gib
    return (tree = t, cuts = cs, k = length(cs), slices = c.slices, flops = c.flops, flops_rel = c.flops / c0.flops, peak_gib = peak_gib,
            est_gib = est, fits = est <= margin * ram_gib, source = String(source))
end


# For ONE fixed tree: search over different amount of cuts k=0 to kmax (if nothing fits, it tells us)
# For every k the cheapest cut set (fewest flops) is searched by brute force, first with the written child order of the tree (so the precompiled
# kernel can be used for TREE_CURRENT), then, only if that does not fit, with the best child order (the plan then contains a reordered tree).
# Return its plan
function onesite_choose_cuts(d::Vector{Float64}; ram_gib::Real, bytes::Integer = 16, allocator::Symbol = :reuse,
                             kmax::Int = 4, margin::Real = 0.8, rules = ONESITE_RAM_RULES, tree = TREE_CURRENT,
                             source::AbstractString = "")
    # Do some checks
    _os_check_alloc(allocator)
    0 <= kmax <= 5 || error("need 0 <= kmax <= 5, got kmax = $kmax")

    # Calculate the max number of allowed elements = cap
    allowed_gib = (margin * ram_gib - rules.baseline_gib) / rules[allocator]
    cap = allowed_gib * 2^30 / bytes

    # Define keywords for onesite_plan_of function
    kw = (; ram_gib, bytes, allocator, margin, rules, tree, source)
    # Loop over all amount of cuts, stop at lowest k that works
    for k in 0:kmax, order in (:written, :best)
        # If no cut
        cuts = if k == 0
            # If within cap: use no cuts 'Symbol[]', otherwise nothing
            onesite_cost(tree, d; order).peak <= cap ? Symbol[] : nothing
        # if cut
        else
            # Find the best cuts for the same constant tree (lowest FLOPS, where peak<cap)
            r = onesite_best_cuts(tree, d, k; cap, order)
            # if no r found, do nothing, otherwise return r.cuts
            r === nothing ? nothing : r.cuts
        end

        # As soon as a cut if found (for lowest k), return its plan
        cuts === nothing || return onesite_plan_of(d, cuts; kw..., reorder = (order === :best))
    end

    # If nothing fitted
    # if also no cuts allowed, then we just use an empty cut list
    # Otherwise we use kmax cuts, and find the best contraction/slicing with the lowest peak memory
    cuts = kmax == 0 ? Symbol[] : onesite_best_cuts(tree, d, kmax; order = :best).cuts
    # Now we return the plan, fits=False will hold
    return onesite_plan_of(d, cuts; kw..., reorder = true)
end


# Choose the plan (tree + cuts) for these bond dimensions d.
#   cuts given               -> that cut set on `tree` (still checked against the budget)
#   tree given (not default) -> brute-force cuts on that tree
#   otherwise                -> brute-force cuts on TREE_CURRENT, and the ready-made plans of `plans` (csv file, or nothing to ignore the file)
#                               made for exactly these bond dimensions and predicted to fit:
#      select = :measured (default): only plans with a measured time (onesite_plan_time) count, the fastest of them replaces the brute-force plan
#      select = :model             : all plans count, the one with the fewest PREDICTED flops wins (the model ignores GEMM efficiency and data
#                                    volume, a tree with fewer flops can (AND OFTEN WILL!) run slower: check with development/benchmark_contract_onesite.jl)
function onesite_choose_plan(d::Vector{Float64}; ram_gib::Real, bytes::Integer = 16, allocator::Symbol = :reuse, cuts = nothing,
                             tree = TREE_CURRENT, plans = ONESITE_PLAN_FILE, select::Symbol = :measured, kmax::Int = 4,
                             margin::Real = 0.8, rules = ONESITE_RAM_RULES)
    select === :measured || select === :model || error("select must be :measured or :model, got :$select")
    kw = (; ram_gib, bytes, allocator, margin, rules)

    # 1. the caller fixed the cuts
    cuts === nothing || return onesite_plan_of(d, cuts; kw..., tree, source = "given cuts")

    # 2. No cuts given: brute-force search of the cuts on the fixed tree
    default_tree = (tree == TREE_CURRENT)
    best = onesite_choose_cuts(d; kw..., kmax, tree, source = default_tree ? "TREE_CURRENT, brute-force cuts" : "given tree, brute-force cuts")

    # 3. ready-made plans, only if the caller did not ask for a specific tree
    (plans === nothing || !default_tree) && return best
    chosen = nothing
    chosen_key = Inf
    for r in onesite_plans_for(d, plans)
        # in :measured mode a plan without measured time is not used
        (select === :measured && !isfinite(onesite_plan_time(r))) && continue
        p = onesite_plan_of(d, r.cuts; kw..., tree = r.tree, source = "plan file: " * r.source)
        p.fits || continue
        # the number to minimise: measured time or predicted flops
        k = select === :measured ? onesite_plan_time(r) : p.flops
        if k < chosen_key
            chosen = p
            chosen_key = k
        end
    end
    chosen === nothing && return best
    # in :model mode the plan must also need fewer flops than the brute-force plan; in :measured mode a measured plan is preferred
    # (the brute-force plan has no measured time)
    select === :model && best.fits && best.flops <= chosen.flops && return best
    return chosen
end


# The main contraction function:
# This computes the same number as contract_onesite(env, M), but using slices, number of slices chosen automatically from RAM budget
function contract_onesite_auto(env, M; ram_gib::Real = Sys.total_memory() / 2^30, allocator::Symbol = :reuse, cuts = nothing,
                               tree = TREE_CURRENT, plans = ONESITE_PLAN_FILE, select::Symbol = :measured, kmax::Int = 4,
                               margin::Real = 0.8, rules = ONESITE_RAM_RULES, force::Bool = false, verbosity::Int = 0)

    # ram_gib: memory this PROCESS may use. Default=Sys.total_memory(), pass it explicitely if needed (e.g. = 64)
    # allocator: :reuse or :manual (:reuse = `ReuseAllocator`: 13-19 % faster than `:manual` in Windows runs)
    # cuts: force a cut set (vector of labels such as `[:A9, :L5]`) instead of choosing one; it is still checked against the budget.
    # tree: contraction order (nested pairs of leaf numbers, see TREE_CURRENT); default TREE_CURRENT. If you pass another tree, the plan file is ignored.
    # plans: csv file with ready-made plans (default src/utility/onesite_plans.csv), or `nothing` to ignore plan files.
    # select: :measured (only plans with a measured time_s are used, the fastest wins) or :model (fewest predicted flops), see onesite_choose_plan
    # kmax: largest number of cuts to try (<= 5).  `margin`: fraction of `ram_gib` the prediction may fill.
    # force: run even if nothing is predicted to fit (will probably swap or be killed).

    # Check if allowed allocator
    _os_check_alloc(allocator)
    # Vector of dimensions for all bounds
    d = onesite_dims(env, M)
    # Define number of bytes per element (8 for Float64, 16 for ComplexF64)
    bytes = sizeof(scalartype(M))

    # Choose tree + cuts (given cuts are checked, otherwise brute force on the tree, and the plan file as alternative)
    plan = onesite_choose_plan(d; ram_gib, bytes, allocator, cuts, tree, plans, select, kmax, margin, rules)

    # if the memory cost is too high (and we don't force it)
    if !plan.fits && !force
        # Give error
        error("contract_onesite_auto: no cut set with at most $(cuts === nothing ? kmax : length(cuts)) cuts is predicted to fit ",
              "$(round(margin * ram_gib; digits = 2)) GiB (best: cuts $(plan.cuts), predicted $(round(plan.est_gib; digits = 2)) GiB). ",
              "Give more `ram_gib`, a larger `kmax` (<= 5), or find better plans with development/find_orders_ome.jl; `force = true` runs anyway.")
    end

    # if the memory cost is too high and we DO force it
    if !plan.fits && verbosity > 0
        @warn "contract_onesite_auto: predicted $(round(plan.est_gib; digits = 2)) GiB > $(round(margin * ram_gib; digits = 2)) GiB budget, running anyway (force = true)"
    end
    verbosity > 1 && @info "contract_onesite_auto: $(plan.source); $(plan.k) cuts $(plan.cuts), $(round(Int, plan.slices)) slices, " *
                           "predicted $(round(plan.est_gib; digits = 2)) GiB, $(round(plan.flops; sigdigits = 3)) flops"

    # Get the contraction function of the tree of the plan (the precompiled one for TREE_CURRENT, otherwise generated once and cached)
    kernel = onesite_make_kernel(plan.tree)
    # Choose the correct allocator
    alloc = allocator === :reuse ? ReuseAllocator() : ManualAllocator()

    # Perform the sliced contraction
    try
        return contract_onesite_sliced(env, M, onesite_full_cuts(env, M, plan.cuts); kernel, allocator = alloc, verbosity)
    finally
        # Release memory if not ReuseAllocator
        allocator === :reuse && release!(alloc)
    end
end

# Here we let one key word, ram_gib, decide wether to use the old contract_onesite function (see ctmrg3D.jl) or the sliced contract_onesite_auto function
contract_onesite_dispatch(env, M, ram_gib::Nothing; kwargs...) = contract_onesite(env, M)
contract_onesite_dispatch(env, M, ram_gib::Real; kwargs...) = contract_onesite_auto(env, M; ram_gib, kwargs...)


# ======================================================================================================================
# 2. PLAN FILE (csv) + FINDING TREES WITH OMEinsumContractionOrders.jl
# ======================================================================================================================

# The plan file has one ready-made plan per row, columns separated by ';' (can be opened in Excel):
#    chi_env;chi_peps;D;tree;cuts;source;k;slices;flops;peak_gib;ram_gib;bytes;time_s;time_real_s;allocator;rss_gib;pool_gib;sys_s;minflt;check;note
# Columns that are read: bond dimensions the plan was made for (chi_env, chi_peps, D), the contraction tree (nested pairs of leaf numbers, see
# TREE_CURRENT), the cut bonds (labels separated by a space, empty = no cuts), a free text (source), time_s = the measured time in seconds of
# one full contraction on the dense MOCK environment (onesite_mock_env) and time_real_s = the same on a REAL environment (empty = not measured).
# The time that counts for contract_onesite_auto is time_real_s if it is filled, otherwise time_s (onesite_plan_time).
# The columns k ... bytes are written by onesite_find_plans for the human reader:
# number of cuts, number of slices, predicted flops, predicted peak of the intermediates (GiB), the RAM budget (GiB) and bytes per element it was made for.
+ # The columns after time_real_s are written by development/benchmark_contract_onesite.jl (onesite_update_plans), all empty until the row is measured:
# allocator the time was measured with (reuse/manual), rss_gib = peak resident memory of the process, pool_gib = what the ReuseAllocator asked the OS for,
# sys_s = system CPU seconds and minflt = minor page faults per repetition (Linux only), check = relative deviation of the value from the first row
# measured for the same dimensions in that benchmark run (0 for that first row; should be ~1e-10 or smaller), note = free text (e.g. "skipped: ...").
# These seven columns belong to the real run if time_real_s is filled, otherwise to the mock run.
# Older files with 13 columns (up to time_s) or 20 columns (no time_real_s: the measurement columns are 14..20) are still read: a line is read as
# the 21-column layout only if it has 21 columns. Lines starting with # and the header line are ignored.
const ONESITE_PLAN_FILE = joinpath(@__DIR__, "onesite_plans.csv")
const _os_PLAN_HEADER = "chi_env;chi_peps;D;tree;cuts;source;k;slices;flops;peak_gib;ram_gib;bytes;time_s;time_real_s;allocator;rss_gib;pool_gib;sys_s;minflt;check;note"

# Read the plan file, return a Vector of named tuples (chi_env, chi_peps, D, tree, cuts, source, time_s, time_real_s, allocator, rss_gib, pool_gib, sys_s, minflt, check, note).
# A missing file gives an empty vector. The numbers are NaN and the texts "" if they are not filled in.
function onesite_read_plans(path::AbstractString)
    out = NamedTuple[]
    isfile(path) || return out

    for (n, line) in enumerate(eachline(path))
        s = strip(line)
        # skip empty lines, comments and the header
        (isempty(s) || startswith(s, "#") || startswith(s, "chi_env")) && continue

        # split the columns
        f = split(line, ';')
        length(f) >= 6 || error("$path, line $n: expected at least 6 ';'-separated columns, got $(length(f))")

        # cuts: labels separated by a space, check that all are real bond labels
        cuts = Symbol[Symbol(c) for c in split(f[5])]
        for c in cuts
            haskey(ONESITE_LABEL_INDEX, c) || error("$path, line $n: unknown bond label $c")
        end

        # column i as text / as number (NaN if the column is missing or empty)
        txt(i) = length(f) >= i ? String(strip(f[i])) : ""
        num(i) = isempty(txt(i)) ? NaN : parse(Float64, txt(i))
        # 21 columns: time_real_s is column 14 and the measurement columns follow (15..21); 13 or 20 columns (older files): no time_real_s, measurements at 14..20
        o = length(f) >= 21 ? 1 : 0

        push!(out, (chi_env = parse(Int, f[1]), chi_peps = parse(Int, f[2]), D = parse(Int, f[3]),
                    tree = onesite_parse_tree(strip(f[4])), cuts = cuts, source = String(strip(f[6])), time_s = num(13), time_real_s = o == 1 ? num(14) : NaN,
                    allocator = txt(14 + o), rss_gib = num(15 + o), pool_gib = num(16 + o), sys_s = num(17 + o), minflt = num(18 + o), check = num(19 + o), note = txt(20 + o)))
    end
    return out
end

# The time of a plan row that counts when plans are compared: the real-environment time if there is one, otherwise the mock-environment time (NaN if none)
onesite_plan_time(r) = isfinite(r.time_real_s) ? r.time_real_s : r.time_s

# Define a memory of plan files that were already read, so the CSV is only parsed again after it changes on disk.
const _os_PLAN_CACHE = Dict{String,Tuple{Float64,Vector{NamedTuple}}}()

# this looks up the path and compares the cache stored mtime with the file’s current mtime
# If they are equal, it returns the stored rows without reading the file.
# Otherwise onesite_read_plans is called, which parses every line of csv and stores the result
function _os_cached_plans(path::AbstractString)
    isfile(path) || return NamedTuple[]
    mt = mtime(path)
    c = get(_os_PLAN_CACHE, String(path), nothing)
    (c !== nothing && c[1] == mt) && return c[2]
    rows = onesite_read_plans(path)
    _os_PLAN_CACHE[String(path)] = (mt, rows)
    return rows
end

# The plans of the file made for exactly these bond dimensions d (we look at everything stored in the cache)
onesite_plans_for(d::Vector{Float64}, path::AbstractString) =
    [r for r in _os_cached_plans(path) if onesite_dims(r.chi_env, r.chi_peps, r.D) == d]

# Add plans to the file (rows as returned by onesite_find_plans). A row whose dimensions, tree and cuts are already in the file is NOT added
# again (so a time_s that you filled in is never overwritten). Returns the number of rows that were added.
function onesite_write_plans(path::AbstractString, rows)
    key(chi_env, chi_peps, D, tree, cuts) = (chi_env, chi_peps, D, string(tree), Symbol[Symbol(c) for c in cuts])

    # The lines already in the file (the header is written again) and the keys of the plans in them
    old = String[]
    oldkeys = Set{Any}()
    if isfile(path)
        for line in eachline(path)
            s = strip(line)
            (isempty(s) || startswith(s, "chi_env")) && continue
            push!(old, line)
            if !startswith(s, "#")
                f = split(line, ';')
                push!(oldkeys, key(parse(Int, f[1]), parse(Int, f[2]), parse(Int, f[3]), onesite_parse_tree(strip(f[4])), split(f[5])))
            end
        end
    end

    # The new lines
    newlines = String[]
    for r in rows
        k = key(r.chi_env, r.chi_peps, r.D, r.tree, r.cuts)
        k in oldkeys && continue
        push!(oldkeys, k)
        time_s = hasproperty(r, :time_s) && isfinite(r.time_s) ? string(r.time_s) : ""
        push!(newlines, join((r.chi_env, r.chi_peps, r.D, string(r.tree), join(r.cuts, ' '), replace(String(r.source), ';' => ','),
                              r.k, round(Int, r.slices), r.flops, round(r.peak_gib; digits = 3), r.ram_gib, r.bytes, time_s, "", "", "", "", "", "", "", ""), ';')) # time_real_s, empty columns: allocator ... note
    end

    open(path, "w") do io
        println(io, _os_PLAN_HEADER)
        foreach(line -> println(io, line), old)
        foreach(line -> println(io, line), newlines)
    end

    # The cache gets deleted
    delete!(_os_PLAN_CACHE, String(path))
    return length(newlines)
end


# The fields of a line of the plan file in the 21-column layout (older 13- or 20-column lines get an empty time_real_s column and empty missing columns)
function _os_plan_fields(f::Vector{String})
    length(f) >= 21 && return f[1:21]
    g = f[1:min(end, 13)]
    while length(g) < 13
        push!(g, "")
    end
    push!(g, "")                                            # time_real_s
    length(f) > 13 && append!(g, f[14:min(end, 20)])        # allocator ... note
    while length(g) < 21
        push!(g, "")
    end
    return g
end

# Fill in the measurements of development/benchmark_contract_onesite.jl in the plan file.
# `updates` = named tuples with the key of a row (chi_env, chi_peps, D, tree, cuts) and the columns to set: time_s, allocator, rss_gib, pool_gib, sys_s, minflt, check, note.
# `env = :mock`: `time_s` of the update is written into column time_s (dense mock environment); `env = :real`: it is written into column time_real_s (real environment).
# The seven columns (allocator ... note) belong to the real run if there is one: an `env = :real` update always writes them, a :mock update only if time_real_s of that row is still empty.
# A property that is missing, or a number that is NaN, gives an empty column.
# An update WITHOUT a finite time_s (a run that was skipped or failed) only writes `note` (on the same condition) and keeps everything else of the row.
# All other lines (comments, other rows) are kept as they are (older lines keep their layout), a row that is updated is written in the 21-column layout,
# the header is written again, and the file is replaced in one step (temporary file + move),
# Returns the number of rows that were updated; an update whose row is not in the file is an error.
function onesite_update_plans(path::AbstractString, updates; env::Symbol = :mock)
    # Some prior checks
    env === :mock || env === :real || error("onesite_update_plans: env must be :mock or :real, got :$env")
    isfile(path) || error("onesite_update_plans: $path does not exist")
    
    # Small local function: turns row's identity into tuple that can be compared
    key(chi_env, chi_peps, D, tree, cuts) = (chi_env, chi_peps, D, string(tree), Symbol[Symbol(c) for c in cuts])
    
    # Build Dict from key (which we use to compare with rows in csv file) to the full update yuple
    pending = Dict{Any,Any}(key(u.chi_env, u.chi_peps, u.D, u.tree, u.cuts) => u for u in updates)
    nup = length(pending)

    # a number as text (empty if NaN or missing), a free text without ';'
    num(x) = (x isa Real && isfinite(x)) ? string(x) : ""
    txt(x) = replace(String(x), ';' => ',', '\n' => ' ')

    lines = String[]            # new content of file
    done = 0                    # count rows that were updated (should go to nup)
    # Loop over all lines of csv
    for line in eachline(path)
        s = strip(line)

        # Old header is skipped , fresh header is written at end
        startswith(s, "chi_env") && continue
        # Skip comments and empty lines                    
        if isempty(s) || startswith(s, "#")
            push!(lines, line)
            continue
        end
        # Split line at ; into string fields
        f = String.(split(line, ';'))
        # Build the key of this line in the file
        k = key(parse(Int, f[1]), parse(Int, f[2]), parse(Int, f[3]), onesite_parse_tree(strip(f[4])), split(f[5]))
        
        # Check if there is an update for this row? (is key also in pending?)
        if haskey(pending, k)
            # In that case u is the update info
            u = pending[k]
            # Make sure row has exactly 21 fields (add empty fields if necessary)
            f = _os_plan_fields(f)
            # t is time of this run, if no t then NaN
            t = get(u, :time_s, NaN)
            # Should the columns (allocator ... note) belong to this run? Yes, if real run, no if mock run and already real run done before (f[14] is time_real_s) 
            mine = env === :real || isempty(strip(f[14]))
            
            # If there is a time, write it in correct column (real/mock)
            if isfinite(t)
                f[env === :real ? 14 : 13] = num(t)
                # If update (allocator ... note) should belong to this run, rewrit the old columns
                mine && (f[15:21] = [txt(get(u, :allocator, "")), num(get(u, :rss_gib, NaN)), num(get(u, :pool_gib, NaN)), num(get(u, :sys_s, NaN)),
                                     num(get(u, :minflt, NaN)), num(get(u, :check, NaN)), txt(get(u, :note, ""))])
            # If no finite time (run was skipped/failed), only write note if mine (e.g. Note = "skipped: estimated 3.3 GiB > 85% of free 3.4 GiB (set FORCE = true)" or "failed: ...")
            else
                mine && (f[21] = txt(get(u, :note, "")))
            end

            # Glue all fields back into a line with ;
            line = join(f, ';')
            # Delete this update from pending
            delete!(pending, k)
            done += 1
        end
        # Every line goes into lines (also non updated rows)
        push!(lines, line)
    end
    # if an update was never matched, error. Means that file was not initialized written before
    isempty(pending) || error("onesite_update_plans: $(length(pending)) of $nup rows are not in $path (first: $(first(keys(pending))))")

    # The complete new file is written to plans.csv.tmp
    tmp = string(path, ".tmp")
    open(tmp, "w") do io
        # Also add a header now, then all lines
        println(io, _os_PLAN_HEADER)
        foreach(l -> println(io, l), lines)
    end
    # Now replace the old file with the original file
    mv(tmp, path; force = true)
    # Forget the cached copy of this file, next read sees the new content this way
    delete!(_os_PLAN_CACHE, String(path))
    return done
end


# Turn the output of OMEinsumContractionOrders (a binary tree `NestedEinsum` over the input tensors) into our nested tuple of leaf numbers
function _os_tree_from_ome(ne)
    OME.isleaf(ne) && return ne.tensorindex
    length(ne.args) == 2 || error("OMEinsumContractionOrders returned a contraction of $(length(ne.args)) tensors, expected pairs")
    return (_os_tree_from_ome(ne.args[1]), _os_tree_from_ome(ne.args[2]))
end

# Ask OMEinsumContractionOrders.jl (simulated annealing, TreeSA) for contraction trees of our network with bond dimensions d.
#   First a tree with only flops in the score; its largest tensor has 2^sc_max elements.
#   Then n_targets trees that are also punished for tensors larger than 2^(sc_max - 2j) elements (j = 1..n_targets): less memory, more flops.
# Returns a Vector of (tree, source, tc); source is "ome-flops" (flops only) or "ome-scN" (target 2^N elements). Trees with outer products are thrown away. The search is random: another run can give other trees.
# ntrials / niters are the TreeSA settings (more = better trees, slower). Takes a few seconds per tree at (8,5,5).
function onesite_find_trees(d::Vector{Float64}; n_targets::Int = 1, sc_step::Int = 2, ntrials::Int = 4, niters::Int = 40, verbosity::Int = 0)    # The network as an einsum: one input tensor per row of ONESITE_TENSORS, no output indices (closed network = scalar)
    code = OME.EinCode([copy(labs) for (_, _, labs) in ONESITE_TENSORS], Symbol[])
    sizes = Dict{Symbol,Int}(l => round(Int, d[i]) for (i, l) in enumerate(ONESITE_LABELS))

    # Run TreeSA with a given score: tc = log2(flops), sc = log2(largest tensor), sc_target in log2 as well
    function search(sc_weight, sc_target)
        # score = tc_weight · 2^tc  +  rw_weight · 2^rw  +  sc_weight · max(0, 2^sc − 2^sc_target)
        score = OME.ScoreFunction(tc_weight = 1.0, sc_weight = sc_weight, rw_weight = 0.0, sc_target = sc_target)
        return OME.optimize_code(code, sizes, OME.TreeSA(; ntrials, niters, score))
    end

    found = NamedTuple[]
    function keep!(ne, source)
        t = _os_tree_from_ome(ne)
        (_os_has_outer(t) || any(f -> f.tree == t, found)) && return
        # tc = log2(flops) as computed by OMEinsumContractionOrders (can be compared with log2(onesite_cost(tree, d).flops))
        push!(found, (tree = t, source = source, tc = OME.contraction_complexity(ne, sizes).tc))
    end

    # 1. flops only
    ne0 = search(0.0, 0.0)
    keep!(ne0, "ome-flops")
    sc_max = OME.contraction_complexity(ne0, sizes).sc
    verbosity > 0 && @info "onesite_find_trees: flops-only tree has 2^$(round(sc_max; digits = 1)) elements in its largest tensor"

    # 2. with a memory target
        for j in 1:n_targets
        target = round(Int, sc_max) - sc_step * j     # an integer, so the name ome-sc<target> is exactly the target we search with
        # Stop early if target falls below 2^10 elements
        target < 10 && break
        keep!(search(1.0, target), "ome-sc$(round(Int, target))")
    end
    return found
end


# Make plans (tree + cuts) for bond dimensions (χenv, χpeps, D) and a list of RAM budgets, and return them as rows for onesite_write_plans.
# Candidates: TREE_CURRENT, TREE_FAST and the trees of onesite_find_trees.  For each budget, every candidate gets its best cuts (brute force,
# at most kmax of them); candidates that do not fit are dropped, the same plan for several budgets is kept once (for the smallest budget).
# Parameters of the cost model that you can vary: bytes (16 = ComplexF64, 8 = Float64), allocator, margin, rules; (all of them are used again by
# contract_onesite_auto when it checks a plan, so use the same values there).
# The rows have no measured time yet (time_s = NaN): development/benchmark_contract_onesite.jl times them and writes the time and the other measurements into the plan file.
function onesite_find_plans(χenv::Int, χpeps::Int, D::Int; ram_gibs = (16, 32, 64, 128, 256, 512), bytes::Int = 16, allocator::Symbol = :reuse,
                            kmax::Int = 4, margin::Real = 0.8, rules = ONESITE_RAM_RULES, n_targets::Int = 1, sc_step::Int = 2, ntrials::Int = 4,
                            niters::Int = 40, verbosity::Int = 1)
    d = onesite_dims(χenv, χpeps, D)
    candidates = NamedTuple[(tree = TREE_CURRENT, source = "TREE_CURRENT"), (tree = TREE_FAST, source = "TREE_FAST")]
    for t in onesite_find_trees(d; n_targets, sc_step, ntrials, niters, verbosity)        # Is there a candidate c whose tree matches with the found tree t? If no put t into candidate list
        any(c -> c.tree == t.tree, candidates) || push!(candidates, (tree = t.tree, source = t.source))
    end
    verbosity > 0 && @info "onesite_find_plans: $(length(candidates)) candidate trees for (χenv, χpeps, D) = ($χenv, $χpeps, $D)"

    rows = NamedTuple[]
    for ram in sort(collect(ram_gibs)), c in candidates
        p = onesite_choose_cuts(d; ram_gib = ram, bytes, allocator, kmax, margin, rules, tree = c.tree, source = c.source)
        if !p.fits
            # c.source = e.g. TREE_CURRENT or ome-flops or ome-sc28 (intermediate tensors are only free if below 2^28 elements (otherwise penalised))
            verbosity > 0 && @warn "onesite_find_plans: $(c.source) does not fit $ram GiB with at most $kmax cuts"
            continue
        end
        # the tree of the plan can be a reordered version of the candidate; the same tree + cuts for a larger budget is not added again
        any(r -> r.tree == p.tree && r.cuts == p.cuts, rows) && continue
        verbosity > 0 && @info "onesite_find_plans: $ram GiB, $(c.source): $(p.k) cuts $(p.cuts), $(round(Int, p.slices)) slices, predicted $(round(p.est_gib; digits = 1)) GiB, $(round(p.flops; sigdigits = 3)) flops"
        push!(rows, (chi_env = χenv, chi_peps = χpeps, D = D, tree = p.tree, cuts = p.cuts, source = c.source, k = p.k, slices = p.slices,
                     flops = p.flops, peak_gib = p.peak_gib, ram_gib = ram, bytes = bytes, time_s = NaN))
    end
    return rows
end


# ======================================================================================================================
# 3. THE NETWORK: table of the 27 tensors, bonds, bond classes, dimensions and the two reference contraction trees
# ======================================================================================================================

# Each row is: kind tensor (corner, face, ...), index of that kind, leg labels in the correct leg order (see leg-order table at the
# top of ctmrg3D.jl; the kernels below are generated from this table, so it is the single source of truth together with TREE_CURRENT)
# e.g. (:E, 1, [:U1, :U4, :U2, :A2]) = env.E[1][U1 U4; U2 A2]
const ONESITE_TENSORS = Tuple{Symbol,Int,Vector{Symbol}}[
    (:C, 1, [:A1, :U3, :U1]),                       
    (:E, 1, [:U1, :U4, :U2, :A2]),                  
    (:C, 2, [:U5, :A3, :U2]),
    (:E, 5, [:B1, :M3, :A1, :M1]),                  
    (:C, 5, [:L3, :B1, :L1]),                       
    (:A, 2, [:NORTH, :B2, :M1, :A2, :M2]),          
    (:E, 6, [:A3, :M5, :B3, :M2]),
    (:E, 9, [:L1, :B2, :L2, :L4]),
    (:C, 6, [:B3, :L5, :L2]),
    (:E, 2, [:U10, :A6, :U5, :U7]),
    (:C, 3, [:A9, :U10, :U12]),                     
    (:A, 1, [:UP, :U4, :U6, :U9, :U7]),           
    (:E, 4, [:U3, :A4, :U8, :U6]),
    (:E, 3, [:U11, :A8, :U12, :U9]),
    (:C, 4, [:U8, :A7, :U11]),
    (:A, 5, [:WEST, :A4, :M3, :B4, :M8]),
    (:E, 8, [:A7, :M8, :B7, :M11]),
    (:E, 12, [:L8, :B4, :L3, :L6]),
    (:C, 8, [:B7, :L8, :L11]),
    (:M, 1, [:UP, :NORTH, :EAST, :DOWN, :SOUTH, :WEST]),
    (:A, 3, [:EAST, :A6, :M10, :B6, :M5]),
    (:A, 4, [:SOUTH, :A8, :M11, :B8, :M12]),
    (:E, 7, [:B9, :M10, :A9, :M12]),               
    (:A, 6, [:DOWN, :L9, :L6, :L4, :L7]),         
    (:E, 11, [:L11, :L9, :L12, :B8]),               
    (:E, 10, [:L5, :B6, :L10, :L7]),
    (:C, 7, [:L10, :B9, :L12]),                   
]


# Number of leafs of contraction tree = number of one site tensors = 27 in our case
const ONESITE_NLEAF = length(ONESITE_TENSORS)

# Create a dict: bound label (lab, e.g. :L10) -> list of ends (kind, index, legposition, e.g. [E, 10, 3] and [C, 7, 1])
function _onesite_build_bonds()
    ends = Dict{Symbol,Vector{Tuple{Symbol,Int,Int}}}()
    # Loop over all tensors, and for each tensor loop over all legs/labels
    for (kind, i, labs) in ONESITE_TENSORS, (k, lab) in enumerate(labs)
        # (kind, i, k) = (kind, index, legposition)
        push!(get!(ends, lab, Tuple{Symbol,Int,Int}[]), (kind, i, k))
    end

    # Check that each label/bound occurs at two tensors exactly
    all(length(v) == 2 for v in values(ends)) || error("every index label must occur exactly twice")

    # Transform the list in each dictionarry value to a tuple
    # ends[:L5] == ((:C,6,2), (:E,10,1))
    return Dict{Symbol,NTuple{2,Tuple{Symbol,Int,Int}}}(lab => (v[1], v[2]) for (lab, v) in ends)
end

# Now we define some constants
"label => ((kind, index, leg position), (kind, index, leg position)); the leg position counts codomain legs first, then domain legs"
const ONESITE_BONDS = _onesite_build_bonds()
const ONESITE_LABELS = sort!(collect(keys(ONESITE_BONDS)))                  # the 54 bonds sorted (:L1, :L10, :L11, :L12, :L2, ...)
# Define the inverse map: bond label -> number 1-54
const ONESITE_LABEL_INDEX = Dict(l => i for (i, l) in enumerate(ONESITE_LABELS))
# Mask of 54 ones's, one bit per bond
const _os_ALLBITS = (UInt64(1) << length(ONESITE_LABELS)) - 1
# Mask for all leafs: Vector of 27 numbers, one per tensor, each number is a 54-bit mask over the bonds.
const _os_LEAFMASK = UInt64[reduce(|, (UInt64(1) << (ONESITE_LABEL_INDEX[l] - 1) for l in labs); init = UInt64(0)) for (_, _, labs) in ONESITE_TENSORS]

"Given a bond label: return its kind (:env, :peps, :D)"
function onesite_bond_class(lab::Symbol)
    # Given bond label, find its ends, then take the first element (:C, :E, ...) of both ends
    kinds = Set((ONESITE_BONDS[lab][1][1], ONESITE_BONDS[lab][2][1]))
    kinds == Set((:C, :E)) && return :env
    kinds == Set((:A, :E)) && return :peps
    kinds == Set((:A, :M)) && return :D
    error("unexpected bond $lab: $kinds")
end


#    onesite_dims(χenv, χpeps, D)  or  onesite_dims(env, M)

# Vector of the dimensions of the 54 bonds (ordered like `ONESITE_LABELS`).  The second form reads the TOTAL dimension of every bond
# from the actual tensors (dense upper bound for symmetric tensors).

# Given values of (χenv, χpeps, D), return for all bond labels the bond dimension
onesite_dims(χenv::Real, χpeps::Real, D::Real) =
    Float64[(c = onesite_bond_class(l); c === :env ? χenv : c === :peps ? χpeps : D) for l in ONESITE_LABELS]

# Given a CTMRG environment and M tensor, return for all bond labels the bond dimension
function onesite_dims(env, M)
    # Allocate vector d
    d = zeros(Float64, length(ONESITE_LABELS))

    # For all bond labels
    for (j, lab) in enumerate(ONESITE_LABELS)
        # Find the first end of the bond, and its kind, i, pos
        kind, i, pos = ONESITE_BONDS[lab][1]
        # Now we go from (kind, i) to the real tensor T
        T = kind === :C ? env.C[i] : kind === :E ? env.E[i] : kind === :A ? env.A[i] : M
        # Now we search the tensors dimension in that leg direction (pos)
        d[j] = dim(space(T, pos))
    end
    return d
end


# Now we define the two reference trees for the full code, we write them as nested tuples ((a, b) = "contract a with b", `a` is evaluated first by @tensor)
"Current contraction used in `contract_onesite`: memory-lean (38.8 GiB for unsliced 8,5,5; the children of two products are ordered so the big block is evaluated first)"
const TREE_CURRENT = (1, (2, (3, ((((16, (17, (18, 19))), (20, (21, (22, (23, (24, (25, (26, 27)))))))), ((10, 11), (12, (13, (14, 15))))), ((4, 5), (6, (7, (8, 9))))))))
"Contraction found by opt=true: less FLOPS needed (doesn't automatically mean faster though!!), but super memory intensive (238 GiB for unsliced 8,5,5)"
const TREE_FAST = (((((20, (((24, ((((25, 19), 27), ((17, 15), (((23, 11), 22), 14))), (((4, (13, 1)), 16), (18, 5)))), 8), (9, 26))), 6), 21), ((12, 10), (2, 3))), 7)


# ======================================================================================================================
# 4. COST MODEL: flops / peak memory of any pairwise contraction order (tree) with sliced bonds (cuts)
# ======================================================================================================================

# Now we define a cost model

# product of the dimensions of the bonds set to one in `mask`
@inline function _os_lp(mask::UInt64, d::Vector{Float64})
    p = 1.0
    # Loop over all bits in mask
    while mask != 0
        # trailing_zeros(mask) + 1: brings us from mask bit to an index for d[] (which gives: label bond -> bond dimension)
        p *= d[trailing_zeros(mask) + 1]
        # Clear the lowest set bit, loop moves on to next bond
        # & is bitwise AND (11>1 all rest >0)
        mask &= mask - 1
    end
    return p
end

# Turn a list of cut labels e.g. [:L5, :L4], into a bit mask (one on the index bond that is cut)
_os_cutmask(cuts) = reduce(|, (UInt64(1) << (ONESITE_LABEL_INDEX[c] - 1) for c in cuts); init = UInt64(0))

# This function recursively evaluates one (sub)tree and returns 
# (flops, peak, sum of intermediate sizes, size of this node, open-bond mask)
function _os_ev(@nospecialize(t), d::Vector{Float64}, nk::UInt64, best::Bool)
    # t = tree
    # nk = mask of bonds that are NOT cut
    # best = pick cheaper child order or not?

    # If the subtree is simply a single leaf, then we return all 0's and bond mask
    t isa Int && return (0.0, 0.0, 0.0, 0.0, _os_LEAFMASK[t])

    # Evaluate the left and right child of t, and unpack their results
    f1, p1, q1, s1, m1 = _os_ev(t[1], d, nk, best)
    f2, p2, q2, s2, m2 = _os_ev(t[2], d, nk, best)

    # XOR of the child masks to find open bonds of the result
    mo = m1 ⊻ m2
    # Bonds in both operands are contracted away and drop out
    so = _os_lp(mo & nk, d)

    # Calculate the peak meamory usage given that left tensor gets computed first in @tensor
    # while computing the left child, the peak is p1;
    # then the left result is held while the right is computed: s1 + p2;
    # then both operands and the output exist at once: s1 + s2 + so.
    pk = max(p1, s1 + p2, s1 + s2 + so)

    # If requested we could also calculate the right tensor first in @tensor, if this makes big difference
    if best
        pk = min(pk, max(p2, s2 + p1, s1 + s2 + so))
    end

    # Return recursively
    return (f1 + f2 + _os_lp((m1 | m2) & nk, d), pk, q1 + q2 + so, so, mo)
end


# Given a set of cut bonds, return: flops, peak and number of slices of that tree
function onesite_cost(@nospecialize(tree), d::Vector{Float64}; cuts = Symbol[], order::Symbol = :written)
    # Make a mask of the cuts
    K = _os_cutmask(cuts)
    # Create nk (uncut bonds) from _os_ALLBITS & ~K
    # Run the above cost function, and keep the flops and peak memory usage
    f, pk, _, _, _ = _os_ev(tree, d, _os_ALLBITS & ~K, order === :best)

    # Calculate the product of the cut dimensions, this gives number of slices
    slices = _os_lp(K, d)

    # total flops = f * slices
    return (flops = f * slices, peak = pk, slices = slices, flops_per_slice = f)
end

# Return an equivalent tree whose written child order already achieves the :best peak
# aka: permute some tensors, such that we can reuse the @tensor left operation first logic.
function onesite_reorder(@nospecialize(tree), d::Vector{Float64}; cuts = Symbol[])
    # build uncut mask as above and return only the first element
    return _os_best_order(tree, d, _os_ALLBITS & ~_os_cutmask(cuts))[1]
end

# Recursive function for onesite_reorder, it returns: (reordered tree, peak, size, mask)
function _os_best_order(@nospecialize(t), d, nk)
    # If t is a leaf, just return
    t isa Int && return (t, 0.0, 0.0, _os_LEAFMASK[t])

    # Recursively reorder both children and get their peaks and sizes
    A, pa, sa, ma = _os_best_order(t[1], d, nk)
    B, pb, sb, mb = _os_best_order(t[2], d, nk)

    # Calculate the open bounds and the size of the result
    mo = ma ⊻ mb
    so = _os_lp(mo & nk, d)

    # Now we can calculate the peak memory size for 2 possible orders of contracting the 2 children
    # if A goes first
    o1 = max(pa, sa + pb, sa + sb + so)
    # if B goes first
    o2 = max(pb, sb + pa, sa + sb + so)

    # Depending on which peak memory size is bigger, replace the 2 children or not
    return o1 <= o2 ? ((A, B), o1, so, mo) : ((B, A), o2, so, mo)
end

# Detect a possible outer product step (we don't want this), returns Boolean
_os_has_outer(@nospecialize(t)) = _os_outer(t)[1]

# Recursive function
function _os_outer(@nospecialize(t))
    # if t is leaf, then no outer
    t isa Int && return (false, _os_LEAFMASK[t])

    # Check both children of t
    # o: True (if outer)/False
    o1, m1 = _os_outer(t[1]); o2, m2 = _os_outer(t[2])

    # If the child already had an outer product o1 or o2, or the 2 children share no open bond (m1 & m2 == 0)
    # Second element is the open mask of the result.
    return (o1 || o2 || (m1 & m2) == 0, m1 ⊻ m2)
end

# ======================================================================================================================
# 5. TREES AS TEXT (plan file) and validation of a tree
# ======================================================================================================================

# Convert the parsed form of a string like "((1, 2), 3)" into a real nested tuple
# If x is a leaf, then let it be
_os_tree_from_expr(x::Int) = x
function _os_tree_from_expr(ex)
    # Check if ex has the right Expr structure:
    # e.g. Meta.parse("((1, 2), 3)") returns Expr with head == :tuple and args == [Expr(:tuple, 1, 2), 3]
    (ex isa Expr && ex.head === :tuple && length(ex.args) == 2) || error("a contraction tree must consist of pairs, got: $ex")
    
    # Return a tuple, which gets recursively run
    return (_os_tree_from_expr(ex.args[1]), _os_tree_from_expr(ex.args[2]))
end

# Return all leaf numbers in a tree
# If t is already a leaf, just return t
_os_leaves(t::Int) = (t,)
# If t has 2 children, take them, and splice them into one flat tuple, and recursively run
_os_leaves(@nospecialize(t)) = (_os_leaves(t[1])..., _os_leaves(t[2])...)


# Given a tree represented as a string `"((1, 2), (3, 4))"` build a tree, rejects invalid trees
function onesite_parse_tree(s::AbstractString)
    # go from string to Expr struct to tree struct:
    t = _os_tree_from_expr(Meta.parse(s))

    # Perform checks to see if t is valid tree
    sort!(collect(_os_leaves(t))) == collect(1:ONESITE_NLEAF) || error("tree must contain each leaf 1..$ONESITE_NLEAF exactly once")
    _os_has_outer(t) && error("tree contains an outer-product step (two operands without a common bond)")
    return t
end


# ======================================================================================================================
# 6. BEST CUTS of a FIXED tree by exhaustive (brute force) enumeration
# ======================================================================================================================

# Call f once for every way to choose k items from list 1-n, argument of k is the index vector (e.g. [3, 17, 40])
function _os_each_combo(f, n::Int, k::Int)
    # Only one 0-subset:
    if k == 0
        f(Int[]); return
    end

    # Start with the first combination [1, 2, ..., k]
    idx = collect(1:k)

    # Now we perform while loop
    while true
        # Run f(idx)
        f(idx)

        i = k

        # Check if last index of idx is n, then we can't make the last index one higher
        # If so, then we have to check the second to last index to see if it has its max value (n-1), ...
        # Once we have an index i of idx which we can raise with one, then we go further
        while i >= 1 && idx[i] == n - k + i
            i -= 1
        end

        # If no position i can be increased, stop the loop, we have done all combinations
        i == 0 && break

        # Raise idx[i] by one
        idx[i] += 1

        # Reset everything to the right of idx[i], to consecutive values
        for j in (i + 1):k
            idx[j] = idx[j - 1] + 1
        end
    end
end

# Given a fixed tree, this finds the best set of exactly k bonds to cut
# order = :written: the children of every product are evaluated in the written order, order = :best: in the order that gives the smallest peak
#   (the tree can then be rewritten with onesite_reorder(tree, d; cuts) to have this peak with the written order)
# cap = Inf: search for set that minimises the peak memory
# cap = finite: search for set with the fewest total flops among those whose peak is at most cap
function onesite_best_cuts(@nospecialize(tree), d::Vector{Float64}, k::Int; cap::Real = Inf, order::Symbol = :written)
    # more than 5 cuts is too many combinations for brute force (use fewer cuts, or a plan from development/find_orders_ome.jl)
    k <= 5 || error("k > 5 would take very long (C(54,6) = 25.8 million cut sets); use at most 5 cuts")
    
    # nl = 54 = number of bonds, best = best candidate so far, bestkey is its score
    nl = length(ONESITE_LABELS); best = nothing; bestkey = (Inf, Inf)
    
    # Calculate all possible cut combinations idx and loop over them
    _os_each_combo(nl, k) do idx
        # Given idx, build the cut mask K
        K = UInt64(0)
        for i in idx
            K |= UInt64(1) << (i - 1)
        end

        # Evaluate the tree given those bonds cut
        # f = flops per slice, pk = peak memory
        f, pk, _, _, _ = _os_ev(tree, d, _os_ALLBITS & ~K, order === :best)
        
        # sl = number of slices (product of cut dimensions), ft = total flops
        sl = _os_lp(K, d); ft = f * sl
        
        # if the peak memory is below the cap: consider them as candidates
        if pk <= cap
            # If finite cap: then we prioritize flops, then peaks
            # If infinite cap: prioritize peak, then flops
            key = isfinite(cap) ? (ft, pk) : (pk, ft)

            # Now we compare the key with the best one, compare first element first then second if equal
            if key < bestkey
                # If better key, update bestkey, and update best
                bestkey = key; best = (copy(idx), sl, ft, pk)
            end
        end
    end
    
    # If cap never reached, return nothing
    best === nothing && return nothing

    # Otherwise, convert the indices back to labels and return the named tuples
    idx, sl, ft, pk = best
    return (cuts = [ONESITE_LABELS[i] for i in idx], slices = sl, flops = ft, peak = pk)
end


# ======================================================================================================================
# 7. CODE GENERATION: tree -> Julia function with the same @tensor syntax as `contract_onesite`
# ======================================================================================================================

# This function creates the @tensor text of one leaf: e.g. leaf 12 gives "A[1][UP; U9 U7 U4 U6]"
function _os_leaf_text(i::Int)
    # Given index i of the leaf, look it up
    kind, n, labs = ONESITE_TENSORS[i]

    # Depending on the kind return something:
    kind === :M && return "M[UP NORTH EAST; DOWN SOUTH WEST]"
    kind === :A && return "A[$n][$(labs[1]); $(join(labs[2:end], ' '))]"
    kind === :E && return "E[$n][$(join(labs[1:2], ' ')); $(join(labs[3:4], ' '))]"
    return "C[$n][$(join(labs, ' '))]"
end

# This function creates the indented text with explicit () of a given tree, e.g. if ind = 2: ((1, 2), (3, (4, 5))) becomes
#        (l1 * l2) *
#        (
#            l3 *
#            (l4 * l5)
#        )
function _os_render(@nospecialize(t), ind::Int)
    # ind levels of 4 spaces
    pad = "    "^ind

    # Local helper that renders one side of the product:
    # a leaf: gives its text
    operand(c) = c isa Int ? _os_leaf_text(c) :
        # a pair of two leaves: gives a compact one liner (a * b)
        (c[1] isa Int && c[2] isa Int ? "($(_os_leaf_text(c[1])) * $(_os_leaf_text(c[2])))" : "(\n" * _os_render(c, ind + 1) * "\n" * pad * ")")
        # anything bigger gives (, a newline, the recursive rendering one level deeper, a newline, the padding, and )

    # Return the full text
    return pad * operand(t[1]) * " *\n" * pad * operand(t[2])
end


# This produces the full Julia function that contracts the network in the order given by tree
function onesite_kernel_source(@nospecialize(tree); name::Symbol = :contract_onesite_kernel_custom)
    # Return the function header, @tensor allocator = allocator ( ... ), the rendered tree, closing ) and end
    return "function $name(C, E, A, M; allocator = ManualAllocator())\n    return @tensor allocator = allocator (\n" *
           _os_render(tree, 2) * "\n    )\nend\n"
end


# The two static kernels are generated HERE (at load time) from the data above, so a contraction order is defined only once:
#   contract_onesite_kernel(C, E, A, M; allocator)        = TREE_CURRENT   (`contract_onesite(env, M)` in ctmrg3D.jl calls this)
#   contract_onesite_kernel_fast(C, E, A, M; allocator)   = TREE_FAST      (only to be used together with cuts: it needs much more memory unsliced)
include_string(@__MODULE__, onesite_kernel_source(TREE_CURRENT; name = :contract_onesite_kernel))
include_string(@__MODULE__, onesite_kernel_source(TREE_FAST; name = :contract_onesite_kernel_fast))


# This will give every generated function a unique name
const _os_KERNEL_COUNTER = Ref(0)
# Generated kernels of earlier calls: tree -> function. One tree is compiled only ONCE per Julia session (compiling the 26 contractions takes a while)
const _os_KERNEL_CACHE = Dict{Any,Any}()


# This turns a tree into a callable kernel (C, E, A, M; allocator) -> scalar, without writing a file
# TREE_CURRENT and TREE_FAST give the two precompiled functions above, every other tree is generated at run time and cached.
function onesite_make_kernel(@nospecialize(tree))
    tree == TREE_CURRENT && return contract_onesite_kernel
    tree == TREE_FAST && return contract_onesite_kernel_fast

    # If tree already compiled (and in cache) return the function, otherwise go to code below
    return get!(_os_KERNEL_CACHE, tree) do
        # Increment the counter
        _os_KERNEL_COUNTER[] += 1
        # Build a unique name such as: contract_onesite_kernel_custom_3
        name = Symbol("contract_onesite_kernel_custom_", _os_KERNEL_COUNTER[])

        # Generate the source text, Meta.parse turns it into an expression
        # Core.eval(@__MODULE__, ) evaluates it in this module
        # this function definition evaluates to the function object, stored in f
        f = Core.eval(@__MODULE__, Meta.parse(onesite_kernel_source(tree; name)))

        # Return a closure with the function's signature (invokelatest because f is newer than the running world age)
        (C, E, A, M; allocator = ManualAllocator()) -> Base.invokelatest(f, C, E, A, M; allocator)
    end
end


# ======================================================================================================================
# 8. SLICING MACHINERY: turns “fix bond ℓ to piece j” into ordinary tensor operations
# ======================================================================================================================

# Given a kind (C,E,A,M) and an index i, it returns the tensor kind[i]
_os_tensor_of(C, E, A, M, kind::Symbol, i::Int) = kind === :C ? C[i] : kind === :E ? E[i] : kind === :A ? A[i] : M

# Returns the index range of chunk j when 1:m is cut into n consecutive, nearly equal chuncks
# e.g. : if m=5, n=2: j=1or2 -> Chunk 1 is 1:3 and chunk 2 is 4:5.
_os_chunk(m::Int, n::Int, j::Int) = (div(m, n) * (j - 1) + min(j - 1, m % n) + 1):(div(m, n) * j + min(j, m % n))

# Split the basis of a bond space into n consecutive groups, also works for graded spaces
# every piece is again a graded space (a symmetric subspace).
function _os_bond_pieces(Vref::ElementarySpace, n::Int)
    # Given Vref as bond space, split it up in sectors: c = sector, r = degeneracy index
    # If ℂ^χ: basis is just 1...χ
    basis = [(c, r) for c in sectors(Vref) for r in 1:dim(Vref, c)]

    # Split it up in n pieces and return
    # e.g. Z2 with 0=>2, 1=>2 and n=2: the basis is [(0,1),(0,2),(1,1),(1,2)], 
    # so piece 1 is the two even states and piece 2 is the two odd states.
    n = min(n, length(basis))
    return [basis[_os_chunk(length(basis), n, j)] for j in 1:n]
end

# Returns the subspace of Vref spanned by `piece`
function _os_piece_space(Vref::ElementarySpace, piece)
    # If plaine space:
    if Vref isa ComplexSpace
        W = ComplexSpace(length(piece))

        # if Vref is dual, then return dual(W), to keep arrow orientation
        return isdual(Vref) ? dual(W) : W
    end

    # If graded space: 
    # counts = how many states of each sector are in a a piece
    counts = Dict{Any,Int}()
    # Loop over all c = sector, r = degeneracy of piece
    for (c, r) in piece
        # keep track of amount of each sector
        counts[c] = get(counts, c, 0) + 1
    end
    # Check if Vref is dual
    flip = isdual(Vref)
    
    # Build the non-dual graded space with those multiplicities
    W = Vect[sectortype(Vref)]((flip ? dual(c) : c) => m for (c, m) in counts)

    # Return W, or dual(W) depending on flip
    return flip ? dual(W) : W
end

# returns the isometry  D ← W: W is piece space, D is full bond space either Vref (one end of the bond) or dual(Vref) (the other end)
function _os_piece_isometry(::Type{T}, Vref::ElementarySpace, piece, D::ElementarySpace) where {T}
    # Check wether D=Vref or D=dual(Vref)
    flip = (D != Vref)

    # Calculate the piece space W0
    W0 = _os_piece_space(Vref, piece)

    # Make it dual if nececssary, W should always sit in D (both dual, or both non-dual)
    W = flip ? dual(W0) : W0

    # Create a  zero tensor
    ι = zeros(T, D ← W)
    # Group the piece by sector
    rows = Dict{Any,Vector{Int}}()
    for (c, r) in piece
        # For each sector c, rows holds the degeneracy indices r 
        push!(get!(rows, c, Int[]), r)
    end

    # For all sectors
    for (c, rs) in rows
        # Create the matrix block for that sector
        b = block(ι, flip ? dual(c) : c)
        # For all degeneracy
        for (col, r) in enumerate(rs)
            # Add a 1 at (r, col)
            b[r, col] = one(T)
        end
    end
    # e.g. if ℂ⁴ and piece 3:4: ι is the 4×2 matrix with ones at (3,1) and (4,2).
    return ι
end

# Replace leg k of T (space V) by the piece space: put leg k alone in the domain, compose with ι, put it back where it was.
function _os_slice_leg(T::AbstractTensorMap, k::Int, ι::AbstractTensorMap)
    # N = total number of legs, N1 = total number in codomain
    N, N₁ = numind(T), numout(T)

    # All legs except k
    rest = Tuple(m for m in 1:N if m != k)

    # put leg k alone in the domain, all rest in codomain
    # Then multiply by ι, to go to piece space, delete that leg
    R = permute(T, (rest, (k,))) * ι

    # now we need to find the current position of all legs
    pos(m) = m < k ? m : (m == k ? N : m - 1)

    # Move every leg back to its original position and restore original codomain/domain split
    return permute(R, (Tuple(pos(m) for m in 1:N₁), Tuple(pos(m) for m in (N₁ + 1):N)))
end

# Define a container for one cut bond
struct BondCut
    ends::NTuple{2,Tuple{Symbol,Int,Int}}               # holds the two (kind, index, leg) ends of the bond
    ιs::NTuple{2,Vector{Any}}                           # ιs[e][j] = isometry for end e (1 or 2) and piece j
end

# Helper function that returns the number of pieces for a given bond cut 
npieces(b::BondCut) = length(b.ιs[1])

# Build the cut for one named bond from the real tensors, and check that the setup is consistent
function BondCut(env, M, label::Symbol, n::Int)
    # Two ends of the bond
    ends = ONESITE_BONDS[label]

    # Get the two tensors at those ends
    T1 = _os_tensor_of(env.C, env.E, env.A, M, ends[1][1], ends[1][2])
    T2 = _os_tensor_of(env.C, env.E, env.A, M, ends[2][1], ends[2][2])

    # Define the spaces of the two legs
    s1, s2 = space(T1, ends[1][3]), space(T2, ends[2][3])
    # Spaces should be the dual of each other
    s2 == dual(s1) || error("bond $label: the two legs are not dual to each other: $s1 vs $s2")
    
    # Return the spaces of s1 devided into n pieces 
    pieces = _os_bond_pieces(s1, n)

    # Define both isommetries, we use dual(si) because in _os_slice_leg we need the dual
    ι1 = Any[_os_piece_isometry(scalartype(T1), s1, p, dual(s1)) for p in pieces]
    ι2 = Any[_os_piece_isometry(scalartype(T2), s1, p, dual(s2)) for p in pieces]

    # Return all data
    return BondCut(ends, (ι1, ι2))
end


# ======================================================================================================================
# 9. ALLOCATORS (for @tensor allocator = ...)
# ======================================================================================================================

# 2: now we add two new allocators (for @tensor allocator = ), which might handle the memory allocation 
# for the repeated contractions of the different slicings better

# ManuelAllocator = default: Every temporary is malloced and freed, but this cost a lot
# ReuseAllocator: keeps freed blocks in pools per size class and hands them out again. Every slice requests the same sequence of sizes, so after the first slice nothing new is allocated.
# TrackingAllocator: wraps another allocator and counts the live and peak bytes of the temporaries.
# This also includes the allocation used for the permutation copies, so result will be higher than the naive model

# Round byte count up to a size class, so that slightly different requests share the same pool
function _os_sizeclass(nb::Int)
    # if nb is below 4kiB: round up to a multiple of 64 bytes
    nb <= 4096 && return max(64, (nb + 63) & ~63)

    # step = 2^(k-3), where k is the exponent of leading power of two
    step = 1 << (63 - leading_zeros(nb - 1) - 3)
    # nb >= 2^k

    # Return a multiple of step
    # e.g.: nb = 5000, this is above 4096, largest leading power of 2 is 2^12=4096
    # This gives k = 12, step = 2^9 = 512, closest multiple of step = 10 -> Class 5120
    return cld(nb, step) * step
end

# Define the state of the pool allocator (we want to keep pools of reusable allocated memory as long as possible)
mutable struct ReuseAllocator
    pools::Dict{Int,Vector{Ptr{Cvoid}}}     # size class in bytes -> stack of free pointers
    max_pool_bytes::Int         # free pools beyond this are given back to the OS
    max_total_bytes::Int        # max live + pooled; on a pool miss, pooled blocks are freed first when this would be exceeded
    live_bytes::Int             # size-class bytes currently handed out
    pooled_bytes::Int           # size class bytes sitting in the pools
    peak_os_bytes::Int          # Peak live+pooled asked bytes, benchmark prints this as pool GiB
    hits::Int                   # Number of Pool reuses
    misses::Int                 # Number of Pool not being able to be reused

    # Inner constructor: take two limits as keywords (default = unlimited)
    # Create an empty state with zero counters.
    function ReuseAllocator(; max_pool_bytes::Integer = typemax(Int), max_total_bytes::Integer = typemax(Int))
        a = new(Dict{Int,Vector{Ptr{Cvoid}}}(), max_pool_bytes, max_total_bytes, 0, 0, 0, 0, 0)
        return finalizer(release!, a)
    end
end

"free all pooled blocks of `a` (blocks still in use are not touched)"
function release!(a::ReuseAllocator)
    for ptrs in values(a.pools)
        foreach(Libc.free, ptrs)
        empty!(ptrs)
    end
    a.pooled_bytes = 0
    return a
end

# Free pooled bloks, largest blocks first, until need bytes have been released
# Called if max_total_bytes would be exceeded
function _os_evict!(a::ReuseAllocator, need::Int)        # free pooled blocks (largest classes first) until `need` bytes are released
    for cls in sort!(collect(keys(a.pools)); rev = true)
        ptrs = a.pools[cls]
        while !isempty(ptrs) && need > 0
            Libc.free(pop!(ptrs)); a.pooled_bytes -= cls; need -= cls
        end
        need <= 0 && break
    end
    return nothing
end

# This function hands out scratch memory every time @tensor needs space for an intermediate result
# It first checks a shelf of memory blocks that were used before and are now free
# Such that it doesn't have to ask for fresh memory
function TensorOperations.tensoralloc(::Type{A}, structure, ::Val{istemp}, a::ReuseAllocator) where {A <: AbstractArray, istemp}
    T = eltype(A)
    if istemp && isbitstype(T)
        dims = structure isa Integer ? (Int(structure),) : map(Int, Tuple(structure))
        cls = _os_sizeclass(prod(dims) * sizeof(T))
        pool = get(a.pools, cls, nothing)
        if pool !== nothing && !isempty(pool)
            p = pop!(pool); a.pooled_bytes -= cls; a.hits += 1
        else
            a.misses += 1
            excess = a.live_bytes + a.pooled_bytes + cls - a.max_total_bytes
            excess > 0 && _os_evict!(a, excess)
            p = Libc.malloc(cls)
            p == C_NULL && throw(OutOfMemoryError())
        end
        a.live_bytes += cls
        a.peak_os_bytes = max(a.peak_os_bytes, a.live_bytes + a.pooled_bytes)
        return TensorOperations.PtrArray(Ptr{T}(p), dims...; check_dims = false)
    end
    return TensorOperations.tensoralloc(A, structure, Val(istemp))        # same fall-back as ManualAllocator: ordinary (GC) array
end

# This function places memory into pools (instead of OS) for later reusing every time 
# @tensor wants to free memory
function TensorOperations.tensorfree!(C::TensorOperations.PtrArray, a::ReuseAllocator)
    cls = _os_sizeclass(length(C) * sizeof(eltype(C)))
    a.live_bytes -= cls
    if a.pooled_bytes + cls > a.max_pool_bytes
        Libc.free(C.ptr)
    else
        push!(get!(() -> Ptr{Cvoid}[], a.pools, cls), Ptr{Cvoid}(C.ptr)); a.pooled_bytes += cls
    end
    return nothing
end

# Define a container for a wrapper Allocator that just counts bytes and forwards the real allocations to
# inner (can be either ManualAllocator or ReuseAllocator)
mutable struct TrackingAllocator{A}
    inner::A                # ManualAllocator or ReuseAllocator
    live_bytes::Int         # temporary bytes currently active
    peak_bytes::Int         # max of above so far
    nalloc::Int             # number of tempraries
    total_bytes::Int        # total bytes requested
    sizes::Dict{UInt,Int}   # pointer adress -> bytes
end

# Outer constructor
TrackingAllocator(inner = ManualAllocator()) = TrackingAllocator(inner, 0, 0, 0, 0, Dict{UInt,Int}())


# If TrackingAllocator: this allocates through inner, then records the allocation
function TensorOperations.tensoralloc(::Type{A}, structure, ::Val{istemp}, t::TrackingAllocator) where {A <: AbstractArray, istemp}
    C = TensorOperations.tensoralloc(A, structure, Val(istemp), t.inner)
    if istemp
        nb = length(C) * sizeof(eltype(C))
        if nb > 0
            t.sizes[UInt(pointer(C))] = nb
            t.live_bytes += nb; t.nalloc += 1; t.total_bytes += nb
            t.peak_bytes = max(t.peak_bytes, t.live_bytes)
        end
    end
    return C
end

# If TrackingAllocator: uncounts the blocks, then frees through inner
function TensorOperations.tensorfree!(C::AbstractArray, t::TrackingAllocator)
    nb = pop!(t.sizes, UInt(pointer(C)), 0)
    t.live_bytes -= nb
    TensorOperations.tensorfree!(C, t.inner)
    return nothing
end


# ======================================================================================================================
# 10. THE SLICED CONTRACTION
# ======================================================================================================================

# Turn the kernel argument into a function with signature: 
# (C, E, A, M; allocator) -> scalar
function _os_resolve_kernel(kernel)
    # If kernel is already a function, it just returns it
    kernel isa Function && return kernel
    # if kernel is :current or :fast then it returns the above defined functions
    kernel === :current && return contract_onesite_kernel
    kernel === :fast && return contract_onesite_kernel_fast

    # If kernel is the contraction tree (nested pairs of leaf ids): call onesite_make_kernel(kernel)
    # This makes a new function
    kernel isa Tuple && return onesite_make_kernel(kernel)

    # All rest raises error
    error("kernel must be a function, :current, :fast or a contraction tree; got $(typeof(kernel))")
end

# Convert bond label (which we want to slice) into pairs (label -> n) where n is the bond dimension
# e.g. [:L5 => 8, :L4 => 5, :A8 => 5]
function onesite_full_cuts(env, M, labels)
    # Read the 54 bond dimensions from the actual tensors 
    d = onesite_dims(env, M)

    # For all bonds in labels, find the index of that label in the full bond list
    # Then we calculate its bond dimension via d[]
    # Then we return the pair e.g. :L5 => 8
    return Pair{Symbol,Int}[Symbol(l) => round(Int, d[ONESITE_LABEL_INDEX[Symbol(l)]]) for l in labels]
end


# Count the number of slices, by building BondCut for each (label, n), then seeing what npieces is (npieces <= n) and multiplying
onesite_nslices(env, M, cuts) = prod((npieces(BondCut(env, M, lab, n)) for (lab, n) in cuts); init = 1)


# IMPORTANT: this function computes the same scalar as contract_onesite(env, M), as a serial sum over slices.
function contract_onesite_sliced(env, M, cuts::AbstractVector{<:Pair{Symbol,Int}};
                                 kernel = contract_onesite_kernel, allocator = ManualAllocator(), max_slices::Int = typemax(Int),
                                 verbosity::Int = 0)
    # cuts = vector of (label => n) pairs
    # kernel = contraction order
    # allocator = chosen allocator
    # max_slices = makes result a partial sum
    # verbosity = if > 2: logs the value of every slice and the accumulated sum
    
    # Create the kernel once, for the whole call
    kern = _os_resolve_kernel(kernel)

    # If no cuts, just perform and return the contraction
    isempty(cuts) && return kern(env.C, env.E, env.A, M; allocator)

    # Build all cuts, with isometries for all pieces, once
    bcs = [BondCut(env, M, lab, n) for (lab, n) in cuts]
    
    # Running sum (accumulated result)
    acc = zero(scalartype(M))

    # Total number of slices that will actually run (needed for the progress message)
    nslices = min(prod(npieces(b) for b in bcs), max_slices)

    # Now we will replace tensors one by one with their cut version:
    # For all slices (for all possible combinations of indexes per cut bond)
    for (n, js) in enumerate(Iterators.product((1:npieces(b) for b in bcs)...))
        # if n > max_slices then break and return the acc before full run
        n > max_slices && break

        # Copy the vectors, not the tensors. Replacing an entry never changes env.
        C, E, A, Mj = copy(env.C), copy(env.E), copy(env.A), M

        # For a given slice: run over all cut bonds b (with chosen piece j) and over both ends of that bond
        for (b, j) in zip(bcs, js), e in 1:2

            # Find out which tensor and which leg we are talking about
            kind, i, leg = b.ends[e]

            # Define the isometry for that end and piece
            ι = b.ιs[e][j]

            # Replace the tensor at that end by the sliced tensor
            # Even works if tensor is already sliced previously by other bond
            if kind === :C
                C[i] = _os_slice_leg(C[i], leg, ι)
            elseif kind === :E
                E[i] = _os_slice_leg(E[i], leg, ι)
            elseif kind === :A
                A[i] = _os_slice_leg(A[i], leg, ι)
            else
                Mj = _os_slice_leg(Mj, leg, ι)
            end
        end
        
        # Contract the sliced network with the new tensors, and time it
        t0 = time_ns()
        zslice = kern(C, E, A, Mj; allocator)
        dt = (time_ns() - t0) / 1e9
        acc += zslice

        # Per-slice progress: value of this slice and the accumulated sum so far
        verbosity > 2 && @info "contract_onesite_sliced: slice $n/$nslices, pieces $(js), value = $zslice, accumulated = $acc, $(round(dt; digits = 2)) s"
    end

    # Return the answer
    return acc
end

# Performs a correctness check, compares the sliced result with a reference value
function check_slicing(env, M, cuts; rtol = 1e-8, reference = nothing, kwargs...)
    # If no reference was given, then we compare to the result of the unsliced contract_onesite_kernel answer
    z0 = reference === nothing ? contract_onesite_kernel(env.C, env.E, env.A, M) :
         reference isa Function ? reference(env, M) : reference
    # if reference was a function, then we use reference(env, M)

    # Now we calculate the sliced result
    z1 = contract_onesite_sliced(env, M, cuts; kwargs...)

    # And compare both results
    return isapprox(z0, z1; rtol), z0, z1
end


# ======================================================================================================================
# 11. MOCK ENVIRONMENT: random tensors with the leg structure of the 27 tensors (no CTMRG3D needed)
# ======================================================================================================================

struct OnesiteMockEnv
    C::Vector{Any}
    E::Vector{Any}
    A::Vector{Any}
end

# This builds random tensors with the correct leg structure and bond classes, returning (env, M)
function onesite_mock_env(Venv, Vpeps, VD; T = ComplexF64, seed = 1)
    Random.seed!(seed)

    # Dict mapping the three bond classes to the corresponding bond spaces
    space_of = Dict(:env => Venv, :peps => Vpeps, :D => VD)
    
    # legspace: maps (kind, index, leg position) to a space
    legspace = Dict{Tuple{Symbol,Int,Int},Any}()
    # Loop over all bounds
    for (lab, (e1, e2)) in ONESITE_BONDS
        V = space_of[onesite_bond_class(lab)]
        # For each bond: first end gets V, and second end gets dual(V)
        legspace[e1] = V; legspace[e2] = dual(V)
        # This does not automatically correspond to arrow convention
    end

    # number of codomain legs per kind
    nout = Dict(:C => 0, :E => 2, :A => 1, :M => 0)
    
    # legs maps (kind, index) to the leg/label list of ONESITE_TENSORS
    legs = Dict((kind, i) => labs for (kind, i, labs) in ONESITE_TENSORS)

    # Now we can build all tensors of a given kind (C, E, A, ..)
    function build(kind, i)
        # list the leg spaces of a given kind in order of leg indices
        s = [legspace[(kind, i, k)] for k in 1:length(legs[(kind, i)])]
        
        # define total and codomain leg counts 
        N = length(s); N₁ = nout[kind]

        # Create a random symmetric map with the first leg in the codomain and rest in domain (dual)
        t = randn(T, s[1] ← ⊗(dual.(s[2:N])...))

        # Now we permute the legs, such that the N1 is correct
        return permute(t, (Tuple(1:N₁), Tuple((N₁ + 1):N)))
    end

    # Return the struct with 8 corners, 12 edges, 6 faces, and M
    return OnesiteMockEnv(Any[build(:C, i) for i in 1:8], Any[build(:E, i) for i in 1:12], Any[build(:A, i) for i in 1:6]), build(:M, 1)
end


# ======================================================================================================================
# 12. TWO-SITE OBSERVABLES: only described here, NOT implemented
# ======================================================================================================================
# A two-site observable <O_1 O_2> of two neighbouring sites (say along x) needs a bigger closed network than the 3x3x3 one above: a 4x3x3 grid.
# Positions along x are 1..4, the two M tensors (observable sites) sit at x = 2, 3 and y = z = 2:
#      2 M
#     10 faces A  (2 at the x ends, 8 next to the two M in the +-y and +-z directions)
#     16 edges E  (8 at the x ends, 8 along the long axis)
#      8 corners C
#   = 36 tensors and 75 bonds  (3x3x3: 27 tensors and 54 bonds).
#   OPEN QUESTION (CTMRG side): the environment stores the C, E, A tensors for ONE site. The ones at the two middle positions (x = 2, 3) have to be
#   built from it; this is the part that needs a change in ctmrg3D.jl/ctmrg_cubic.jl, and it is not done.
# What has to be done on the contraction side (all with the tools of this file):
#   1. A table ONESITE2_TENSORS with the 36 rows (kind, index, leg labels), built like ONESITE_TENSORS; the bond classes env / peps / D stay the same.
#   2. The cost model uses bit masks with one bit per bond: 75 bonds do not fit in a UInt64, use UInt128. _os_ev, _os_best_order, _os_each_combo,
#      onesite_dims and onesite_bond_class must then take the table / label list / masks as arguments instead of the global constants.
#   3. A contraction tree of the 36 tensors: onesite_find_trees(d) works unchanged once `code` is built from ONESITE2_TENSORS (OMEinsumContractionOrders
#      does not care what network it gets). Do this ONCE per (chi_env, chi_peps, D) and keep the best tree as TREE2_CURRENT (like TREE_CURRENT above).
#   4. onesite_kernel_source / _os_leaf_text generate the @tensor code of the new tree from the table.
#   5. contract_onesite_sliced needs only the table of bond ends (ONESITE_BONDS) of the new network.
#   6. onesite_best_cuts on the fixed tree: C(75,k) = 75, 2775, 67525, 1.2e6, 1.7e7 candidate cut sets for k = 1..5. Each candidate costs one evaluation of
#      the cost model, which is tiny compared to the contraction itself. Measured for the 54 bonds here: 14 microseconds per candidate
#      (k = 3: 0.35 s, k = 4 (316251 sets): 4 s); for 75 bonds expect about 1.5x more per candidate, so k = 4 in the order of 30 s and k = 5 minutes.
#   7. Plan file: add a column "network" (3x3x3 or 4x3x3) so that both live in one csv.
