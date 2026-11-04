# Goal of this file is to perform automatic slicing for `contract_onesite`, given a RAM budget
# It picks the smallest number of cuts (0..kmax) whose predicted memory fits,
# then run the sliced contraction.  Serial, single process

# The memory rules below are emperical/hand wavy (but they work on my windows laptop).

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


# 1. the plan: which bonds to cut
# Given a cut set, evaluate it, compute its cost with and without the cuts
# Return some data in Tuple
function onesite_plan_of(d::Vector{Float64}, cuts; ram_gib::Real, bytes::Integer = 16, allocator::Symbol = :reuse,
                         margin::Real = 0.8, rules = ONESITE_RAM_RULES, tree = TREE_CURRENT)
    # bytes depends on what type of number: ComplexF64 is 16, Float64 is 8
    # d is vector of 54 bond dimensions, cuts is one combination of bond labels to cut
    # margin = fraction of RAM we allow to be taken by this

    # Check if allocator is one of reuse/manual
    _os_check_alloc(allocator)
    
    # Make a Vector of the cuts
    cs = Symbol[Symbol(c) for c in cuts]
    # Calculate the cost of the unsliced network
    c0 = onesite_cost(tree, d)
    # Calculate the cost of the cutted network
    c = onesite_cost(tree, d; cuts = cs)

    # Calculate the theoretical expected max peak in GiB
    peak_gib = c.peak * bytes / 2^30

    # Calculate the estimated peak in GiB
    est = rules.baseline_gib + rules[allocator] * peak_gib

    # Return a Tuple: cuts, amount of cuts, product of cut dimensions, relative needed flops, fits=true if est <= margin * ram_gib
    return (cuts = cs, k = length(cs), slices = c.slices, flops_rel = c.flops / c0.flops, peak_gib = peak_gib,
            est_gib = est, fits = est <= margin * ram_gib)
end


# Search over different amount of cuts k=0 to kmax (if nothing fits, it tells us) 
# Return its plan  
function onesite_choose_cuts(d::Vector{Float64}; ram_gib::Real, bytes::Integer = 16, allocator::Symbol = :reuse,
                             kmax::Int = 3, margin::Real = 0.8, rules = ONESITE_RAM_RULES, tree = TREE_CURRENT)
    # Do some checks
    _os_check_alloc(allocator)
    0 <= kmax <= 5 || error("need 0 <= kmax <= 5, got kmax = $kmax")
    
    # Calculate the max number of allowed elements = cap
    allowed_gib = (margin * ram_gib - rules.baseline_gib) / rules[allocator]
    cap = allowed_gib * 2^30 / bytes

    # Define keywords for onesite_plan_of function
    kw = (; ram_gib, bytes, allocator, margin, rules, tree)
    # Loop over all amount of cuts, stop at lowest k that works
    for k in 0:kmax
        # If no cut
        cuts = if k == 0
            # If within cap: use no cuts 'Symbol[]', otherwise nothing
            onesite_cost(tree, d).peak <= cap ? Symbol[] : nothing
        # if cut
        else
            # Find the best cuts (lowest FLOPS, where peak<cap)
            r = onesite_best_cuts(tree, d, k; cap)
            # if no r found, do nothing, otherwise return r.cuts
            r === nothing ? nothing : r.cuts
        end
        
        # As soon as a cut if found (for lowest k), return its plan
        cuts === nothing || return onesite_plan_of(d, cuts; kw...)
    end

    # If nothing fitted
    # if also no cuts allowed, then we just use an empty cut list
    # Otherwise we use kmax cuts, and find the best contraction/slicing with the lowest peak memory
    cuts = kmax == 0 ? Symbol[] : onesite_best_cuts(tree, d, kmax).cuts
    # Now we return the plan, fits=False will hold
    return onesite_plan_of(d, cuts; kw...)                                         
end


# This computes the same number as contract_onesite(env, M), but using slices, number of slices chosen automatically from RAM budget
function contract_onesite_auto(env, M; ram_gib::Real = Sys.total_memory() / 2^30, allocator::Symbol = :reuse, cuts = nothing,
                               kmax::Int = 3, margin::Real = 0.8, rules = ONESITE_RAM_RULES, force::Bool = false, verbosity::Int = 0)
    
    # ram_gib: memory this PROCESS may use. Default=Sys.total_memory(), pass it explicitely if needed (e.g. = 64)
    # allocator: :reuse or :manuel (:reuse = `ReuseAllocator`: 13-19 % faster than `:manual` in the Windows runs)
    # cuts: force a cut set (vector of labels such as `[:A9, :L5]`) instead of choosing one; it is still checked against the budget.
    # kmax: largest number of cuts to try (<= 5).  `margin`: fraction of `ram_gib` the prediction may fill.
    # force: run even if nothing is predicted to fit (will probably swap or be killed).
    
    # Check if allowed allocator
    _os_check_alloc(allocator)
    # Vector of dimensions for all bounds
    d = onesite_dims(env, M)
    # Define number of bytes per element (8 for Float64, 16 for ComplexF64)
    bytes = sizeof(scalartype(M))
    
    plan = cuts === nothing ?
        # If no cuts were given initially, search for best cuts
        onesite_choose_cuts(d; ram_gib, bytes, allocator, kmax, margin, rules) :
        # If cuts were given, check its plan and memory cost
        onesite_plan_of(d, cuts; ram_gib, bytes, allocator, margin, rules)
    
    # if the memory cost is too high (and we don't force it)    
    if !plan.fits && !force
        # Give error
        error("contract_onesite_auto: no cut set with at most $(cuts === nothing ? kmax : length(cuts)) cuts is predicted to fit ",
              "$(round(margin * ram_gib; digits = 2)) GiB (best: cuts $(plan.cuts), predicted $(round(plan.est_gib; digits = 2)) GiB). ",
              "Give more `ram_gib`, a larger `kmax` (<= 5), or find more cuts with development/find_orders_cotengra.py; `force = true` runs anyway.")
    end

    # Otherwise give info (not if plan doesn't fit, we already have above error for this)
    if !plan.fits && verbosity > 0
        @warn "contract_onesite_auto: predicted $(round(plan.est_gib; digits = 2)) GiB > $(round(margin * ram_gib; digits = 2)) GiB budget, running anyway (force = true)"
    end
    # Choose the correct allocator
    alloc = allocator === :reuse ? ReuseAllocator() : ManualAllocator()
    
    # Perform the sliced contraction
    try
        return contract_onesite_sliced(env, M, onesite_full_cuts(env, M, plan.cuts); allocator = alloc, verbosity)
    finally
        # Release memory if not ReuseAllocator
        allocator === :reuse && release!(alloc)
    end
end

# Here we let one key word, ram_gib, decide wether to use the old contract_onesite function or the sliced contract_onesite_auto function
contract_onesite_dispatch(env, M, ram_gib::Nothing; kwargs...) = contract_onesite(env, M)
contract_onesite_dispatch(env, M, ram_gib::Real; kwargs...) = contract_onesite_auto(env, M; ram_gib, kwargs...)
