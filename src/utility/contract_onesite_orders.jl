# WHAT THIS FILE DOES
#   * describes the closed 27-tensor network of `contract_onesite` as a table: (tensors, legs, bonds, bond classes);
#   * a dense cost model for ANY pairwise contraction order ("tree") with an optional set of SLICED bonds (cuts):
#        onesite_cost(tree, d; cuts)      -> flops, peak live memory (elements), number of slices

# COST MODEL:
#   cost of a pairwise step = product of the dimensions of all DISTINCT indices of both operands (dense tensors);
#   peak = intermediate memory peak (the 27 inputs are not counted):  peak(node) = max(p1, s1 + p2, s1 + s2 + s_out).
#   Slicing a bond: its dimension becomes 1 inside the loop, total flops = prod(cut dims) * flops(sliced network).
#   It does NOT model: GEMM efficiency, permutation copies inside TensorKit, allocation/page-fault cost.  LESS FLOPS /= FASTER


using TensorKit, TensorOperations
using TensorOperations: ManualAllocator


# Each row is: kind tensor (corner, face, ...), index of that kind, leg indices (all based on order in `contract_onesite`)
# e.g. (:E, 1, [:U2, :A2, :U1, :U4]) = env.E[1][U2 A2; U1 U4]
const ONESITE_TENSORS = Tuple{Symbol,Int,Vector{Symbol}}[
    (:C, 1, [:U1, :A1, :U3]),
    (:E, 1, [:U2, :A2, :U1, :U4]),
    (:C, 2, [:U5, :A3, :U2]),
    (:E, 5, [:A1, :M1, :B1, :M3]),
    (:C, 5, [:B1, :L1, :L3]),
    (:A, 2, [:NORTH, :A2, :M2, :B2, :M1]),
    (:E, 6, [:A3, :M5, :B3, :M2]),
    (:E, 9, [:L1, :B2, :L2, :L4]),
    (:C, 6, [:B3, :L5, :L2]),
    (:E, 2, [:U10, :A6, :U5, :U7]),
    (:C, 3, [:U12, :A9, :U10]),
    (:A, 1, [:UP, :U9, :U7, :U4, :U6]),
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
    (:E, 7, [:A9, :M12, :B9, :M10]),
    (:A, 6, [:DOWN, :L4, :L7, :L9, :L6]),
    (:E, 11, [:L12, :B8, :L11, :L9]),
    (:E, 10, [:L5, :B6, :L10, :L7]),
    (:C, 7, [:B9, :L12, :L10]),
]
# Number of leafs of contraction tree = number of one site tensors = 27 in our case
const ONESITE_NLEAF = length(ONESITE_TENSORS)

# Create a dict: bound label (lab, e.g. :L10) -> list of ends (kind, index, legposition, e.g. [C, 7, 3] and [E, 10, 3])
function _onesite_build_bonds()
    ends = Dict{Symbol,Vector{Tuple{Symbol,Int,Int}}}()
    # Loop over all tensors, and for each tensor loop over all legs/labels
    for (kind, i, labs) in ONESITE_TENSORS, (k, lab) in enumerate(labs)
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
# Mask for all leafs
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

#Vector of the dimensions of the 54 bonds (ordered like `ONESITE_LABELS`).  The second form reads the TOTAL dimension of every bond
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


# Now we define 2 reference trees, we write them as nested tuples ((a, b) = "contract a with b", `a` is evaluated first by @tensor)
"Current contraction used in `contract_onesite`: memory-lean (39.1 GiB for unsliced 8,5,5)"
const TREE_CURRENT = (1, (2, (3, (((4, 5), (6, (7, (8, 9)))), (((10, 11), (12, (13, (14, 15)))), ((16, (17, (18, 19))), (20, (21, (22, (23, (24, (25, (26, 27)))))))))))))
"Contraction found by opt=true: less FLOPS needed (doesn't automatically mean faster though!!), but super memory intensive (238 GiB for unsliced 8,5,5)"
const TREE_FAST = (((((20, (((24, ((((25, 19), 27), ((17, 15), (((23, 11), 22), 14))), (((4, (13, 1)), 16), (18, 5)))), 8), (9, 26))), 6), 21), ((12, 10), (2, 3))), 7)


# Now we define a cost model

# product of the dimensions of the bonds set to one in `mask`
@inline function _os_lp(mask::UInt64, d::Vector{Float64})
    p = 1.0
    # Loop over all bits in mask
    while mask != 0
        # trailing_zeros(mask) + 1: brings us from mask bit to an index for d[] (which gives: label bond -> bond dimension)
        p *= d[trailing_zeros(mask) + 1]
        # Clear the lowest set bit, loop moves on to next bond
        mask &= mask - 1
    end
    return p
end

# Turn a list of cut labels e.g. [:L5, :L4], into a bit mask (one one index bond that is cut)
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

# 4. reading / writing the files that connect this code to development/find_orders_cotengra.py

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
# If t has 2 children, take them, and splice them into one flat tuple, and recursively tun
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


# Read the TSV written by find_orders_cotengra.py and transform into a vector of named tuples:
# e.g. current/unsliced<TAB><TAB>(1, (2, (3, ...)))<TAB>2.272612e+12<TAB>2.602000e+09<TAB>1
# to: Vector{NamedTuple}:  (name = "current/unsliced", cuts = Symbol[], tree = (1, (2, (3, ...))), ctg_flops = 2.272612e12, ctg_peak = 2.602e9, slices = 1)
function onesite_read_orders(path::AbstractString)
    out = NamedTuple[]

    # Loop over all lines in TSV
    for line in eachline(path)
        # Skip header files
        (isempty(strip(line)) || startswith(line, "#")) && continue
        
        # Split row if tab
        f = split(line, '\t')
        # Should be 6 elements per row
        length(f) == 6 || error("expected 6 tab-separated columns, got $(length(f)): $line")
        
        # Column 2: comma seperated list of cut indices
        cuts = Symbol[Symbol(c) for c in split(f[2], ',') if !isempty(c)]

        # for all cuts, check if it corresponds to a real bond label
        for c in cuts
            haskey(ONESITE_LABEL_INDEX, c) || error("unknown bond label $c in $path")
        end

        # Adds a named tuple to out
        push!(out, (name = String(f[1]), cuts = cuts, tree = onesite_parse_tree(f[3]),
                    ctg_flops = parse(Float64, f[4]), ctg_peak = parse(Float64, f[5]), slices = parse(Int, f[6])))
    end
    return out
end

# Given a vector d with for each bond a bond dimension
# Write the 54 bond dimensions to a text file, so python cal use the real dimensions of the real environment
function onesite_write_dims(path::AbstractString, d::Vector{Float64})
    open(path, "w") do io
        # Write a comment header
        println(io, "# bond dimensions of the contract_onesite network (total dimension of each bond space)")
        
        # Pair each label with its dimension
        for (l, x) in zip(ONESITE_LABELS, d)
            # Write a line: label dim (e.g. L5 8)
            println(io, l, " ", round(Int, x))
        end
    end
    return path
end


# Define a table of interesting cut sets, with corresponding names
const ONESITE_TESTED_CUTS = (
    current = [Symbol[], [:L5], [:L5, :L4], [:L5, :L4, :A8]],
    fast = [Symbol[], [:U10], [:U1, :U10], [:U10, :U6, :U9]],
)


# 5. best cut sets of a FIXED tree by exhaustive enumeration (Julia-side counterpart of cotengra's slicing)

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

        # 
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
# cap = Inf: search for set that minimises the peak memory
# cap = finite: search for set with the fewest total flops among those whose peak is at most cap
function onesite_best_cuts(@nospecialize(tree), d::Vector{Float64}, k::Int; cap::Real = Inf, order::Symbol = :written)
    # if number of allowed cuts is above 5 than you should use contengra, too many combinations otherwise
    k <= 5 || error("k > 5 would take very long; use cotengra (development/find_orders_cotengra.py) instead")
    
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


# 6. code generation: tree -> Julia function with the same @tensor syntax as `contract_onesite`

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


# Given the above string, save it to a file, such that we can read/include it later on
function onesite_write_kernel_file(path::AbstractString, @nospecialize(tree); name::Symbol = :contract_onesite_kernel_custom)
    # Open the file at path
    open(path, "w") do io
        # Write a comment line 
        write(io, "# generated by onesite_write_kernel_file; needs `using TensorOperations: ManualAllocator` and @tensor in scope\n")
        # Then write the above source string
        write(io, onesite_kernel_source(tree; name))
    end
    return path
end


# This will give every generated function a unique name 
const _os_KERNEL_COUNTER = Ref(0)

# This turns a tree into a callable kernel at run time, without writing a file
function onesite_make_kernel(@nospecialize(tree))
    # Increment the counter
    _os_KERNEL_COUNTER[] += 1
    # Build a unique name such as: contract_onesite_kernel_custom_3
    name = Symbol("contract_onesite_kernel_custom_", _os_KERNEL_COUNTER[])

    # Generate the source text, Meta.parse turns it into an expression
    # core.eval(@__MODULE__, ) evaluates it in this module
    # this function definition evaluates to the function object, stored in f
    f = Core.eval(@__MODULE__, Meta.parse(onesite_kernel_source(tree; name)))

    # Return a closure with the function's signature, 
    return (C, E, A, M; allocator = ManualAllocator()) -> Base.invokelatest(f, C, E, A, M; allocator)
end


# 7. self check: the cost model against the numbers of the Python model it was 
# ported from (dense, (χenv, χpeps, D) = (8, 5, 5))

# Here we define a table of reference results (from Python model)
const _os_SELFCHECK = [
    # name, tree, cuts, flops, peak (elements), slices
    ("current (no cuts)", TREE_CURRENT, Symbol[], 2272611776512.0, 2627200000.0, 1.0),
    ("current + L5", TREE_CURRENT, Symbol[:L5], 2276800938496.0, 341000000.0, 8.0),
    ("current + L5,L4", TREE_CURRENT, Symbol[:L5, :L4], 2295962042880.0, 79720000.0, 40.0),
    ("current + L5,L4,A8", TREE_CURRENT, Symbol[:L5, :L4, :A8], 2323064230400.0, 17480000.0, 200.0),
    ("fast (no cuts)", TREE_FAST, Symbol[], 1512621097600.0, 16000000000.0, 1.0),
    ("fast + U10,U6,U9", TREE_FAST, Symbol[:U10, :U6, :U9], 1553400616000.0, 80000000.0, 200.0),
    ("fast + U1,U10", TREE_FAST, Symbol[:U1, :U10], 1549349683200.0, 250000000.0, 64.0),
]


# Returns true if Julia cost model reproduces all reference numbers
function onesite_selfcheck(; verbose::Int = 1)
    # define the dimenions of the reference scenario
    d = onesite_dims(8, 5, 5)
    ok = true

    # For all reference values
    for (name, tree, cuts, f, p, s) in _os_SELFCHECK
        # Calculate its cost using our Julia function
        c = onesite_cost(tree, d; cuts)
        
        # Check flops, peak and slices against reference values 
        good = isapprox(c.flops, f; rtol = 1e-9) && isapprox(c.peak, p; rtol = 1e-9) && isapprox(c.slices, s; rtol = 1e-9)
        
        # Only if everything is good, then ok = True
        ok &= good

        # If verbose is on, print data per reference value
        verbose > 0 && println(rpad(name, 30), " flops ", c.flops, " peak ", c.peak, " (", round(c.peak * 16 / 2^30, digits = 3), " GiB) slices ", round(Int, c.slices),
                           good ? "   OK" : "   MISMATCH (expected $f, $p, $s)")
    end
    return ok
end
