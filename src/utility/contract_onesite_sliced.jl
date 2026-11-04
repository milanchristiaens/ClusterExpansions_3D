# Idea ("slicing", Gray & Kourtis, Quantum 5, 410 (2021), Sec. 4.7.1): the closed 27-tensor network is a sum over every bond.
# Pick a few bonds, split each bond space into n pieces, contract the network once per combination of pieces (each contraction
# sees the cut bonds with the small piece space) and add the scalars.  The peak memory of the big intermediates drops by about the
# product of the n's; the flops stay close to the original (or even drop, if the tree is re-optimised for the sliced network).
# The slices are run one after the other; not parallel (yet?)
#
# What is in this file
#   1. contract_onesite_kernel          the CURRENT bracketing of contract_onesite, written on plain vectors C, E, A, M
#      contract_onesite_kernel_fast     a different bracketing (fewer flops, much more memory when unsliced)
#      (any other tree: onesite_make_kernel(tree) from contract_onesite_orders.jl, or pass the tree itself as `kernel`)
#   2. slicing machinery  (_os_slice_leg, _os_bond_pieces, ..., BondCut)
#   3. allocators: ReuseAllocator (re-uses temporaries), TrackingAllocator (measures the peak of the temporaries)
#   4. contract_onesite_sliced(env, M, cuts; kernel, allocator), onesite_full_cuts, check_slicing
#   5. a mock environment (random tensors with the right leg structure) for tests and benchmarks without CTMRG3D
#
# Names of the bonds are the index names of the @tensor expression, e.g. :L5 is `env.C[6][B3 L5 L2]`-`env.E[10][L5 B6; L10 L7]`.


# inside the ClusterExpansions module these are already loaded, they are repeated here so that the file can also be tested standalone
using TensorKit, TensorOperations, Random
using TensorOperations: ManualAllocator


# 1. Define some contraction kernels

# Redefine the current contraction, reading this time from (C, E, A, M) instead of (env, M) (that's handier for slicing)
function contract_onesite_kernel(C, E, A, M; allocator = ManualAllocator())
    return @tensor allocator = allocator (
        C[1][U1 A1 U3] *
        (
            E[1][U2 A2; U1 U4] *
            (
                C[2][U5 A3 U2] *
                (
                    (
                        (E[5][A1 M1; B1 M3] * C[5][B1 L1 L3]) *
                        (
                            A[2][NORTH; A2 M2 B2 M1] *
                            (E[6][A3 M5; B3 M2] * (E[9][L1 B2; L2 L4] * C[6][B3 L5 L2]))
                        )
                    ) *
                    (
                        (
                            (E[2][U10 A6; U5 U7] * C[3][U12 A9 U10]) *
                            (
                                A[1][UP; U9 U7 U4 U6] *
                                (E[4][U3 A4; U8 U6] * (E[3][U11 A8; U12 U9] * C[4][U8 A7 U11]))
                            )
                        ) *
                        (
                            (
                                A[5][WEST; A4 M3 B4 M8] *
                                (E[8][A7 M8; B7 M11] * (E[12][L8 B4; L3 L6] * C[8][B7 L8 L11]))
                            ) *
                            (
                                M[UP NORTH EAST; DOWN SOUTH WEST] *
                                (
                                    A[3][EAST; A6 M10 B6 M5] *
                                    (
                                        A[4][SOUTH; A8 M11 B8 M12] *
                                        (
                                            E[7][A9 M12; B9 M10] *
                                            (
                                                A[6][DOWN; L4 L7 L9 L6] *
                                                (E[11][L12 B8; L11 L9] * (E[10][L5 B6; L10 L7] * C[7][B9 L12 L10]))
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
end

# Alternative bracketing (corresponding to opt=true): found by a joint search over (contraction tree, sliced bonds) with cuts {:U1, :U10}.
# Unsliced it needs ~6x MORE memory than the kernel above (at chi_env = 8, chi_peps = D = 5) and ~1.5x fewer flops; the child order
# inside every product is chosen for the sliced network.  
# ONLY use it together with cuts, otherwise peak memory needed is way too high.
function contract_onesite_kernel_fast(C, E, A, M; allocator = ManualAllocator())
    return @tensor allocator = allocator (
        (
            (
                (
                    (
                        M[UP NORTH EAST; DOWN SOUTH WEST] *
                        (
                            (
                                (
                                    A[6][DOWN; L4 L7 L9 L6] *
                                    (
                                        (
                                            (
                                                (E[11][L12 B8; L11 L9] * C[8][B7 L8 L11]) *
                                                C[7][B9 L12 L10]
                                            ) *
                                            (
                                                (E[8][A7 M8; B7 M11] * C[4][U8 A7 U11]) *
                                                (
                                                    (
                                                        (E[7][A9 M12; B9 M10] * C[3][U12 A9 U10]) *
                                                        A[4][SOUTH; A8 M11 B8 M12]
                                                    ) *
                                                    E[3][U11 A8; U12 U9]
                                                )
                                            )
                                        ) *
                                        (
                                            (
                                                (
                                                    E[5][A1 M1; B1 M3] *
                                                    (E[4][U3 A4; U8 U6] * C[1][U1 A1 U3])
                                                ) *
                                                A[5][WEST; A4 M3 B4 M8]
                                            ) *
                                            (E[12][L8 B4; L3 L6] * C[5][B1 L1 L3])
                                        )
                                    )
                                ) *
                                E[9][L1 B2; L2 L4]
                            ) *
                            (C[6][B3 L5 L2] * E[10][L5 B6; L10 L7])
                        )
                    ) *
                    A[2][NORTH; A2 M2 B2 M1]
                ) *
                A[3][EAST; A6 M10 B6 M5]
            ) *
            (
                (A[1][UP; U9 U7 U4 U6] * E[2][U10 A6; U5 U7]) *
                (E[1][U2 A2; U1 U4] * C[2][U5 A3 U2])
            )
        ) *
        E[6][A3 M5; B3 M2]
    )
end


# 2. slicing machinery: turns “fix bond ℓ to piece j” into ordinary tensor operations

# Given a kind (C,E,A,M) and an index i, it returns the tensor kind[i]
_os_tensor_of(C, E, A, M, kind::Symbol, i::Int) = kind === :C ? C[i] : kind === :E ? E[i] : kind === :A ? A[i] : M

# Returns the index range of chunk j when 1:m is cut into n consecutive, nearly equal chuncks
# e.g. : if m=5, n=2: q=2 and r=1 -> Chunk 1 is 1:3 and chunk 2 is 4:5.
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


# 3: now we add two new allocators (for @tensor allocator = ), which might handle the memory allocation 
# for the repeated contractions of the different slicings better

# ManuelAllocator = default: Every temporary is malloced and freed, but this cost a lot
# ReuseAllocator: keeps freed blocks in pools per size class and hands them out again. Every slice requests the same sequence of sizes, so after the first slice nothing new is allocated.
# TrackingAllocator: wraps another allocator and counts the live and peak bytes of the temporaries.
# This also includes the allocation used for the permutation copies, so result will be higher than our naive model

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
    peak_bytes::Int         # max of above so fat
    nalloc::Int             # number of tempraries
    total_bytes::Int        # total bytes requested
    sizes::Dict{UInt,Int}   # pointer adress -> bytes
end

# Outer constructor
TrackingAllocator(inner = ManualAllocator()) = TrackingAllocator(inner, 0, 0, 0, 0, Dict{UInt,Int}())


"forget the counters (call between cases); `peak_bytes` restarts from the current live size"
function reset_peak!(t::TrackingAllocator)
    t.peak_bytes = t.live_bytes; t.nalloc = 0; t.total_bytes = 0
    return t
end

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


# 4. the sliced contraction

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


# 5. mock environment: random tensors with the leg structure of the 27 tensors (no CTMRG3D needed)

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
