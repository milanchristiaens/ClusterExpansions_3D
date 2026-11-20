# ctmrg3D_helpers.jl -- HELPER file of ctmrg3D_main.jl (the entry point: expectation_value, leading_boundary).
# Here: the CTMRG3D struct and its initialisation, the rotations, the CTMRG steps (ctmrgstep), contract_onesite and transfer_matrix.

# Implementation of the 3C-CTMRG algorithm from Orus 2012 (Phys. Rev. B 85, 205117 – Published 10 May, 2012)
#
# ============================================================================================
#  Conventions
# ============================================================================================
#
# The environment of a single site O is a 3x3x3 cube: O in the centre, surrounded by
# 6 face tensors A, 12 edge tensors E and 8 corner tensors C.
#
# Directions: EAST = +x, NORTH = +y, UP = +z (right-handed), WEST/SOUTH/DOWN the opposites.
#
#   O (0,6):  O[; UP NORTH EAST DOWN SOUTH WEST]
#
# Positions (letters give the direction(s) of the tensor as seen from the centre):
#
#   A1 = U    A2 = N    A3 = E    A4 = S    A5 = W    A6 = D
#
#   upper ring:     E1 = UN   E2 = UE   E3 = US   E4 = UW
#   vertical edges: E5 = NW   E6 = NE   E7 = SE   E8 = SW
#   lower ring:     E9 = DN   E10 = DE  E11 = DS  E12 = DW
#
#   upper corners:  C1 = UNW  C2 = UNE  C3 = USE  C4 = USW
#   lower corners:  C5 = DNW  C6 = DNE  C7 = DSE  C8 = DSW
#
#   Seen from above (N on top), both layers look like
#
#       C1 ─── E1 ─── C2        E5 ─── A2 ─── E6                C5 ─── E9 ─── C6
#       │              │        │              │                │              │
#       E4     A1     E2        A5      O     A3               E12     A6     E10       
#       │              │        │              │                │              │  
#       C4 ─── E3 ─── C3        E8 ─── A4 ──  E7                C8 ─── E11 ── C7
#           upper layer           middle layer                 lower layer
#
# Leg order. Every environment tensor lists its virtual legs counterclockwise as seen from
# outside the cube. The first leg is fixed by the rules below, which are chosen relative to
# the W-E axis, i.e. the direction in which `partial_ctmrgstep` absorbs (it grows the WEST side).
#
#   A (1,4):  A[O; l1 l2 l3 l4], O = the leg contracting with the central tensor.
#             A1, A2, A4, A6 (faces parallel to x): l2 = the edge on the west side,
#                                                   l4 = the edge on the east side.
#             A3, A5 (faces on the x axis):         l1 = the edge on the upper side.
#   E (2,2):  E[c1 f1; c2 f2], c = corner legs, f = face legs.
#             E1, E3, E9, E11 (edges parallel to x): c1 = west corner, c2 = east corner.
#             all other edges (in the W or E plane): f1 = A5 (west) resp. A3 (east).
#   C (0,3):  C[; e1 e2 e3], e1 and e2 are the edges in the corner's own (W or E) plane,
#             e3 is the edge along x.
#
# Explicitly:
#
#   A1 = [O; E1 E4 E3 E2]     E1 = [C1 A1; C2 A2]     E7  = [C7 A3; C3 A4]     C1 = [E5 E4 E1]
#   A2 = [O; E9 E5 E1 E6]     E2 = [C3 A3; C2 A1]     E8  = [C4 A5; C8 A4]     C2 = [E2 E6 E1]
#   A3 = [O; E2 E7 E10 E6]    E3 = [C4 A4; C3 A1]     E9  = [C5 A2; C6 A6]     C3 = [E7 E2 E3]
#   A4 = [O; E3 E8 E11 E7]    E4 = [C1 A5; C4 A1]     E10 = [C6 A3; C7 A6]     C4 = [E4 E8 E3]
#   A5 = [O; E4 E5 E12 E8]    E5 = [C5 A5; C1 A2]     E11 = [C8 A6; C7 A4]     C5 = [E12 E5 E9]
#   A6 = [O; E11 E12 E9 E10]  E6 = [C2 A3; C6 A2]     E12 = [C8 A5; C5 A6]     C6 = [E6 E10 E9]
#                                                                              C7 = [E10 E7 E11]
#                                                                              C8 = [E8 E12 E11]
#
# This convention is covariant under the rotation around the W-E axis (S -> U -> N -> D):
# the tensors on the west side (C4, C1, C5, C8), (E3, E1, E9, E11), (E4, E5, E12, E8),
# (A1, A2, A6, A4) are mapped onto each other with identical leg order, which is why
# `enlarged_corners`, `enlarged_edges`, `renormalize_corners` and `renormalize_edges` can use
# the same index pattern for all four of them.
#
# Spaces (see `CTMRG3D_init`): 
# O ∈ one ← Vp ⊗ Vp ⊗ Vp ⊗ Vp ⊗ Vp ⊗ Vp. 
# A ∈ Vp ← Vbp' ⊗ Vbp' ⊗ Vbp' ⊗ Vbp', 
# E ∈ Vv' ⊗ Vbp' ← Vv ⊗ Vbp,
# C ∈ one ← Vv' ⊗ Vv' ⊗ Vv'. 
# Corner legs may contract with either c1 or c2 and A legs with either
# f1 or f2, so cyclic permutations of legs (see `cycle_legs`) never break the arrow structure.
#
# Rotations. A full CTMRG step (`ctmrgstep`) absorbs from the west, rotating the whole network
# in between with `rot_ud_90`, `rot_ns_90` and `rot_we_90` such that every side is grown.
# A rotation moves each tensor to a new position (permutation p) and, because no leg convention
# can be covariant under all cube rotations (e.g. the 3-fold rotation around a corner diagonal
# cyclically permutes that corner's legs), it also cyclically shifts the legs of the tensor by
# s such that it obeys the convention of its new position:
#
#   new.X[i] = cycle_legs(old.X[p[i]], s[i])      (leg k of the new tensor = leg k + s[i] of the old one)
#
# The shifts were derived from the geometry above. A rotation is an exact symmetry of the
# closed network, so `contract_onesite(env, M)` is invariant under any rotation of `env`
# together with the same permutation of the legs of M.
# ============================================================================================

mutable struct CTMRG3D{T, S, TT <: AbstractTensorMap{T, S, 0, 6}, TA <: AbstractTensorMap{T, S, 1, 4}, TE <: AbstractTensorMap{T, S, 2, 2}, TC <: AbstractTensorMap{T, S, 0, 3}} <: CTMRG3D_alg
    O::TT
    A::Array{TA,1}
    E::Array{TE,1}
    C::Array{TC,1}
    trunc_bpeps::MatrixAlgebraKit.TruncationStrategy
    trunc_env::MatrixAlgebraKit.TruncationStrategy
    miniter::Int
    maxiter::Int
    tol::Float64
    verbosity::Int
    west_updates::Union{Int, Symbol}   # West updates per side: Int >= 1, or :auto (stop once ϵ < west_tol, at most 4)
    west_tol::Float64
    projectors::Symbol                 # :svd = U * sqrt(S) from the SVD of ONE enlarged tensor (original), :oblique = one projector pair per bond (Proposal 1, get_projectors_oblique)
    projector_exponent::Float64        # P of the :svd projectors U * S^P (original: P = 1/2; P = 0 = isometries U); not used by :oblique
    max_seconds::Float64               # time limit of leading_boundary in seconds (Inf = none)

    # env0 = nothing: cold start, the 26 environment tensors depend on init:
    #        init = :random (default) every tensor random (randn), for both projector kinds.    
    #        The random numbers come from a local generator with seed init_seed, so a run is reproducible and the global seed is not touched.
    #        init = :ones   all ones, for both projector kinds. Note: this start is rank one, and :oblique keeps a rank-one
    #                       environment rank one (its projectors are exact on the support of R1 * R2, and the pseudo-inverse drops the rest.
    #        init = :nishino_okunishi   the 3×3×3 cube of O tensors: every corner, edge and face is O itself, with its outward legs closed by one random
    #                       Gaussian vector (seed init_seed, the same vector on every outer leg), see nishino_okunishi_init. All bonds then start at the
    #                       bond dimension D of O and grow to Vv / Vbp (trunc_env / trunc_bpeps) in the first iterations. Needs a cube-symmetric O
    #                       (checked), because every corner, edge and face gets the same tensor.
    # env0 = an earlier CTMRG3D: warm start, we copy its C, E, A and only take the new O
function CTMRG3D(O::TT; trunc_bpeps = notrunc(), trunc_env = notrunc(), Vp = space(O)[1]', Vbp = oneunit(Vp), Vv = oneunit(Vp), miniter::Int = 5, 
                maxiter::Int = 150, tol::Float64 = 1e-8, verbosity::Int = 0 , west_updates::Union{Int, Symbol} = :auto, west_tol::Float64 = 0.01 * tol, 
                projectors::Symbol = :svd, projector_exponent::Real = 0.5, env0 = nothing, init::Symbol = :ones, init_seed::Int = 1, 
                max_seconds::Real = Inf) where {T, S, TT <: AbstractTensorMap{T, S, 0, 6}}        
        init in (:ones, :random, :nishino_okunishi) || throw(ArgumentError("init must be :ones, :random or :nishino_okunishi, got :$init"))
        if env0 === nothing && init === :nishino_okunishi
            A, E, C = nishino_okunishi_init(O; Vp, seed = init_seed)
        elseif env0 === nothing
            A, E, C = CTMRG3D_init(O; Vp, Vbp, Vv)
            init === :random && ((A, E, C) = random_init(A, E, C; seed = init_seed))
        else
            space(env0.O) == space(O) || throw(ArgumentError("warm start: the O of env0 has other spaces than the new O ($(space(env0.O)) against $(space(O)))"))
            # copies of the arrays (not of the tensors): the steps replace tensors in these arrays, env0 must keep its own
            A, E, C = copy(env0.A), copy(env0.E), copy(env0.C)
        end
        (west_updates === :auto || (west_updates isa Int && west_updates ≥ 1)) ||
            throw(ArgumentError("west_updates must be :auto or an integer ≥ 1"))
        projectors in (:svd, :oblique) || throw(ArgumentError("projectors must be :svd or :oblique, got :$projectors"))
        
        max_seconds > 0 || throw(ArgumentError("max_seconds must be > 0, got $max_seconds"))
        new{T, S, TT, typeof(A[1]), typeof(E[1]), typeof(C[1])}(O, A, E, C, trunc_bpeps, trunc_env, miniter, maxiter, tol, 
                                                                verbosity, west_updates, west_tol, projectors, Float64(projector_exponent), Float64(max_seconds))
    end
end

function CTMRG3D_init(O::TensorMap{T,S,0,6}; Vp = space(O)[1]', Vbp = oneunit(Vp), Vv = oneunit(Vp)) where {T,S}
    S_type = scalartype(O)
    
    A = fill(ones(S_type, Vp, Vbp' ⊗ Vbp' ⊗ Vbp' ⊗ Vbp'), 6)
    E = fill(ones(S_type, Vv' ⊗ Vbp', Vv ⊗ Vbp), 12)
    C = ones(S_type, Vv, Vv' ⊗ Vv')
    C = fill(permute(C, ((),(1,2,3))), 8)
    return A, E, C
end

# Random tensors (randn) with the spaces of the tensors of CTMRG3D_init, from a local random number generator (seed), so the global seed is not touched
function random_init(A, E, C; seed = 1)
    rng = Random.Xoshiro(seed)
    make(x) = randn!(rng, similar(x))
    return make.(A), make.(E), make.(C)
end

# The Nishino-Okunishi start: the 26 environment tensors are O itself, with the legs that point out of the 3×3×3 cube closed by the random Gaussian
# vector v (seed; the same v on every outer leg). O[UP, NORTH, EAST, DOWN, SOUTH, WEST] (all legs of dimension D):
#   corner C[UP, NORTH, EAST]          = O with DOWN, SOUTH, WEST closed (the three kept legs point to the three neighbouring edges)
#   edge   E[UP, NORTH; DOWN, EAST]    = O with SOUTH, WEST closed (UP, DOWN along the edge = Vv legs; NORTH, EAST to the two faces = Vbp legs)
#   face   A[centre; UP, NORTH, DOWN, SOUTH] = O with WEST closed and EAST (towards the centre O) as centre leg (the WEST face, as A[5])
# The spaces are those of CTMRG3D_init with Vv = Vbp = the space of the legs of O, and the numbers are copied in as arrays (no symmetry sectors).
# This uses the same tensor for all 8 corners, all 12 edges and all 6 faces, so it is only right for an O that is symmetric under the cube rotations and
# reflections (the CE PEPOs with rot_invariance = true); we check that O is invariant under the permutations of the three axes, and that the legs of
# O have the same dimension, and give an error otherwise.
function nishino_okunishi_init(O::TensorMap{T,S,0,6}; Vp = space(O)[1]', seed = 1) where {T,S}
    # Turn O into array, D is bond dimension of O
    a = convert(Array, O)
    D = size(a, 1)
    all(==(D), size(a)) || throw(ArgumentError("init = :nishino_okunishi needs the same bond dimension on all six legs of O, got $(size(a))"))
    
    # axis permutations (U,N,E,D,S,W): the three axis exchanges, the inversion and the reflection U<->D alone.
    # Together they generate all 48 symmetries of the cube (rotations and reflections)
    for perm in ((2, 1, 3, 5, 4, 6), (3, 2, 1, 6, 5, 4), (1, 3, 2, 4, 6, 5), (4, 5, 6, 1, 2, 3), (4, 2, 3, 1, 5, 6))
        norm(permutedims(a, perm) - a) <= 1e-8 * norm(a) || throw(ArgumentError("init = :nishino_okunishi needs a cube-symmetric O (rot_invariance = true)"))
    end
    
    # Define a random Gaussian boundary vector
    v = randn(Random.Xoshiro(seed), real(T), D)
    # Create the closed bottom-south-west corner of the cube
    @tensor c[u, n, e] := a[u, n, e, d, s, w] * v[d] * v[s] * v[w]
    # Create the closed vertical edge at south-west of the cube
    @tensor ed[u, n, d, e] := a[u, n, e, d, s, w] * v[s] * v[w]
    # Create the closed west face of the cube
    @tensor f[e, u, n, d, s] := a[u, n, e, d, s, w] * v[w]

    # Now we call CTMRG3D_init() to get the right spaces ((non)-dual, (co)-domain, ...) for all spaces having dimension D
    # space(0) = 0 ← V⊗V⊗V⊗V⊗V⊗V, space(O)[1]' = domain(O)[1] = V
    A0, E0, C0 = CTMRG3D_init(O; Vp, Vbp = space(O)[1]', Vv = space(O)[1]')
    # A helper that takes array x and builds a normalized Tensormap with correct spaces and element type from t
    fill_data(t, x) = TensorMap(convert(Array{scalartype(t)}, x ./ norm(x)), space(t))
    # Return the 6 faces, 12 edges, and 8 corners. (using the duality and spaces and such info from A0 and E0 and C0, but the numbers from f, ed and c)
    return [fill_data(A0[1], f) for _ in 1:6], [fill_data(E0[1], ed) for _ in 1:12], [fill_data(C0[1], c) for _ in 1:8]
end

# Change of the corner spectrum between two iterations. While the bond dimension grows (init = :nishino_okunishi) the two spectra have different
# spaces: then we compare the sorted values, the shorter list filled up with zeros.
function spectrum_change(S_new, S_old)
    # If same space:
    space(S_new) == space(S_old) && return norm(S_new - S_old)
    # If they have different spaces:
    # Helper: take diagonal elements of S and sort it from large to small
    values(S) = (x = convert(Array, S); sort([x[i, i] for i in axes(x, 1)]; rev = true))
    a, b = values(S_new), values(S_old)
    n = max(length(a), length(b))
    # Add extra 0's untill both have length of n
    return norm([a; zeros(n - length(a))] - [b; zeros(n - length(b))])
end

envspace(env::CTMRG3D) = domain(env.E[1])[1]

# Cyclically shift the virtual legs of an environment tensor: leg k of the result is leg k + s
# of the input. Since all tensors are ordered counterclockwise as seen from outside the cube,
# a proper rotation can only change the starting leg, so this is all that is needed.
cycle_legs(C::AbstractTensorMap{T,S,0,3}, s) where {T,S} =
    iszero(mod(s, 3)) ? C : permute(C, ((), Tuple(mod1(k + s, 3) for k in 1:3)))
cycle_legs(A::AbstractTensorMap{T,S,1,4}, s) where {T,S} =
    iszero(mod(s, 4)) ? A : permute(A, ((1,), Tuple(1 + mod1(k + s, 4) for k in 1:4)))
function cycle_legs(E::AbstractTensorMap{T,S,2,2}, s) where {T,S}
    iszero(mod(s, 4)) && return E
    @assert mod(s, 4) == 2 "edges can only be shifted by 2 (corner legs are at positions 1 and 3)"
    return permute(E, ((3, 4), (1, 2)))
end

# Move the tensor at position p[i] to position i and shift its legs by s[i] such that it
# satisfies the leg convention of its new position (see top of this file).
function rotl_90!(env::CTMRG3D, p_corners, p_edges, p_as, s_corners, s_edges, s_as)
    env.C[:] = [cycle_legs(env.C[p], s) for (p, s) in zip(p_corners, s_corners)]
    env.E[:] = [cycle_legs(env.E[p], s) for (p, s) in zip(p_edges, s_edges)]
    env.A[:] = [cycle_legs(env.A[p], s) for (p, s) in zip(p_as, s_as)]
end

function rot_ud_90(env::CTMRG3D)
    # negative (clockwise seen from above) rotation along the down -> up axis: W -> N -> E -> S -> W
    p_corners = [4, 1, 2, 3, 8, 5, 6, 7]
    p_edges = [4, 1, 2, 3, 8, 5, 6, 7, 12, 9, 10, 11]
    p_as = [1, 5, 2, 3, 4, 6]
    s_corners = [1, 2, 1, 2, 2, 1, 2, 1]
    s_edges = [2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2]
    s_as = [1, 2, 2, 0, 0, 3]
    rotl_90!(env, p_corners, p_edges, p_as, s_corners, s_edges, s_as)
    env.O = permute(env.O, ((),(1,6,2,4,3,5)))
end

function rot_ns_90(env::CTMRG3D)
    # positive rotation along the south -> north axis: U -> E -> D -> W -> U
    p_corners = [5, 1, 4, 8, 6, 2, 3, 7]
    p_edges = [5, 4, 8, 12, 9, 1, 3, 11, 6, 2, 7, 10]
    p_as = [5, 2, 1, 4, 6, 3]
    s_corners = [2, 1, 2, 1, 1, 2, 1, 2]
    s_edges = [0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 0, 2]
    s_as = [1, 3, 1, 1, 1, 1]
    rotl_90!(env, p_corners, p_edges, p_as, s_corners, s_edges, s_as)
    env.O = permute(env.O, ((),(6,2,1,3,5,4)))
end

function rot_we_90(env::CTMRG3D)
    # positive rotation along the east -> west axis: S -> U -> N -> D -> S.
    # The leg convention is covariant under this rotation, except for the faces on the axis.
    p_corners = [4, 3, 7, 8, 1, 2, 6, 5]
    p_edges = [3, 7, 11, 8, 4, 2, 10, 12, 1, 6, 9, 5]
    p_as = [4, 1, 3, 6, 5, 2]
    s_corners = zeros(Int, 8)
    s_edges = zeros(Int, 12)
    s_as = [0, 0, 1, 0, 3, 0]
    rotl_90!(env, p_corners, p_edges, p_as, s_corners, s_edges, s_as)
    env.O = permute(env.O, ((),(5,1,3,2,4,6)))
end

function enlarged_corners(env::CTMRG3D)
    @tensor opt = true ec1[UP1 UP2 NORTH1 NORTH2 EAST] := 
                        env.C[4][UP1 NORTH1 1] * flip(env.E[3], [2 3 4])[1 NORTH2; EAST UP2]
    @tensor opt = true ec2[UP1 UP2 NORTH1 NORTH2 EAST] := 
                        env.C[1][UP1 NORTH1 1] * flip(env.E[1], [2 3 4])[1 NORTH2; EAST UP2]
    @tensor opt = true ec3[UP1 UP2 NORTH1 NORTH2 EAST] := 
                        env.C[5][UP1 NORTH1 1] * flip(env.E[9], [2 3 4])[1 NORTH2; EAST UP2]
    @tensor opt = true ec4[UP1 UP2 NORTH1 NORTH2 EAST] := 
                        env.C[8][UP1 NORTH1 1] * flip(env.E[11], [2 3 4])[1 NORTH2; EAST UP2]
    
    return [permute(ec, ((),(1,2,3,4,5))) for ec in [ec1 ec2 ec3 ec4]]
end

function enlarged_edges(env::CTMRG3D)
    @tensor opt = true ee1[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 EAST] := 
                        env.E[4][UP1 NORTH1 DOWN1 1] * flip(env.A[1], [1 2 4 5])[NORTH2; UP2 1 DOWN2 EAST]
    @tensor opt = true ee2[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 EAST] := 
                        env.E[5][UP1 NORTH1 DOWN1 1] * flip(env.A[2], [1 2 4 5])[NORTH2; UP2 1 DOWN2 EAST]
    @tensor opt = true ee3[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 EAST] := 
                        env.E[12][UP1 NORTH1 DOWN1 1] * flip(env.A[6], [1 2 4 5])[NORTH2; UP2 1 DOWN2 EAST]
    @tensor opt = true ee4[UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 EAST] := 
                        env.E[8][UP1 NORTH1 DOWN1 1] * flip(env.A[4], [1 2 4 5])[NORTH2; UP2 1 DOWN2 EAST]
    return ee1, ee2, ee3, ee4
end

function enlarged_side(env::CTMRG3D)
    @tensor opt = true es[EAST; UP1 UP2 NORTH1 NORTH2 DOWN1 DOWN2 SOUTH1 SOUTH2] :=
                        env.A[5][1; UP1 NORTH1 DOWN1 SOUTH1] * flip(env.O, [1 2 3 4 5])[UP2 NORTH2 EAST; DOWN2 SOUTH2 1]
    return es
end

# Normally (see Orus) the projectors are isometries U: one side of a new bond gets conj(U), the other side U, so
# the bond carries the projector conj(U) * transpose(U), which is exact without truncation.
# Instead we use U * sqrt(S), this inserts conj(U) * S * transpose(U), biasing every absorption.
# This helps with stability. Up to debate!
function get_projectors_corner(ecs, trunc; P = 0.5)
    Uys = []
    Uzs = []
    for i = 1:4
        ecy = @timeit to "corner projector: permute (y)" permute(ecs[i], ((1,2),(3,4,5)))
        Uy, Sy, Vyᴴ = @timeit to "corner projector: svd_trunc (y)" svd_trunc(ecy; trunc)
        ecz = @timeit to "corner projector: permute (z)" permute(ecs[i], ((3,4),(1,2,5)))
        Uz, Sz, Vzᴴ = @timeit to "corner projector: svd_trunc (z)" svd_trunc(ecz; trunc)

        @timeit to "corner projector: U*sqrt(S)" begin
            # U * S^P (P = 1/2: the original U * sqrt(S))
            push!(Uys, P == 0.5 ? Uy * sqrt(Sy) : Uy * PEPSKit.sdiag_pow(Sy, P))
            push!(Uzs, P == 0.5 ? Uz * sqrt(Sz) : Uz * PEPSKit.sdiag_pow(Sz, P))
        end
    end
    return Uys, Uzs
end

function get_projectors_edge(ees, trunc; P = 0.5)
    Ws = []
    for i = 1:4
        eep = @timeit to "edge projector: permute" permute(ees[i], ((3,4),(1,2,5,6,7)))
        W, S, M = @timeit to "edge projector: svd_trunc" svd_trunc(eep; trunc)

        @timeit to "edge projector: W*sqrt(S)" push!(Ws, P == 0.5 ? W * sqrt(S) : W * PEPSKit.sdiag_pow(S, P))
    end
    return Ws
end

# ============================================================================================
# projectors = :oblique: one projector pair per bond of the new West plane
# ============================================================================================
# The projectors above only look at ONE side of a bond (the SVD of ec or ee alone). Here every bond gets a pair (P1, P2) built from
# the two enlarged tensors on both sides of it, as in Verraes eqs. 3.48-3.51 (and oblique_projector in local_truncation.jl):
#   X (left) = Q1 * R1,  Y (right) = R2 * Q2   (QR / RQ, so that the big X and Y are never multiplied; X * Y is the doubled bond)
#   R1 * R2 ≈ U * S * V'   (truncated SVD),   P1 = R2 * V * S^(-1/2),   P2 = S^(-1/2) * U' * R1
# so that X * P1 * P2 * Y = Q1 * (U S V')_truncated * Q2 ≈ X * Y. Without truncation P1 * P2 is the identity on the bond.
# S^(-1/2) is a pseudo-inverse
# The 12 bonds (i = 1..4, in the order of enlarged_corners / enlarged_edges):
#   y_i:  UP pair of corner i     <->  DOWN pair of edge i        (truncated to χenv)
#   z_i:  NORTH pair of corner i  <->  UP pair of edge i-1        (truncated to χenv; renormalize_edges uses Uzs[i+1] on edge i)
#   f_i:  NORTH pair of edge i    <->  pair i of the face (UP, NORTH, DOWN, SOUTH), only this pair sees O   (truncated to χpeps)
# See Typst file for clearer convention explanation, also NORTH and UP refer to leg names, not to the locations of these enlarged (= pair) tensors
# X is always the edge (bonds y, z) or the face (bonds f), Y the corner (y, z) or the edge (f).
function oblique_pair(R1, R2, trunc)
    U, S, Vᴴ = @timeit to "oblique projector: svd_trunc of R1*R2" svd_trunc(R1 * R2; trunc)
    return @timeit to "oblique projector: build P1 P2" begin
        isqS = PEPSKit.sdiag_pow(S, -1 / 2)
        (R2 * Vᴴ' * isqS, isqS * U' * R1)
    end
end

# R1 of X = Q1 * R1 (bond = the domain of X) and R2 of Y = R2 * Q2 (bond = the codomain of Y)
_R_left(X) = (@timeit to "oblique projector: left_orth (QR)" left_orth(X))[2]
_R_right(Y) = (@timeit to "oblique projector: right_orth (RQ)" right_orth(Y))[1]

# Returns the projectors in the form that renormalize_corners, renormalize_edges and renormalize_side expect.
# renormalize_* expects P[pair; new]: the two legs of the pair in the codomain, the single compressed leg in the domain.
# Those functions apply conj(P)[pair; new] on the corner side and on the NORTH pair of the edges (right), and P[pair; new] on the other sides (left),
# so the corner-side member P2 (new ← pair) is passed as its adjoint (conj(P2')[pair; new] = P2[new; pair]).
function get_projectors_oblique(ecs, ees, es, trunc_env, trunc_bpeps)
    Uys_c, Uzs_c, Uys_e, Uzs_e = [], [], [], []
    for i = 1:4
        # bond y_i: This returns the projectors
        P1, P2 = oblique_pair(
            _R_left( permute(ees[i], ((1,2,3,4,7),(5,6))) ),        # X: regroup the legs such that domain becomes the bond (all rest codomain)
            _R_right( permute(ecs[i], ((1,2),(3,4,5))) ),           # Y: regroup the legs such that codomain becomes the bond (all rest domain)
            trunc_env)
        # We have to take the adjoint of P2: P2'[pair; new] = conj(P2[new; pair])
        push!(Uys_e, P1); push!(Uys_c, copy(P2'))

        # bond z_i:This returns the projectors
        P1, P2 = oblique_pair(
            _R_left( permute(ees[mod1(i-1,4)], ((3,4,5,6,7),(1,2))) ),  # X: regroup the legs such that domain becomes the bond (all rest codomain)
            _R_right( permute(ecs[i], ((3,4),(1,2,5))) ),               # Y: regroup the legs such that codomain becomes the bond (all rest domain)
            trunc_env)
        # We have to take the adjoint of P2: P2'[pair; new] = conj(P2[new; pair])
        # Note that Uzs_e[i] belongs to edge i−1 (z2 = E1-C2), which is why renormalize_edges reads Uzs[mod1(i+1,4)] for edge i.
        push!(Uzs_e, P1); push!(Uzs_c, copy(P2'))
    end

    Ws_e, Ws_f = [], []
    for k = 1:4
        # bond f_k: the face with its pair k (legs 2k, 2k+1 of es) as the domain, edge k with its NORTH pair as the codomain
        pair = (2k, 2k+1)
        P1, P2 = oblique_pair(
            _R_left( permute(es, (Tuple(setdiff(1:9, pair)), pair)) ),  # X: face with pair k as its domain, all rest codomain
            _R_right( permute(ees[k], ((3,4),(1,2,5,6,7))) ),           # Y: edge with bond in codomain, rest in domain
            trunc_bpeps)
        # We have to take the adjoint of P2: P2'[pair; new] = conj(P2[new; pair])
        push!(Ws_f, P1); push!(Ws_e, copy(P2'))
    end
    return Uys_c, Uzs_c, Uys_e, Uzs_e, Ws_e, Ws_f
end


function partial_ctmrgstep(env::CTMRG3D)
    ecs = @timeit to "enlarged_corners (4 contractions)" enlarged_corners(env)
    ees = @timeit to "enlarged_edges (4 contractions)" enlarged_edges(env)
    es = @timeit to "enlarged_side (1 contraction)" enlarged_side(env)

    if env.projectors === :svd
        # svd: one projector per bond, the same one on both sides
        Uys, Uzs = @timeit to "get_projectors_corner (8 SVDs)" get_projectors_corner(ecs, env.trunc_env; P = env.projector_exponent)
        Ws = @timeit to "get_projectors_edge (4 SVDs)" get_projectors_edge(ees, env.trunc_bpeps; P = env.projector_exponent)

        @timeit to "renormalize_corners" renormalize_corners(env, ecs, Uys, Uzs)
        @timeit to "renormalize_edges" renormalize_edges(env, ees, Uys, Uzs, Ws)
        @timeit to "renormalize_side" renormalize_side(env, es, Ws)
    else
        # oblique: one projector pair per bond, each side of the bond gets its own member
        Uys_c, Uzs_c, Uys_e, Uzs_e, Ws_e, Ws_f = @timeit to "get_projectors_oblique (12 bonds)" get_projectors_oblique(ecs, ees, es, env.trunc_env, env.trunc_bpeps)

        @timeit to "renormalize_corners" renormalize_corners(env, ecs, Uys_c, Uzs_c)
        @timeit to "renormalize_edges" renormalize_edges(env, ees, Uys_e, Uzs_e, Ws_e)
        @timeit to "renormalize_side" renormalize_side(env, es, Ws_f)
    end

    @timeit to "normalize! (all 26 tensors)" normalize!(env)

    return
end


# Corner spectrum used as convergence quantity: singular values of C[1] in the CURRENT frame
corner_spectrum(env::CTMRG3D) = svd_trunc(permute(env.C[1], ((1,2),(3,))))[2]

# Variable number of west updates:
function ctmrg_4steps(env::CTMRG3D; side::String = "")
    env.verbosity > 1 && @info "CTMRG3D side $side"

    # First check how many west updates we want
    auto = env.west_updates === :auto
    nmax = auto ? 4 : env.west_updates

    # Calculate current spectrum of C[1], if auto or if vebosity > 2
    track = auto || env.verbosity > 2
    S_prev = nothing
    if track   # baseline at the start of this side, in the current frame
        S_prev = @timeit to "convergence quantity: svd_trunc of C[1]" corner_spectrum(env)
    end

    # Perform the West updates
    for i = 1:nmax
        @timeit to "partial_ctmrgstep" partial_ctmrgstep(env)

        # If we track the spectrum (vebosity > 2 or auto)
        if track
            S = @timeit to "convergence quantity: svd_trunc of C[1]" corner_spectrum(env)
            ϵ = spectrum_change(S, S_prev)
            S_prev = S
            env.verbosity > 2 && @info "CTMRG3D side $side, West update $i/$nmax: ϵ = $ϵ"
            
            # If auto, and the spectrum's change is smaller than the tolerance, then we stop the updates
            if auto && ϵ < env.west_tol
                env.verbosity > 2 && @info "CTMRG3D side $side: stopped after $i West update(s)"
                break
            end
        end
    end
end

# One full iteration: all six sides are updated (nmax West updates each).
function ctmrgstep(env::CTMRG3D)
    for side in ("W", "S", "E", "N")
        ctmrg_4steps(env; side)
        @timeit to "rot_ud_90" rot_ud_90(env)
    end
    @timeit to "rot_ns_90" rot_ns_90(env)
    ctmrg_4steps(env; side = "D")
    @timeit to "rot_ns_90" rot_ns_90(env)
    @timeit to "rot_ns_90" rot_ns_90(env)
    ctmrg_4steps(env; side = "U")
    @timeit to "rot_ns_90" rot_ns_90(env)
    S = @timeit to "convergence quantity: svd_trunc of C[1]" corner_spectrum(env)
    return S
end


function renormalize_corners(env::CTMRG3D, ecs, Uys, Uzs)
    corners = [4, 1, 5, 8]
    for i = 1:4
        C′ = @timeit to "renormalize_corners: contraction" begin
            @tensor opt = true C′[u n e] :=
                                ecs[i][UP1 UP2 NORTH1 NORTH2 e] *
                                conj(Uys[i][UP1 UP2; u]) * conj(Uzs[i][NORTH1 NORTH2; n])
            C′
        end
        env.C[corners[i]] = @timeit to "renormalize_corners: permute" permute(C′, ((),(1,2,3)))
    end
end

function renormalize_edges(env::CTMRG3D, ees, Uys, Uzs, Ws)
    edges = [4, 5, 12, 8]
    for i = 1:4
        E′ = @timeit to "renormalize_edges: contraction" begin
            @tensor opt = true E′[u n; d e] :=
                                ees[i][UP1 UP2 NORTH1 NORTH2; DOWN1 DOWN2 e] *
                                Uzs[mod1(i+1,4)][UP1 UP2; u] * Uys[i][DOWN1 DOWN2; d] *
                                conj(Ws[i][NORTH1 NORTH2; n])
            E′
        end
        env.E[edges[i]] = @timeit to "renormalize_edges: flip" flip(E′, 2)
    end
end

function renormalize_side(env::CTMRG3D, es, Ws)
    A′ = @timeit to "renormalize_side: contraction" begin
        @tensor opt = true A′[e; u n d s] :=
                            es[e; UP1 UP2 NORTH1 NORTH2 DOWN1 DOWN2 SOUTH1 SOUTH2] *
                            Ws[1][UP1 UP2; u] * Ws[2][NORTH1 NORTH2; n] *
                            Ws[3][DOWN1 DOWN2; d] * Ws[4][SOUTH1 SOUTH2; s]
        A′
    end
    env.A[5] = @timeit to "renormalize_side: flip" flip(A′, [2 3 4 5])
end

function normalize!(env::CTMRG3D)
    for i = 1:8
        env.C[i] /= norm(env.C[i])
    end
    for i = 1:12
        env.E[i] /= norm(env.E[i])
    end
    for i = 1:6
        env.A[i] /= norm(env.A[i])
    end
end

# Memory-minimal contraction order for the closed cube network of snippet.txt.
#
# At χ = 8, D = 5 (corner legs χ, all other legs D):
#
#                     flops      largest intermediate   live intermediates (peak)
#   opt = true        1.52e12    119.2 GiB (χ⁴D⁹)       238.4 GiB
#   this order        2.27e12     23.8 GiB (χ⁴D⁸)        38.8 GiB
#
# Contraction order is defined in TREE_CURRENT in contract_onesite_slicing.jl

# This function below is the most costly function we have, two times (numerator/denomantaror) run 
# per observation we want to perform
function contract_onesite(env::CTMRG3D, M; allocator = ManualAllocator())
    return contract_onesite_kernel(env.C, env.E, env.A, M; allocator)
end

function transfer_matrix(env::CTMRG3D)
    @tensor opt = true TM[A1 A3 A9 A7; B1 B3 B9 B7] := 
                            flip(env.E[1], 1)[A7 1; B7 4] * flip(env.E[3], [1 2 4])[A1 2; B1 1] * 
                            flip(env.E[11], 1)[A3 3; B3 2] * flip(env.E[9], [1 2 4])[A9 4; B9 3]
    return TM
end

function calculate_universal_3D_spectrum(env::CTMRG3D)
    @tensor opt=true mat_e[M1a M4a; M1b M4b] := env.C[8][M4b B4 B3] * flip(env.C[5], [1 2 3])[B4 M1b B1] * env.C[6][M2 B2 B1] * flip(env.C[7], [1 2 3])[B2 M3 B3] * 
                                flip(env.C[4], [1 2 3])[A4 M4a A3] * env.C[1][M1a A4 A1] * flip(env.C[2], [1 2 3])[A2 M2 A1] * env.C[3][M3 A2 A3]

    @tensor opt=true mat_c[M1a; M1b] := mat_e[M1a M4; M1b M4]
    _, S_e, V = svd_full(permute(mat_e, ((1,3),(2,4))))
    S_e /= tr(S_e)
    _, S_c, V = svd_full(mat_c)
    S_c /= tr(S_c)

    EE_e = -tr(S_e * log(S_e))
    EE_c = -tr(S_c * log(S_c))
    return EE_e - EE_c
end