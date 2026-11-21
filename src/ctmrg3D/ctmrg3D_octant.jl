# ctmrg3D_octant.jl -- Proposal 2: cube-symmetric CTMRG3D with octant projectors (CTMRG3DOctant).
#
# Use: expectation_value(PEPO, obss, bpepsspace, envspace; ctm_alg = CTMRG3DOctant, ...) (same entry point as CTMRG3D), or
#      CTMRG3DOctant(O; Vv, Vbp, ...) followed by leading_boundary(env) (the leading_boundary of ctmrg3D_main.jl).
# Needs an O that is symmetric under the cube rotations and reflections (CE PEPOs with rot_invariance = true; checked), real (Float64),
# without symmetry sectors (we work with plain arrays). Not for fermions.
#
# ============================================================================================
#  Idea
# ============================================================================================
# The environment of the central site after n steps is a (2n+1)³ cube cut into 8 corner blocks (n³ sites), 12 edge blocks (n² sites)
# and 6 face blocks (n sites). With cube symmetry all corners are the same tensor C, all edges E and all faces A, so we store only one of each.
# One step grows ALL six sides at once (as the C4v CTMRG in 2D), because a one-sided (West) update leaves a corner with legs in two different
# bases, which a single symmetric C cannot have. Counting sites, (n+1)³ = n³ + 3n² + 3n + 1, so
#     C' = C·E³·A³·O,     E' = E·A²·O,     A' = A·O
# and the new legs are compressed with two projectors:
#     P_c: F = χenv·χpeps²·D  -> χenv   on every corner-edge bond (and on both long legs of every edge)
#     P_f: χpeps·D           -> χpeps  on every edge-face bond (and on the four legs of every face)
# The same orthogonal P on both sides of a bond inserts P·Pᵀ, a projector: without truncation a step is exact.
#
# P_c: the top χenv eigenvectors of ρ_F = Q·Qᵀ, with Q the octant C·E³·A³·O as a matrix (one leg F) × (two other legs F²).
#      Q (F³ numbers) is never stored: we only apply ρ_F to a block of vectors in a block subspace iteration with Rayleigh-Ritz, started from the
#      P_c of the previous step. The part B = C·E_y·E_z·A_E of Q does not touch the vectors: we contract it once per step and form M = B·Bᵀ over its
#      open legs, so that ρ_F·V = X·M·Xᵀ·V with X = E_x·A_U·A_N·O (about 10× fewer flops than Q·(Qᵀ·V) at (16,5,3)).
# P_f: the top χpeps eigenvectors of the reduced matrix of the (face, O) part of a leg F, taken from the truncated ρ_F ≈ U·Λ·Uᵀ
#      (an approximation: the discarded weight of ρ_F is ignored), averaged over the two faces in F.
# C': from Qᵀ·P_c (one product with the χenv columns of P_c per step), then P_c on the two other legs.
#
# ============================================================================================
#  Conventions (plain arrays; directions as in ctmrg3D_helpers.jl: EAST = +x, NORTH = +y, UP = +z)
# ============================================================================================
#   O[u, n, e, d, s, w]       the O of expectation_value (legs UP NORTH EAST DOWN SOUTH WEST)
#   c[x, y, z]                the corner at (+,+,+); leg x points to -x (to the x-edge at (0,+,+)), y to -y, z to -z
#   e[c-, c+, fU, fN]         the x-edge at (0,+,+): c- to the corner at -x, c+ to the corner at +x, fU to the up face (0,0,+) (direction -y),
#                             fN to the north face (0,+,0) (direction -z)
#   a[o, +x, +y, -x, -y]      the up face at (0,0,+): o to the centre (direction -z), then the legs to +x, +y, -x, -y (counterclockwise from above)
# All other corners, edges and faces are proper rotations of these; we use the rotation R: x -> y -> z -> x to place them. In the octant of
# a site s (the new corner = the old blocks in the (+,+,+) directions of s, plus s itself):
#   E_x = e[ox, cx, ux, nx], E_y = e[oy, cy, ey, uy], E_z = e[oz, cz, nz, ez]          (the edges at (0,+,+), (+,0,+), (+,+,0) of s)
#   A_U = a[oU, uy, ux, FxU, FyU], A_N = a[oN, nx, nz, FzN, FxN], A_E = a[oE, ez, ey, FyE, FzE]
#   O[oU, oN, oE, FzO, FyO, FxO]
# and the three legs of the octant are the composites
#   F_x = (ox, FxU, FxN, FxO),  F_y = (oy, FyE, FyU, FyO),  F_z = (oz, FzN, FzE, FzO)      (first index fastest in the reshape)
# The new edge at (0,+,+) of a site t uses the old E = e[o-, o+, u, n], A_U = a[oU, xp, u, xm, ym], A_N = a[oN, n, xp2, zm, xm2] and O of t:
#   c-' = (o-, xm2, xm, Ow),  c+' = (o+, xp, xp2, Oe),  fU' = (ym, Os),  fN' = (zm, Od)
# (c-' lists the north face before the up face: that is the order of the +x leg of the corner at (-,+,+), the rotation of the corner at
# (+,+,+) about the z axis; c+' has the order of F_x. The 2-fold rotation of the edge maps c+' onto c-' with this order.)
# The new face: a' = (a·O)[d; (+x, e), (+y, n), (-x, w), (-y, s)], each pair (face leg, O leg) compressed with P_f.
#
# For the observables we copy c, e, a into the 27 tensors of a CTMRG3D in Sander's conventions (expand_to_ctmrg3d) and use contract_onesite
# (or the sliced version through contract_onesite_dispatch), so the plans of onesite_plans.csv apply unchanged:
#   C1 = [E5 E4 E1] -> permutedims(c, (2,3,1)) (corner (+,+,+) is C2 = [E2 E6 E1] = c[y z x]),  E1 = [C1 A1; C2 A2] -> permutedims(e, (1,3,2,4)),
#   A1 = [O; E1 E4 E3 E2] -> permutedims(a, (1,3,4,5,2)),  the same tensor at every position (each is invariant under its own rotations).

using LinearAlgebra: I, Diagonal, Symmetric, diagm, eigen, qr, svdvals

mutable struct CTMRG3DOctant{T<:Real, TT} <: CTMRG3D_alg
    O::TT                              # the TensorMap O (used for the spaces of the 27-tensor copy)
    o::Array{T,6}                      # O as an array
    c::Array{T,3}
    e::Array{T,4}
    a::Array{T,5}
    P::Union{Nothing, Matrix{T}}       # P_c of the last step (F × χenv), start vectors of the next subspace iteration
    chi_env::Int
    chi_peps::Int
    miniter::Int
    maxiter::Int
    tol::Float64
    verbosity::Int
    max_seconds::Float64
    subspace_tol::Float64              # stop the subspace iteration when the residuals of the kept Ritz vectors are below subspace_tol · λ_max
                                       # (default min(1e-6, 10·tol): the CTMRG3D change ϵ then still reaches tol; at 1e-6 the floor of ϵ was ≈ 4e-8,
                                       # and m changed by at most 5e-6 against 1e-10 (sandbox, (8,3,3), 21 Nov), with 2-3× fewer products ρ_F·V)
    subspace_maxiter::Int              # at most this many products ρ_F·V per step
    oversample::Int                    # block size = χenv + oversample
end

# Same keywords as CTMRG3D (so that expectation_value(...; ctm_alg = CTMRG3DOctant) works); projectors, projector_exponent, class_projectors,
# west_updates and west_tol have no meaning here and are ignored.
#   init: the start is the 3×3×3 cube of O tensors with the outer legs closed by a vector v (as init = :nishino_okunishi of CTMRG3D):
#         :random or :nishino_okunishi -> v Gaussian (seed init_seed), :ones -> v = ones. The bonds start at D and grow to χenv, χpeps.
#   env0: an earlier CTMRG3DOctant (warm start, e.g. the previous β): we copy c, e, a, P.
function CTMRG3DOctant(O::AbstractTensorMap{T,S,0,6}; Vv = oneunit(space(O)[1]), Vbp = oneunit(space(O)[1]), trunc_env = nothing, trunc_bpeps = nothing,
                       miniter::Int = 5, maxiter::Int = 150, tol::Float64 = 1e-8, verbosity::Int = 0, env0 = nothing, init::Symbol = :random,
                       init_seed::Int = 1, max_seconds::Real = Inf, subspace_tol::Real = min(1e-6, 10 * tol), subspace_maxiter::Int = 30, oversample::Int = 8,
                       kwargs...) where {T,S}
    T <: Real || throw(ArgumentError("CTMRG3DOctant: O must be real (Float64), got $T"))
    sectortype(O) == Trivial || throw(ArgumentError("CTMRG3DOctant: O must have no symmetry sectors"))
    init in (:ones, :random, :nishino_okunishi) || throw(ArgumentError("init must be :ones, :random or :nishino_okunishi, got :$init"))
    o = convert(Array, O)
    _check_cube_symmetric(o)
    chi_env, chi_peps = dim(Vv), dim(Vbp)        # (expectation_value passes trunc_env = truncrank(dim(Vv)) as well: the same numbers)
    if env0 === nothing
        D = size(o, 1)
        v = init === :ones ? ones(T, D) : randn(Random.Xoshiro(init_seed), T, D)
        @tensor c[w, s, d] := o[u, n, e, d, s, w] * v[u] * v[n] * v[e]
        @tensor ed[w, e, s, d] := o[u, n, e, d, s, w] * v[u] * v[n]
        @tensor a[d, e, n, w, s] := o[u, n, e, d, s, w] * v[u]
        c, ed, a, P = c / norm(c), ed / norm(ed), a / norm(a), nothing
    else
        env0 isa CTMRG3DOctant || throw(ArgumentError("warm start of CTMRG3DOctant needs a CTMRG3DOctant as env0"))
        size(env0.o) == size(o) || throw(ArgumentError("warm start: the O of env0 has other dimensions than the new O"))
        c, ed, a, P = copy(env0.c), copy(env0.e), copy(env0.a), env0.P === nothing ? nothing : copy(env0.P)
    end
    max_seconds > 0 || throw(ArgumentError("max_seconds must be > 0, got $max_seconds"))
    return CTMRG3DOctant{T, typeof(O)}(O, o, c, ed, a, P, chi_env, chi_peps, miniter, maxiter, tol, verbosity, Float64(max_seconds),
                                       Float64(subspace_tol), subspace_maxiter, oversample)
end

function _check_cube_symmetric(o)
    D = size(o, 1)
    all(==(D), size(o)) || throw(ArgumentError("CTMRG3DOctant needs the same bond dimension on all six legs of O, got $(size(o))"))
    for perm in ((2, 1, 3, 5, 4, 6), (3, 2, 1, 6, 5, 4), (1, 3, 2, 4, 6, 5), (4, 5, 6, 1, 2, 3))
        norm(permutedims(o, perm) - o) <= 1e-8 * norm(o) ||
            throw(ArgumentError("CTMRG3DOctant needs an O that is symmetric under the cube rotations and reflections (rot_invariance = true)"))
    end
end

envspace(env::CTMRG3DOctant) = ℂ^size(env.c, 1)

# singular values of c as a matrix (x) × (y z), c normalized: the convergence quantity (as corner_spectrum of CTMRG3D)
function corner_spectrum(env::CTMRG3DOctant)
    χ = size(env.c, 1)
    s = svdvals(reshape(env.c, χ, :))
    return TensorMap(diagm(s), ℂ^χ ← ℂ^χ)
end

# ---------------------------------------------------------------------------------------------
# The octant as a linear map: Qᵀ·V (V on F_x) and Q·W (W on F_y, F_z)
# ---------------------------------------------------------------------------------------------
# B = C·E_y·E_z·A_E [cx, oy, uy, oz, nz, oE, FyE, FzE]: the part of the octant that does not touch F_x (once per step)
function _octant_B(c, e, a)
    @tensor t1[cx, cz, oy, ey, uy] := c[cx, cy, cz] * e[oy, cy, ey, uy]
    @tensor t2[cx, oy, ey, uy, oz, nz, ez] := t1[cx, cz, oy, ey, uy] * e[oz, cz, nz, ez]
    @tensor B[cx, oy, uy, oz, nz, oE, FyE, FzE] := t2[cx, oy, ey, uy, oz, nz, ez] * a[oE, ez, ey, FyE, FzE]
    return B
end

# The octant split in two parts: X = E_x·A_U·A_N·O (touches F_x) and B = C·E_y·E_z·A_E. They share the internal legs a = (cx, uy, nz, oE)
# (dimension χ·χpeps²·D); the open legs of F_y and F_z are split into Y' = (FyU, FzN, FzO, FyO) (on X) and Y'' = (oy, oz, FyE, FzE) (on B).
# Then ρ_F = Q·Qᵀ = X·M·Xᵀ with M[a, a'] = Σ_{Y''} B[a, Y''] B[a', Y''] (once per step), so a product ρ_F·V never forms the F²-sized Qᵀ·V:
#   _octant_xfwd(V) = Xᵀ·V [a, Y', b] -> M · that -> _octant_xbwd(.) = X·(.) [F_x, b]

# Xᵀ·V: V is F × b (F_x composite); the result X4[cx, β, uy, FyU, nz, FzN, oE, FzO, FyO]
function _octant_xfwd(V, e, a, o)
    χ, p, D, b = size(e, 1), size(e, 3), size(o, 1), size(V, 2)
    V5 = reshape(V, χ, p, p, D, b)
    @tensor X1[cx, ux, nx, FxU, FxN, FxO, β] := e[ox, cx, ux, nx] * V5[ox, FxU, FxN, FxO, β]
    @tensor X2[cx, nx, FxN, FxO, β, oU, uy, FyU] := X1[cx, ux, nx, FxU, FxN, FxO, β] * a[oU, uy, ux, FxU, FyU]
    @tensor X3[cx, FxO, β, oU, uy, FyU, oN, nz, FzN] := X2[cx, nx, FxN, FxO, β, oU, uy, FyU] * a[oN, nx, nz, FzN, FxN]
    @tensor X4[cx, β, uy, FyU, nz, FzN, oE, FzO, FyO] := X3[cx, FxO, β, oU, uy, FyU, oN, nz, FzN] * o[oU, oN, oE, FzO, FyO, FxO]
    return X4
end

# X·Y: Y1[β, FyU, FyO, FzN, FzO, cx, uy, nz, oE]; the result is F × b (F_x composite)
function _octant_xbwd(Y1, e, a, o)
    χ, p, D, b = size(e, 1), size(e, 3), size(o, 1), size(Y1, 1)
    @tensor Y2[β, FyU, FzN, cx, uy, nz, oU, oN, FxO] := Y1[β, FyU, FyO, FzN, FzO, cx, uy, nz, oE] * o[oU, oN, oE, FzO, FyO, FxO]
    @tensor Y3[β, FzN, cx, nz, oN, FxO, ux, FxU] := Y2[β, FyU, FzN, cx, uy, nz, oU, oN, FxO] * a[oU, uy, ux, FxU, FyU]
    @tensor Y4[β, cx, FxO, ux, FxU, nx, FxN] := Y3[β, FzN, cx, nz, oN, FxO, ux, FxU] * a[oN, nx, nz, FzN, FxN]
    @tensor R[ox, FxU, FxN, FxO, β] := Y4[β, cx, FxO, ux, FxU, nx, FxN] * e[ox, cx, ux, nx]
    return reshape(R, χ * p * p * D, b)
end

# M[(cx, uy, nz, oE), (cx', uy', nz', oE')] = Σ_{Y''} B B (a matrix of size (χ·χpeps²·D)²)
function _octant_M(B)
    χ, p, D = size(B, 1), size(B, 3), size(B, 6)
    Bm = reshape(permutedims(B, (1, 3, 5, 6, 2, 4, 7, 8)), χ * p * p * D, :)
    return Bm * transpose(Bm)
end

# ρ_F·V = X·M·Xᵀ·V
function _octant_rhoV(V, M, e, a, o)
    χ, p, D, b = size(e, 1), size(e, 3), size(o, 1), size(V, 2)
    X4 = _octant_xfwd(V, e, a, o)                                                   # [cx, β, uy, FyU, nz, FzN, oE, FzO, FyO]
    Xa = reshape(permutedims(X4, (1, 3, 5, 7, 2, 4, 9, 6, 8)), χ * p * p * D, :)      # (cx, uy, nz, oE) × (β, FyU, FyO, FzN, FzO)
    Z = reshape(M * Xa, χ, p, p, D, b, p, D, p, D)                                  # [cx, uy, nz, oE, β, FyU, FyO, FzN, FzO]
    return _octant_xbwd(permutedims(Z, (5, 6, 7, 8, 9, 1, 2, 3, 4)), e, a, o)
end

# Qᵀ·V: V is F × b (F_x composite), the result W is F² × b (F_y composite, then F_z composite); only for C' (once per step, b = χenv)
function _octant_QtV(V, B, e, a, o)
    χ, p, D, b = size(e, 1), size(e, 3), size(o, 1), size(V, 2)
    X4 = _octant_xfwd(V, e, a, o)
    @tensor W[oy, FyE, FyU, FyO, oz, FzN, FzE, FzO, β] := X4[cx, β, uy, FyU, nz, FzN, oE, FzO, FyO] * B[cx, oy, uy, oz, nz, oE, FyE, FzE]
    F = χ * p * p * D
    return reshape(W, F * F, b)
end

# Q·W: W is F² × b, the result is F × b (used by the checks)
function _octant_QW(W, B, e, a, o)
    χ, p, D, b = size(e, 1), size(e, 3), size(o, 1), size(W, 2)
    W9 = reshape(W, χ, p, p, D, χ, p, p, D, b)
    @tensor Y1[β, FyU, FyO, FzN, FzO, cx, uy, nz, oE] := W9[oy, FyE, FyU, FyO, oz, FzN, FzE, FzO, β] * B[cx, oy, uy, oz, nz, oE, FyE, FzE]
    return _octant_xbwd(Y1, e, a, o)
end

# Top nkeep eigenvectors of ρ_F = Q·Qᵀ by block subspace iteration with Rayleigh-Ritz.
# Returns (U, λ, WU): U (F × nkeep) orthonormal, λ the eigenvalues (descending), WU = Qᵀ·U (F² × nkeep).
function _octant_top_eigvecs(env::CTMRG3DOctant, B, nkeep)
    e, a, o = env.e, env.a, env.o
    F = size(e, 1) * size(e, 3)^2 * size(o, 1)
    b = min(F, nkeep + env.oversample)
    T = eltype(e)
    M = @timeit to "octant: M = B·Bᵀ" _octant_M(B)
    if b == F
        V = Matrix{T}(I, F, F)                         # small F: the full space, one product gives ρ_F exactly
    else
        V = randn(Random.Xoshiro(1), T, F, b)
        if env.P !== nothing && size(env.P, 1) == F    # warm start: the P_c of the previous step in the first columns
            k = min(size(env.P, 2), b)
            V[:, 1:k] = env.P[:, 1:k]
        end
        V = Matrix(qr(V).Q)
    end
    local λ, S
    for it in 1:env.subspace_maxiter
        Y = @timeit to "octant: ρ_F·V" _octant_rhoV(V, M, e, a, o)
        H = Symmetric(V' * Y)
        ev = eigen(H)
        order = sortperm(ev.values; rev = true)
        λ, S = ev.values[order], ev.vectors[:, order]
        b == F && break
        R = Y * S[:, 1:nkeep] - V * S[:, 1:nkeep] * Diagonal(λ[1:nkeep])
        res = maximum(norm(R[:, k]) for k in 1:nkeep) / max(abs(λ[1]), eps(T))
        env.verbosity > 2 && @info "octant subspace iteration $it: residual $res"
        res < env.subspace_tol && break
        V = Matrix(qr(Y * S).Q)                        # one power step (Y·S spans ρ_F·V)
    end
    U = V * S[:, 1:nkeep]
    WU = @timeit to "octant: Qᵀ·P_c (for C')" _octant_QtV(U, B, e, a, o)
    return U, λ[1:nkeep], WU
end

# P_f from the truncated ρ_F ≈ U·Λ·Uᵀ: reduced matrix of the (face, O) part of F, averaged over the two faces in F; top nkeep eigenvectors
function _octant_face_projector(U, λ, χ, p, D, nkeep)
    U5 = reshape(U, χ, p, p, D, size(U, 2))
    Ul = U5 .* reshape(max.(λ, 0), 1, 1, 1, 1, :)
    @tensor ρ1[f, d, f2, d2] := U5[x, f, g, d, k] * Ul[x, f2, g, d2, k]
    @tensor ρ2[f, d, f2, d2] := U5[x, g, f, d, k] * Ul[x, g, f2, d2, k]
    ρf = Symmetric(reshape(ρ1 + ρ2, p * D, p * D) / 2)
    ev = eigen(ρf)
    order = sortperm(ev.values; rev = true)
    return ev.vectors[:, order[1:nkeep]]
end

# New c, e, a from P_c (F × χ'), P_f (p·D × p') and WP = Qᵀ·P_c (F² × χ')
function _octant_renormalize(env::CTMRG3DOctant, Pc, Pf, WP)
    e, a, o = env.e, env.a, env.o
    χ, p, D = size(e, 1), size(e, 3), size(o, 1)
    F = χ * p * p * D
    χn, pn = size(Pc, 2), size(Pf, 2)
    W3 = reshape(WP, F, F, χn)
    @tensor c′[i, j, k] := W3[y, z, i] * Pc[y, j] * Pc[z, k]
    P5 = reshape(Pc, χ, p, p, D, χn)
    Pf3 = reshape(Pf, p, D, pn)
    @tensor AUf[oU, xp, u, xm, Os, k] := a[oU, xp, u, xm, ym] * Pf3[ym, Os, k]
    @tensor ANf[oN, n, xp2, xm2, Od, l] := a[oN, n, xp2, zm, xm2] * Pf3[zm, Od, l]
    @tensor T1[xp, u, xm, k, oN, Oe, Od, Ow] := AUf[oU, xp, u, xm, Os, k] * o[oU, oN, Oe, Od, Os, Ow]
    @tensor T2[xp, u, xm, k, Oe, Ow, n, xp2, xm2, l] := T1[xp, u, xm, k, oN, Oe, Od, Ow] * ANf[oN, n, xp2, xm2, Od, l]
    @tensor EP[op, u, n, xm2, xm, Ow, i] := e[om, op, u, n] * P5[om, xm2, xm, Ow, i]
    @tensor T3[xp, k, Oe, xp2, l, op, i] := T2[xp, u, xm, k, Oe, Ow, n, xp2, xm2, l] * EP[op, u, n, xm2, xm, Ow, i]
    @tensor e′[i, j, k, l] := T3[xp, k, Oe, xp2, l, op, i] * P5[op, xp, xp2, Oe, j]
    @tensor AO[p1, p2, p3, p4, n, ee, d, s, w] := a[oA, p1, p2, p3, p4] * o[oA, n, ee, d, s, w]
    @tensor a′[d, k1, k2, k3, k4] := AO[p1, p2, p3, p4, n, ee, d, s, w] * Pf3[p1, ee, k1] * Pf3[p2, n, k2] * Pf3[p3, w, k3] * Pf3[p4, s, k4]
    env.c = c′ / norm(c′)
    env.e = e′ / norm(e′)
    env.a = a′ / norm(a′)
    env.P = Pc
    return env
end

# One step: all six sides grow by one layer. Returns the corner spectrum (convergence quantity of leading_boundary).
function ctmrgstep(env::CTMRG3DOctant)
    χ, p, D = size(env.e, 1), size(env.e, 3), size(env.o, 1)
    F = χ * p * p * D
    χn = min(env.chi_env, F)
    pn = min(env.chi_peps, p * D)
    B = @timeit to "octant: B = C·E·E·A" _octant_B(env.c, env.e, env.a)
    U, λ, WU = @timeit to "octant: P_c (subspace iteration)" _octant_top_eigvecs(env, B, χn)
    Pf = @timeit to "octant: P_f" _octant_face_projector(U, λ, χ, p, D, pn)
    @timeit to "octant: new C, E, A" _octant_renormalize(env, U, Pf, WU)
    return corner_spectrum(env)
end

# ---------------------------------------------------------------------------------------------
# Observables: copy into the 27 tensors of a CTMRG3D and use its contraction
# ---------------------------------------------------------------------------------------------
function expand_to_ctmrg3d(env::CTMRG3DOctant)
    χ, p = size(env.e, 1), size(env.e, 3)
    env27 = CTMRG3D(env.O; Vv = ℂ^χ, Vbp = ℂ^p, init = :ones)
    Ct = TensorMap(permutedims(env.c, (2, 3, 1)), space(env27.C[1]))
    Et = TensorMap(permutedims(env.e, (1, 3, 2, 4)), space(env27.E[1]))
    At = TensorMap(permutedims(env.a, (1, 3, 4, 5, 2)), space(env27.A[1]))
    env27.C[:] = fill(Ct, 8)
    env27.E[:] = fill(Et, 12)
    env27.A[:] = fill(At, 6)
    return env27
end

contract_onesite(env::CTMRG3DOctant, M; kwargs...) = contract_onesite(expand_to_ctmrg3d(env), M; kwargs...)
contract_onesite_dispatch(env::CTMRG3DOctant, M, ram_gib::Nothing; kwargs...) = contract_onesite(expand_to_ctmrg3d(env), M)
contract_onesite_dispatch(env::CTMRG3DOctant, M, ram_gib::Real; kwargs...) = contract_onesite_auto(expand_to_ctmrg3d(env), M; ram_gib, kwargs...)
