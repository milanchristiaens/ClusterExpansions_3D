# ctmrg3D_octant.jl -- cube-symmetric CTMRG3D with octant projectors (CTMRG3DOctant): one step grows all six sides at once, so we store one corner c, one edge e and one face a, compressed with two projectors P_c and P_f (a step is exact without truncation).
# Use: expectation_value(PEPO, obss, bpepsspace, envspace; ctm_alg = CTMRG3DOctant, ...); it needs a real O with the cube symmetry (rot_invariance = true) and no symmetry sectors.
# Plain arrays: O[u, n, e, d, s, w], c[x, y, z] (the corner at (+,+,+), legs to -x, -y, -z), e[c-, c+, fU, fN] (the x-edge at (0,+,+)), a[o, +x, +y, -x, -y] (the up face at (0,0,+)).

using LinearAlgebra: I, Diagonal, Symmetric, diagm, eigen, qr, svdvals, tr

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
                                       # (default min(1e-6, 10·tol): the CTMRG3D change ϵ then still reaches tol)
    subspace_maxiter::Int              # at most this many products ρ_F·V per step
    oversample::Int                    # block size = χenv + oversample
    trunc_window::Int                  # gap cut (see _gap_cut): the cuts of P_c (χenv) and P_f (χpeps) may move by at most trunc_window
    trunc_rtol::Float64                # gap cut: two values count as one group when they differ by at most trunc_rtol (relative)
    face_projector::Symbol             # P_f from the enlarged edge: :unweighted, :sigma_weighted (corner legs weighted with σ) or :lead_weighted (corner legs closed with the leading vector of σ)
    eigensolver::Symbol                # how we find the top eigenvectors of ρ_F: :subspace, :block_lanczos, :dense_block_lanczos or :dense_eigen (see _octant_top_eigvecs)
    stagnation_window::Int             # stagnation stop of leading_boundary (0 = off)
    stagnation_factor::Float64
end

# Same keywords as CTMRG3D (so that expectation_value(...; ctm_alg = CTMRG3DOctant) works); projectors, projector_exponent, class_projectors,
# west_updates and west_tol have no meaning here and are ignored.
#   init: the start is the 3×3×3 cube of O tensors with the outer legs closed by a vector v (as init = :nishino_okunishi of CTMRG3D):
#         :random or :nishino_okunishi -> v Gaussian (seed init_seed), :ones -> v = ones. The bonds start at D and grow to χenv, χpeps.
#   env0: an earlier CTMRG3DOctant (warm start, e.g. the previous β): we copy c, e, a, P.
#   trunc_window, trunc_rtol: the gap cut of P_c and P_f (_gap_cut), the same rule and defaults as truncgap of BetterPEPSKit in the vbPEPS compare
#         (window 4, rtol 1e-3); trunc_window = 0 gives the plain cut at χenv, χpeps.
#   face_projector: :unweighted, :sigma_weighted or :lead_weighted (P_f from the enlarged edge, _octant_face_projector_edge; :lead_weighted closes
#         the corner legs with the leading eigenvector of σ instead of weighting them with σ)
#   eigensolver: P_c from ρ_F by subspace iteration (:subspace), block Lanczos on ρ_F as a linear map (:block_lanczos), or ρ_F built as a dense matrix
#         and then block Lanczos (:dense_block_lanczos) or a dense eigendecomposition (:dense_eigen)
#   stagnation_window, stagnation_factor: the stagnation stop of leading_boundary (0 = off)
function CTMRG3DOctant(O::AbstractTensorMap{T,S,0,6}; Vv = oneunit(space(O)[1]), Vbp = oneunit(space(O)[1]), trunc_env = nothing, trunc_bpeps = nothing,
                       miniter::Int = 5, maxiter::Int = 150, tol::Float64 = 1e-8, verbosity::Int = 0, env0 = nothing, init::Symbol = :random,
                       init_seed::Int = 1, max_seconds::Real = Inf, subspace_tol::Real = min(1e-6, 10 * tol), subspace_maxiter::Int = 30, oversample::Int = 8,
                       trunc_window::Int = 4, trunc_rtol::Real = 1e-3, face_projector::Symbol = :lead_weighted, eigensolver::Symbol = :subspace, stagnation_window::Int = 0, stagnation_factor::Real = 0.5, kwargs...) where {T,S}
    T <: Real || throw(ArgumentError("CTMRG3DOctant: O must be real (Float64), got $T"))
    sectortype(O) == Trivial || throw(ArgumentError("CTMRG3DOctant: O must have no symmetry sectors"))
    init in (:ones, :random, :nishino_okunishi) || throw(ArgumentError("init must be :ones, :random or :nishino_okunishi, got :$init"))
    face_projector in (:unweighted, :sigma_weighted, :lead_weighted) || throw(ArgumentError("face_projector must be :unweighted, :sigma_weighted or :lead_weighted, got :$face_projector"))
    eigensolver in (:subspace, :block_lanczos, :dense_block_lanczos, :dense_eigen) ||
        throw(ArgumentError("eigensolver must be :subspace, :block_lanczos, :dense_block_lanczos or :dense_eigen, got :$eigensolver"))
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
                                       Float64(subspace_tol), subspace_maxiter, oversample, trunc_window, Float64(trunc_rtol), face_projector, eigensolver,
                                       stagnation_window, Float64(stagnation_factor))
end

function _check_cube_symmetric(o)
    D = size(o, 1)
    all(==(D), size(o)) || throw(ArgumentError("CTMRG3DOctant needs the same bond dimension on all six legs of O, got $(size(o))"))
    for perm in ((2, 1, 3, 5, 4, 6), (3, 2, 1, 6, 5, 4), (1, 3, 2, 4, 6, 5), (4, 5, 6, 1, 2, 3), (4, 2, 3, 1, 5, 6))
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

# ρ_F·V = X·M·Xᵀ·V; _octant_rho_columns counts the columns of all products (for the eigensolver checks)
const _octant_rho_columns = Ref(0)
function _octant_rhoV(V, M, e, a, o)
    χ, p, D, b = size(e, 1), size(e, 3), size(o, 1), size(V, 2)
    _octant_rho_columns[] += b
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

# The gap cut, the rule of truncgap in BetterPEPSKit (the truncation of the vbPEPS compare), for values without symmetry sectors:
#   vals: the values, descending (≥ 0: eigenvalues of ρ_F or of the face matrix); target: the number we want to keep (χenv or χpeps).
#   Two neighbours v ≥ v′ form one group if v - v′ <= rtol·v. If the cut at target splits a group, we look for the cut within target ± window
#   that splits none, nearest to target (on a tie: the larger gap, i.e. the smaller ratio v′/v). The last `margin` values are not trusted (the
#   last Ritz values of the block are not converged, as the last value of the block-Krylov decomposition in BetterPEPSKit): the first dropped
#   value must lie above them. If no cut fits, we keep the largest k < target that splits no group (all of the group is dropped, as truncmultiplets).
#   Returns the number of values to keep.
function _gap_cut(vals, target; window = 4, rtol = 1e-3, margin = 0)
    n = length(vals)
    (window <= 0 || target >= n) && return min(target, n)
    splits(k) = vals[k] - vals[k + 1] <= rtol * vals[k]
    splits(target) || return target
    floor = margin > 0 && n > margin ? vals[n - margin] : -Inf
    cands = [k for k in max(1, target - window):min(n - 1, target + window) if !splits(k) && vals[k + 1] >= floor]
    if isempty(cands)
        k = target - 1
        while k >= 1 && splits(k)
            k -= 1
        end
        return k >= 1 ? k : target
    end
    return cands[argmin([(abs(k - target), vals[k + 1] / vals[k]) for k in cands])]
end

# ρ_F as a dense F × F matrix: the products ρ_F·V with V the columns of the identity, `chunk` columns at a time
function _octant_rho_dense(M, e, a, o; chunk = 128)
    F = size(e, 1) * size(e, 3)^2 * size(o, 1)
    T = eltype(e)
    ρ = Matrix{T}(undef, F, F)
    for j0 in 1:chunk:F
        cols = j0:min(j0 + chunk - 1, F)
        V = zeros(T, F, length(cols))
        for (k, j) in enumerate(cols)
            V[j, k] = one(T)
        end
        ρ[:, cols] = _octant_rhoV(V, M, e, a, o)
    end
    return Symmetric((ρ + transpose(ρ)) / 2)
end

# ρ_F / s as an operator of KrylovKit (s: a scale, so that the tolerance of block Lanczos is relative to the largest eigenvalue).
# rho is the dense ρ_F, or nothing for the linear map X·M·Xᵀ. For a block we do one product for all its vectors.
struct _OctantRho{T, TM, TR}
    M::TM
    e::Array{T,4}
    a::Array{T,5}
    o::Array{T,6}
    rho::TR
    s::T
end
_octant_apply(op::_OctantRho, V::AbstractMatrix) = (op.rho === nothing ? _octant_rhoV(V, op.M, op.e, op.a, op.o) : op.rho * V) ./ op.s
(op::_OctantRho)(x::AbstractVector) = vec(_octant_apply(op, reshape(x, :, 1)))
function KrylovKit.apply(op::_OctantRho, b::KrylovKit.Block)
    Y = _octant_apply(op, reduce(hcat, b.vec))
    return KrylovKit.Block([Y[:, j] for j in axes(Y, 2)])
end

# Top nkeep eigenvectors of ρ_F = Q·Qᵀ, with the method env.eigensolver:
#   :subspace             block subspace iteration with Rayleigh-Ritz on the linear map ρ_F = X·M·Xᵀ (block nkeep + oversample)
#   :block_lanczos        block Lanczos of KrylovKit on the same linear map (block nkeep + oversample, Krylov dimension three blocks)
#   :dense_block_lanczos  ρ_F built once as a dense matrix (_octant_rho_dense), then the same block Lanczos
#   :dense_eigen          ρ_F built once as a dense matrix, then the top nkeep + oversample eigenpairs of LAPACK
# All start from the P_c of the previous step (warm start). Returns (U, λ, WU): U (F × k) orthonormal, λ the eigenvalues (descending),
# WU = Qᵀ·U (F² × k), where k = nkeep moved by the gap cut (_gap_cut; with nkeep + oversample values we can also see the gap above nkeep).
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
    solver = b == F ? :subspace : env.eigensolver
    if solver === :subspace
        local λ, Vk
        for it in 1:env.subspace_maxiter
            Y = @timeit to "octant: ρ_F·V" _octant_rhoV(V, M, e, a, o)
            H = Symmetric(V' * Y)
            ev = eigen(H)
            order = sortperm(ev.values; rev = true)
            λ, S = ev.values[order], ev.vectors[:, order]
            Vk = V * S                                 # the Ritz vectors of this V (also when the last iteration does not converge)
            b == F && break
            R = Y * S[:, 1:nkeep] - V * S[:, 1:nkeep] * Diagonal(λ[1:nkeep])
            res = maximum(norm(R[:, k]) for k in 1:nkeep) / max(abs(λ[1]), eps(T))
            env.verbosity > 2 && @info "octant subspace iteration $it: residual $res"
            res < env.subspace_tol && break
            V = Matrix(qr(Y * S).Q)                    # one power step (Y·S spans ρ_F·V)
        end
        margin = b == F ? 0 : 1                        # on the full space all values are exact, otherwise the last Ritz value is not trusted
    else
        ρ = solver === :block_lanczos ? nothing : @timeit to "octant: dense ρ_F" _octant_rho_dense(M, e, a, o)
        if solver === :dense_eigen
            ev = @timeit to "octant: dense eigen" eigen(ρ, (F - b + 1):F)
            order = sortperm(ev.values; rev = true)
            λ, Vk = ev.values[order], ev.vectors[:, order]
            margin = 0
        else
            Y0 = ρ === nothing ? _octant_rhoV(V, M, e, a, o) : ρ * V
            s = maximum(norm(Y0[:, j]) for j in axes(Y0, 2))
            op = _OctantRho(M, e, a, o, ρ, s)
            alg = BlockLanczos(; krylovdim = 3 * b, maxiter = env.subspace_maxiter, tol = env.subspace_tol, verbosity = 0)
            vals, vecs, info = @timeit to "octant: block Lanczos" eigsolve(op, KrylovKit.Block([V[:, j] for j in 1:b]), b, :LR, alg)
            env.verbosity > 2 && @info "octant block Lanczos: $(info.converged) of $b converged, $(info.numops) products"
            λ, Vk = s .* vals, reduce(hcat, vecs)
            margin = 1
        end
    end
    # the gap cut
    k = _gap_cut(λ, nkeep; window = env.trunc_window, rtol = env.trunc_rtol, margin)
    k != nkeep && env.verbosity > 0 && @info "octant: the cut of P_c at $nkeep splits a near-degenerate group (λ ratio $(round(λ[nkeep + 1] / λ[nkeep]; sigdigits = 4))); keeping $k"
    U = Vk[:, 1:k]
    WU = @timeit to "octant: Qᵀ·P_c (for C')" _octant_QtV(U, B, e, a, o)
    return U, λ[1:k], WU
end

# The enlarged edge e·A_U·A_N·O with its two corner legs projected by P_c and both face legs open: Et[i, j, (ym, Os), (zm, Od)]
# (the e′ of _octant_renormalize before P_f)
function _octant_edge(e, a, o, Pc)
    χ, p, D = size(e, 1), size(e, 3), size(o, 1)
    χn = size(Pc, 2)
    P5 = reshape(Pc, χ, p, p, D, χn)
    @tensor T1[xp, u, xm, ym, oN, Oe, Od, Ow, Os] := a[oU, xp, u, xm, ym] * o[oU, oN, Oe, Od, Os, Ow]
    @tensor T2[xp, u, xm, ym, Os, Oe, Ow, n, xp2, xm2, zm, Od] := T1[xp, u, xm, ym, oN, Oe, Od, Ow, Os] * a[oN, n, xp2, zm, xm2]
    @tensor EP[op, u, n, xm2, xm, Ow, i] := e[om, op, u, n] * P5[om, xm2, xm, Ow, i]
    @tensor T3[xp, ym, Os, Oe, xp2, zm, Od, op, i] := T2[xp, u, xm, ym, Os, Oe, Ow, n, xp2, xm2, zm, Od] * EP[op, u, n, xm2, xm, Ow, i]
    @tensor Et[i, j, ym, Os, zm, Od] := T3[xp, ym, Os, Oe, xp2, zm, Od, op, i] * P5[op, xp, xp2, Oe, j]
    return reshape(Et, χn, χn, p * D, p * D)
end

# P_f from the enlarged edge: the Gram matrix on one face leg, with the other legs glued to a copy, averaged over the two face legs.
# weighted = true (:sigma_weighted): the two corner legs get σ = c′·c′ᵀ (the new corner, unfolded on one leg) between bra and ket; lead = true (:lead_weighted):
# both corner legs are closed with the leading eigenvector v₁ of σ (the limit of σ^p for large p), so the edge becomes a matrix in the two face legs.
# Top nkeep eigenvectors, moved by the gap cut.
function _octant_face_projector_edge(e, a, o, Pc, WP, nkeep; weighted = true, lead = false, window = 4, rtol = 1e-3, verbosity = 0)
    F, χn = size(Pc)
    if weighted
        W3 = reshape(WP, F, F, χn)
        @tensor cn[i, j, k] := W3[y, z, i] * Pc[y, j] * Pc[z, k]
        cm = reshape(cn, χn, :)
        ev = eigen(Symmetric(cm * transpose(cm)))
        if lead
            v1 = ev.vectors[:, argmax(ev.values)]
            S = v1 * transpose(v1)
        else
            S = ev.vectors * Diagonal(sqrt.(max.(ev.values, 0) ./ maximum(ev.values))) * transpose(ev.vectors)
        end
    else
        S = Matrix{eltype(Pc)}(I, χn, χn)
    end
    Et = _octant_edge(e, a, o, Pc)
    @tensor Ew[i, j, k, l] := S[i, i2] * S[j, j2] * Et[i2, j2, k, l]
    n = size(Ew, 3)
    Mk = reshape(permutedims(Ew, (3, 1, 2, 4)), n, :)
    Ml = reshape(permutedims(Ew, (4, 1, 2, 3)), n, :)
    Gk, Gl = Mk * transpose(Mk), Ml * transpose(Ml)
    ρf = Symmetric((Gk / tr(Gk) + Gl / tr(Gl)) / 2)
    evf = eigen(ρf)
    order = sortperm(evf.values; rev = true)
    μ = max.(evf.values[order], 0)
    k = _gap_cut(μ, nkeep; window, rtol, margin = 0)
    k != nkeep && verbosity > 0 && @info "octant: the cut of P_f at $nkeep splits a near-degenerate group (ratio $(round(μ[nkeep + 1] / μ[nkeep]; sigdigits = 4))); keeping $k"
    return evf.vectors[:, order[1:k]]
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
    U, _, WU = @timeit to "octant: P_c (eigenvectors of ρ_F)" _octant_top_eigvecs(env, B, χn)
    Pf = @timeit to "octant: P_f" _octant_face_projector_edge(env.e, env.a, env.o, U, WU, pn; weighted = env.face_projector !== :unweighted,
                                                             lead = env.face_projector === :lead_weighted, window = env.trunc_window,
                                                             rtol = env.trunc_rtol, verbosity = env.verbosity)
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
