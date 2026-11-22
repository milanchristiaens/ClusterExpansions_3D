# Checks of the CTMRG3D additions of 20-21 Nov 2026 (prints numbers; every part says what it should give):
#   :class   class_projectors = true is exact without truncation: a random PEPO (D = 2, real and complex), W, S, D updates (west_updates of rot_ud_90 /
#            rot_ns_90), :svd P = 0 and :oblique must give the same m after every update; the original projectors (class_projectors = false) do not.
#   :octant  CTMRG3DOctant: (1) symmetry of c, e, a and of the 27-tensor copy under rot_ud_90, rot_ns_90, rot_we_90 (≈ 1e-16), (2) exactness: from the same
#            3×3×3 start, one octant step without truncation and one full ctmrgstep of CTMRG3D without truncation (class :svd P = 0, west_updates = 1,
#            six exact one-sided updates) describe the same 5×5×5 cube: equal m (sandbox: 0.266665790505 both). Contractions sliced to RAM_GIB.
#   :bp      bp_fixed_point on the exact Ising network against the analytic bethe_ising_magnetization (6 digits), and on the CE PEPO of the PEPO file.
# Run from the clone (PowerShell): julia +1.12 -t 4 --project=. development/checks/ctmrg_3d/check_ctmrg3d_class_octant_bp.jl
# Time in the sandbox (2 cores): :class about 8 min, :octant about 7 min, :bp about 1 min (D_CE 2 and 3 PEPO files needed for :bp).
using ClusterExpansions, TensorKit, MatrixAlgebraKit, MPSKit, LinearAlgebra, Random, Printf, JLD2
const CE = ClusterExpansions
include(joinpath(@__DIR__, "..", "..", "initial_pepos", "pepo_files.jl"))

# ---------------- settings ----------------
const PARTS = Symbol.(split(get(ENV, "CE_CHECK_PARTS", "class,octant,bp"), ","))
const RAM_GIB = 5                   # RAM budget (GiB) of the sliced contractions of :octant (the 5×5×5 cube at (16, 4, 2) needs about 34 GiB unsliced)
const BP_BETAS = [0.10, 0.19, 0.21, 0.23, 0.25, 0.35]
# ------------------------------------------

# Build the O and M tensors
function OM(PEPO, σ)
    @tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]
    O = flip(permute(O, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])
    @tensor M[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 2; -1 -2 -3 -4 -5 -6] * σ[2; 1]
    M = flip(permute(M, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])
    return O, M
end

if :class in PARTS
    println("===== :class: no truncation, m at the start / after W / after S / after D (all four columns equal within a row type) =====")
    TR = trunctol(; rtol = 1e-12)
    for T in (Float64, ComplexF64)
        Random.seed!(7)
        V = ℂ^2
        PEPO = randn(T, V ⊗ V' ← V ⊗ V ⊗ V ⊗ V' ⊗ V' ⊗ V')
        O, M = OM(PEPO, TensorMap(T[1 0; 0 -1], V ← V))
        # Helper: Calculate the expectation value given env
        mval(env) = CE.contract_onesite(env, M) / CE.contract_onesite(env, O)
        # For all 4 variants
        for (lab, proj, Pexp, class) in (("svd P=0", :svd, 0.0, false), ("oblique", :oblique, 0.5, false), ("svd P=0, class", :svd, 0.0, true), ("oblique, class", :oblique, 0.5, true))
            # Build initial environment, seed 1
            env = CE.CTMRG3D(O; trunc_env = TR, trunc_bpeps = TR, Vbp = ℂ^2, Vv = ℂ^2, init = :random, init_seed = 1, projectors = proj,
                             projector_exponent = Pexp, class_projectors = class)
            # obs            
            ms = [mval(env)]
            
            # Perform partial update + obs
            CE.partial_ctmrgstep(env); push!(ms, mval(env))
            # Perform extra partial update on S + obs
            CE.rot_ud_90(env); CE.partial_ctmrgstep(env); foreach(_ -> CE.rot_ud_90(env), 1:3); push!(ms, mval(env))
            # Perform extra partial update on D + obs
            CE.rot_ns_90(env); CE.partial_ctmrgstep(env); foreach(_ -> CE.rot_ns_90(env), 1:3); push!(ms, mval(env))
            
            # Print the 4 obs
            println(@sprintf("  %-10s %-15s %s", T, lab, join(map(x -> @sprintf("%.10f%+.10fi", real(x), imag(x)), ms), "  ")))
        end
    end
    println("  expected: the two class rows equal each other (also after S and D); the original rows differ after S and D (the gauge problem)")
end

if :octant in PARTS
    println("===== :octant =====")
    # a random O with the full cube symmetry (48 permutations of the legs U N E D S W)
    function ohgroup()
        gens = [(2,1,3,5,4,6), (3,2,1,6,5,4), (1,3,2,4,6,5), (4,5,6,1,2,3), (4,2,3,1,5,6)]
        G = Set([Tuple(1:6)]); frontier = collect(G)
        # Compose k[i] = g[h[i]] untill nothing news appears
        while !isempty(frontier)
            new = []
            for g in frontier, h in gens
                k = Tuple(g[h[i]] for i in 1:6)
                k in G || (push!(G, k); push!(new, k))
            end
            frontier = new
        end
        return collect(G)
    end

    # Construct random o by averaging over all Oh group elements
    G = ohgroup(); D = 2
    Random.seed!(3)
    o = randn(D, D, D, D, D, D); o = sum(permutedims(o, g) for g in G) / length(G)
    # Construct random observable m
    S1 = randn(D, D); S2 = randn(D, D)
    @tensor m[u, n, e, d, s, w] := S1[u, u2] * S2[e, e2] * o[u2, n, e2, d, s, w]     # some observable (not proportional to O)
    spaceO = space(CE.CTMRG3D(TensorMap(o, one(ℂ^1) ← ℂ^D ⊗ ℂ^D ⊗ ℂ^D ⊗ ℂ^D ⊗ ℂ^D ⊗ ℂ^D); Vv = ℂ^1, Vbp = ℂ^1).O)
    Ot, Mt = TensorMap(o, spaceO), TensorMap(m, spaceO)
    # Helper function for obs
    mval27(env27) = CE.contract_onesite_auto(env27, Mt; ram_gib = RAM_GIB) / CE.contract_onesite_auto(env27, Ot; ram_gib = RAM_GIB)
    # Define random initial :nishino_okunishi octant env
    oct = CTMRG3DOctant(Ot; Vv = ℂ^16, Vbp = ℂ^4, init = :random, init_seed = 5)
    # Copy single CEA into 8 C's 12 E's and 6 A's 
    start27 = CE.expand_to_ctmrg3d(oct)

    function symmetry_errors(env)
        c, e, a = env.c, env.e, env.a
        sym = (norm(permutedims(c, (2,3,1)) - c) / norm(c), norm(permutedims(e, (2,1,4,3)) - e) / norm(e), norm(permutedims(a, (1,3,4,5,2)) - a) / norm(a))
        e27 = CE.expand_to_ctmrg3d(env)
        rot = map((CE.rot_ud_90, CE.rot_ns_90, CE.rot_we_90)) do r
            f = deepcopy(e27); r(f)
            maximum([maximum(norm(f.C[i] - e27.C[i]) for i in 1:8), maximum(norm(f.E[i] - e27.E[i]) for i in 1:12), maximum(norm(f.A[i] - e27.A[i]) for i in 1:6)])
        end
        return sym, rot
    end
    CE.ctmrgstep(oct)      # one octant step: F = 2·2·2·2 = 16 = χenv, p·D = 4 = χpeps, so nothing is truncated
    sym, rot = symmetry_errors(oct)
    println("  (1) after one step: symmetry errors of c (C3), e (C2), a (C4): ", sym, "; 27-tensor copy under rot_ud_90, rot_ns_90, rot_we_90: ", rot, " (expected ≈ 1e-16)")
    
    # Now compare octant obs with reference svd (P=0) obs, both after one exact step
    m1 = mval27(CE.expand_to_ctmrg3d(oct))
    env = CE.CTMRG3D(Ot; trunc_env = notrunc(), trunc_bpeps = notrunc(), Vv = ℂ^2, Vbp = ℂ^2, projectors = :svd, projector_exponent = 0.0,
                     class_projectors = true, west_updates = 1, env0 = start27)
    CE.ctmrgstep(env)
    m2 = mval27(env)
    println(@sprintf("  (2) 5×5×5 cube: octant step m = %.12f, CTMRG3D six exact updates m = %.12f, difference %.1e (expected ≈ 1e-12); start m = %.12f",
                     m1, m2, abs(m1 - m2), mval27(start27)))
end

if :bp in PARTS
    println("===== :bp =====")
    σz = TensorMap([1.0 0.0; 0.0 -1.0], ℂ^2 ← ℂ^2)
    # Define the exact Ising network, and get exact O and M
    function exact_OM(β, spaceO)
        W = [exp(β) exp(-β); exp(-β) exp(β)]
        F = eigen(Symmetric(W)); Q = F.vectors * Diagonal(sqrt.(F.values)) * F.vectors'
        o = zeros(2, 2, 2, 2, 2, 2); m = zeros(2, 2, 2, 2, 2, 2)
        for I in CartesianIndices(o), s in 1:2
            w = prod(Q[s, I[k]] for k in 1:6)
            o[I] += w; m[I] += (s == 1 ? 1.0 : -1.0) * w
        end
        return TensorMap(o, spaceO), TensorMap(m, spaceO)
    end
    for β in BP_BETAS
        # Load the CE PEPOS for D=2 and D=3
        P2 = only(load_pepos([β], 0.005, 2, 1.0, 0.0)); P3 = only(load_pepos([β], 0.005, 3, 1.0, 0.0))
        Oce, _ = OM(P2, σz)
        Oex, Mex = exact_OM(β, space(Oce))
        # Calculate the exact BP fixed point and obs
        μ, ϵ = bp_fixed_point(Oex)
        mex = bp_expectation_value(Oex, Mex, μ)
        println(@sprintf("  β %.2f  Bethe (analytic) %.6f | BP exact network %.6f (%d it) | BP CE PEPO D_CE 2: %.6f, D_CE 3: %.6f", β, bethe_ising_magnetization(β),
                         abs(mex), length(ϵ), abs(only(bp_expectation_value(P2, [σz]))), abs(only(bp_expectation_value(P3, [σz])))))
    end
    println("  expected: Bethe = BP exact network to 6 digits; the CE PEPO within about 1e-3 of it (CE/Δβ error); sandbox β 0.21: 0.377982 / 0.377320")
end
