# Belief propagation (BP) for the 3D one-site network O of a PEPO, and the analytic Bethe-lattice solution of the classical Ising model.
#
# Network: every site carries the same O (0,6) with legs [UP NORTH EAST DOWN SOUTH WEST] (the O of expectation_value / CTMRG3D).
# The UP leg of a site is contracted with the DOWN leg of the site above it, NORTH with SOUTH, EAST with WEST.
# BP replaces the environment of a site by six vectors (messages) μ[k], one on every leg k: μ[k] is the message that arrives on leg k.
# By translation invariance the message on our UP leg is computed on the site above us: we contract its O with the messages on all its legs
# except its DOWN leg, so
#     μ[k] ∝ Σ O[... leg opp(k) free ...] Π_{j ≠ opp(k)} μ[j],      opp = (DOWN, SOUTH, WEST, UP, NORTH, EAST)
# A fixed point of this map is the Bethe approximation of the network (exact on a tree with coordination z = 6); expectation values are
#     ⟨obs⟩ = Σ M Π_k μ[k] / Σ O Π_k μ[k]
# For the classical Ising model with O = the exact Boltzmann network this gives the Bethe-lattice result bethe_ising_magnetization(β; z = 6).
#
# Restriction: we work with plain arrays (convert(Array, O)), so O must have no symmetry sectors (ComplexSpace/CartesianSpace) and be bosonic.
# The messages are not positive in general (the virtual basis of a CE PEPO is not the spin basis), so we fix the sign of every message by its overlap
# with the previous one.

const BP_OPP = (4, 5, 6, 1, 2, 3)

# we contract the array O (6 legs) with the vectors μs[j] on all legs j ≠ f, leaving leg f open
function _bp_contract_except(O::AbstractArray, μs, f::Int)
    others = Tuple(j for j in 1:6 if j != f)
    T = reshape(permutedims(O, (f, others...)), size(O, f), :)
    v = reduce(kron, (μs[j] for j in reverse(others)))   # column major: the first of `others` runs fastest
    return T * v
end

# Σ O Π_k μ[k]
_bp_contract_all(O::AbstractArray, μs) = dot(reduce(kron, (μs[j] for j in 6:-1:1)), vec(O))

function _bp_array(O::AbstractTensorMap)
    sectortype(O) == Trivial || throw(ArgumentError("bp_fixed_point: O must have no symmetry sectors (ComplexSpace/CartesianSpace)"))
    return convert(Array, O)
end

# BP fixed point of the network of O.
#   init = :random (seed init_seed, breaks the Z2 symmetry), :ones, or μ0 = the messages of an earlier run (warm start, e.g. the previous β)
#   damping α: μ ← (1-α) μ_new + α μ_old (0 = no damping)
# returns (μs, ϵs): the messages (a vector of 6 vectors) and the change per iteration (max over k of ‖μ_new[k] - μ_old[k]‖, messages normalized)
function bp_fixed_point(O::AbstractTensorMap; init = :random, init_seed = 1, μ0 = nothing, maxiter = 10_000, tol = 1e-12, damping = 0.0, verbosity = 0)
    o = _bp_array(O)
    T = promote_type(eltype(o), Float64)
    μs = if μ0 !== nothing
        [T.(copy(μ)) for μ in μ0]
    elseif init == :random
        rng = Random.Xoshiro(init_seed)
        [randn(rng, T, size(o, k)) for k in 1:6]
    elseif init == :ones
        [ones(T, size(o, k)) for k in 1:6]
    else
        throw(ArgumentError("bp_fixed_point: init must be :random or :ones (or give μ0)"))
    end
    μs = [μ / norm(μ) for μ in μs]
    ϵs = Float64[]
    for it in 1:maxiter
        new = similar(μs)
        for k in 1:6
            μ = _bp_contract_except(o, μs, BP_OPP[k])
            μ /= norm(μ)
            s = real(dot(μs[k], μ))
            μ = s < 0 ? -μ : μ
            μ = (1 - damping) * μ + damping * μs[k]
            new[k] = μ / norm(μ)
        end
        ϵ = maximum(norm(new[k] - μs[k]) for k in 1:6)
        push!(ϵs, ϵ)
        μs = new
        verbosity > 1 && println("BP iteration $it: ϵ = $ϵ")
        ϵ < tol && break
    end
    verbosity > 0 && println("BP: $(length(ϵs)) iterations, ϵ = $(ϵs[end])")
    return μs, ϵs
end

# ⟨obs⟩ = Σ M Π μ / Σ O Π μ for a fixed point μs of bp_fixed_point(O)
bp_expectation_value(O::AbstractTensorMap, M::AbstractTensorMap, μs) = _bp_contract_all(_bp_array(M), μs) / _bp_contract_all(_bp_array(O), μs)

# Same entry point as expectation_value(PEPO, obss, ...) of CTMRG3D: we build O and M in exactly the same way (trace of the physical legs with
# twist(PEPO, 2), then flip of legs 4 5 6) and replace the CTMRG3D environment by the BP messages.
# return_messages = true: returns (expvals, μs, ϵs), so that μs can be the μ0 of the next β (warm start)
function bp_expectation_value(PEPO::AbstractTensorMap{A,S,2,6}, obss; init = :random, init_seed = 1, μ0 = nothing, maxiter = 10_000, tol = 1e-12,
                              damping = 0.0, verbosity = 0, return_messages = false) where {A,S}
    @tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]
    O = flip(permute(O, ((),(1,2,3,4,5,6))), [4 5 6])
    μs, ϵs = bp_fixed_point(O; init, init_seed, μ0, maxiter, tol, damping, verbosity)
    expvals = map(obss) do obs
        @tensor M[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 2; -1 -2 -3 -4 -5 -6] * obs[2; 1]
        M = flip(permute(M, ((),(1,2,3,4,5,6))), [4 5 6])
        bp_expectation_value(O, M, μs)
    end
    return return_messages ? (expvals, μs, ϵs) : expvals
end

# Analytic Bethe-lattice (BP) magnetization of the classical Ising model H = -J Σ s_i s_j at inverse temperature β, coordination z.
# The cavity field u (one neighbour) solves u = atanh(tanh(βJ) tanh((z-1) u)) and m = tanh(z u).
# Nonzero for (z-1) tanh(βJ) > 1, i.e. β > βc^Bethe = atanh(1/(z-1)) / J (= ½ ln(3/2) ≈ 0.2027 for z = 6).
function bethe_ising_magnetization(β; z = 6, J = 1.0)
    t = tanh(β * J)
    (z - 1) * t <= 1 && return 0.0
    g(u) = atanh(t * tanh((z - 1) * u)) - u
    lo, hi = 1e-12, atanh(t) + 1.0          # g(lo) > 0 and g(hi) < 0
    for _ in 1:200
        mid = (lo + hi) / 2
        g(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    return tanh(z * (lo + hi) / 2)
end
