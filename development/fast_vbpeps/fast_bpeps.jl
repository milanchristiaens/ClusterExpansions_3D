# The vbPEPS of ClusterExpansions (src/bpeps/) on BetterPEPSKit (Lander Burgelman e.a.), the fast variant of the vbPEPS compare (VBPEPS_IMPL = :betterpepskit). It runs in its own environment (development/fast_vbpeps), because BetterPEPSKit needs MPSKit 0.14 and OptimKit 0.5,
# and reads O, M and the start states from the PEPO file as plain arrays. It is the algorithm of bpeps_optimize / bpeps_observables with the improvements of Lander's example.jl: block-Krylov decompositions, a gap cut of the spectrum (truncgap), ImplicitGradient with GMRES and a start environment grown from a product state.
module FastBpeps

using LinearAlgebra
using Random
using TensorKit
using BetterPEPSKit
using KrylovKit
using OptimKit
using Zygote
using TimerOutputs

const PEPSKit = BetterPEPSKit   # the PEPSKit-qualified names of ClusterExpansions work unchanged
const to = TimerOutput()

# ----------------------------------------------------------------------------------------------------------------------
# Tensors as plain data (pack_tensor of development/initial_pepos/pepo_files.jl): (; cod, dom, data) with cod/dom = [(dimension, isdual) per leg] and the dense array
# (legs: codomain first, then domain). Only for spaces without symmetry (ComplexSpace), as the Ising PEPOs.
# ----------------------------------------------------------------------------------------------------------------------
_space(v) = v[2] ? ComplexSpace(v[1])' : ComplexSpace(v[1])
_prodspace(vs) = isempty(vs) ? one(ComplexSpace) : ⊗(map(_space, vs)...)
unpack_tensor(p) = TensorMap(p.data, _prodspace(p.cod) ← _prodspace(p.dom))

# ----------------------------------------------------------------------------------------------------------------------
# Optimization state x = (peps, env2, env3), retraction, transport and symmetrization: as in section 1 of bpeps_helpers.jl
# ----------------------------------------------------------------------------------------------------------------------
function bpeps_retract((peps, env2, env3), η, α)
    (peps´, env2´), ξ = PEPSKit.peps_retract((peps, env2), η, α)
    env3´ = deepcopy(env3)
    return (peps´, env2´, env3´), ξ
end

function bpeps_transport!(ξ, (peps, env2, env3), η, α, (peps´, env2´, env3´))
    return PEPSKit.peps_transport!(ξ, (peps, env2), η, α, (peps´, env2´))
end

function bpeps_symmetrize_retract_and_finalize!(symm, retract = bpeps_retract, (finalize!) = OptimKit._finalize!)
    isnothing(symm) && return retract, finalize!
    function symmetrize_then_finalize!((peps, env2, env3), E, grad, numiter)
        grad_symm = symmetrize!(grad, symm)
        return finalize!((peps, env2, env3), E, grad_symm, numiter)
    end
    function retract_then_symmetrize((peps, env2, env3), η, α)
        (peps´, env2´, env3´), ξ = retract((peps, env2, env3), η, α)
        return (symmetrize!(peps´, symm), env2´, env3´), symmetrize!(ξ, symm)
    end
    return retract_then_symmetrize, symmetrize_then_finalize!
end

# ----------------------------------------------------------------------------------------------------------------------
# Start environment (Lander's tools.jl): grown from a product state by steps of the CTMRG algorithm itself (dense or block-Krylov), with the cut moved
# to a gap of the spectrum (window: how far from χenv the cut may move). The block-Krylov projectors need such a start.
# ----------------------------------------------------------------------------------------------------------------------
function bpeps_initial_environment(alg, network, χ::Int; window::Int = 4)
    env₀, = PEPSKit.SparseProjectors.product_env(network, χ; alg, maxgrowth = 40)
    env, = PEPSKit.SparseProjectors.gap_grown_env(network, env₀, χ; alg, window)
    return env
end

# ----------------------------------------------------------------------------------------------------------------------
# Timers inside a function that Zygote differentiates (as section 3 of bpeps_helpers.jl): start and stop a timer of `to`, hidden from Zygote
# ----------------------------------------------------------------------------------------------------------------------
# (with the public manual sections of TimerOutputs, begin_timed_section!/end_timed_section!: this environment has TimerOutputs 1.x, whose internals
# differ from the v0.5 internals that bpeps_helpers.jl uses)
_bpeps_tic(label::String) = PEPSKit.ignore_derivatives(() -> to.enabled ? TimerOutputs.begin_timed_section!(to, label) : nothing)
function _bpeps_toc(tok)
    tok === nothing || PEPSKit.ignore_derivatives(() -> (TimerOutputs.end_timed_section!(to, tok); nothing))
    return nothing
end

# ----------------------------------------------------------------------------------------------------------------------
# The cost function x = (peps, env2, env3) -> (cost, gradient), as section 4 of bpeps_helpers.jl:
#   differentiate_environment = false: the environments are converged outside of Zygote and kept fixed in the gradient
#   differentiate_environment = true:  the gradient goes through the CTMRG (hook_pullback with ImplicitGradient)
# boundary_alg2 and boundary_alg3 are separate copies of the CTMRG algorithm (the block-Krylov state of each network).
# ----------------------------------------------------------------------------------------------------------------------
function generate_bpeps_cost_function(T::InfinitePEPO; symmetrization = nothing, boundary_alg2, boundary_alg3, gradient_alg, differentiate_environment::Bool = false)
    function cost_from_values(n3, env3, n2, env2)
        tok = _bpeps_tic("bPEPS: network_value and cost")
        λ3 = network_value(n3, env3)
        λ2 = network_value(n2, env2)
        cost = -log(abs(λ3 / λ2))
        _bpeps_toc(tok)
        return cost
    end

    function bpeps_costfun_fixed_env((peps, env2, env3))
        return @timeit to "bPEPS: cost function (value + gradient)" begin
            n2 = InfiniteSquareNetwork(peps)
            n3 = InfiniteSquareNetwork(peps, T)
            env2′, = @timeit to "bPEPS: n2 leading_boundary (CTMRG of <psi|psi>)" leading_boundary(env2, n2, boundary_alg2)
            env3′, = @timeit to "bPEPS: n3 leading_boundary (CTMRG of <psi|O|psi>)" leading_boundary(env3, n3, boundary_alg3)
            PEPSKit.update!(env2, env2′)
            PEPSKit.update!(env3, env3′)
            E, back = @timeit to "bPEPS: forward pass" Zygote.pullback(
                ψ -> cost_from_values(InfiniteSquareNetwork(ψ, T), env3, InfiniteSquareNetwork(ψ), env2), peps,
            )
            gs = @timeit to "bPEPS: backward pass (gradient of the network values)" back(one(E))
            g = only(gs)
            @timeit to "bPEPS: symmetrize gradient" symmetrize!(g, symmetrization)
            (E, g)
        end
    end

    function forward(ψ, env2, env3)
        n2 = InfiniteSquareNetwork(ψ)
        tok = _bpeps_tic("bPEPS: n2 leading_boundary (CTMRG of <psi|psi>)")
        env2′, = PEPSKit.hook_pullback(leading_boundary, env2, n2, boundary_alg2; alg_rrule = gradient_alg)
        _bpeps_toc(tok)
        n3 = InfiniteSquareNetwork(ψ, T)
        tok = _bpeps_tic("bPEPS: n3 leading_boundary (CTMRG of <psi|O|psi>)")
        env3′, = PEPSKit.hook_pullback(leading_boundary, env3, n3, boundary_alg3; alg_rrule = gradient_alg)
        _bpeps_toc(tok)
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env2, env2′)
            PEPSKit.update!(env3, env3′)
        end
        return cost_from_values(n3, env3′, n2, env2′)
    end

    function bpeps_costfun_differentiated_env((peps, env2, env3))
        return @timeit to "bPEPS: cost function (value + gradient)" begin
            E, back = @timeit to "bPEPS: forward pass" Zygote.pullback(ψ -> forward(ψ, env2, env3), peps)
            gs = @timeit to "bPEPS: backward pass (gradient through CTMRG)" back(one(E))
            g = only(gs)
            @timeit to "bPEPS: symmetrize gradient" symmetrize!(g, symmetrization)
            (E, g)
        end
    end

    return differentiate_environment ? bpeps_costfun_differentiated_env : bpeps_costfun_fixed_env
end

# ----------------------------------------------------------------------------------------------------------------------
# bpeps_optimize / bpeps_observables: as in bpeps_main.jl, with O the 2D PEPO tensor (bpeps_pepo_tensor of ClusterExpansions) and the start state psi0 (an InfinitePEPS) given, because its power method is a function of ClusterExpansions.
# Keywords as bpeps_optimize, and: boundary_c4v (C4vCTMRG, or SimultaneousCTMRG), symmetric (RotateReflect() or none), sparse (block-Krylov or dense decompositions),
# truncation_window and truncation_rtol (the gap cut of truncgap), gradient_tol, gradient_maxiter and gradient_krylovdim (the GMRES solver of ImplicitGradient)
# ----------------------------------------------------------------------------------------------------------------------
struct BpepsTimeLimit <: Exception end

function bpeps_optimize(
        O, psi0, χpeps::Int, χenv::Int;
        boundary_c4v::Bool = true, symmetric::Bool = true, sparse::Bool = true, truncation_window::Int = 4, truncation_rtol = 1.0e-3,
        boundary_tol = 1.0e-8, boundary_maxiter = 400, boundary_verbosity = 0,
        gradient_tol = 1.0e-6, gradient_maxiter = 20, gradient_krylovdim = 30,
        optimizer_memory = 64, optimizer_tol = 1.0e-5, optimizer_maxiter = 75, optimizer_verbosity = 0,
        state0 = nothing, differentiate_environment::Bool = false, max_seconds = Inf,
    )
    t_start = time()
    T = InfinitePEPO(O)
    symmetrization = symmetric ? RotateReflect() : nothing

    # the CTMRG with the gap cut and (sparse = true) block-Krylov decompositions; with block-Krylov the last Ritz value of every sector is not trusted
    # for the cut (margin = 1)
    trunc = truncgap(χenv; window = truncation_window, margin = sparse ? 1 : 0, rtol = truncation_rtol)
    decomposition = sparse ? (; decomposition_alg = (; alg = :BlockKrylov)) : (;)
    boundary_alg_type = boundary_c4v ? C4vCTMRG : SimultaneousCTMRG
    boundary_alg = boundary_alg_type(; tol = boundary_tol, maxiter = boundary_maxiter, verbosity = boundary_verbosity, trunc, decomposition...)
    gradient_alg = ImplicitGradient(; tol = gradient_tol, solver_alg = (; alg = :GMRES, krylovdim = gradient_krylovdim, maxiter = gradient_maxiter))
    optimizer_alg = LBFGS(optimizer_memory; gradtol = optimizer_tol, maxiter = optimizer_maxiter, verbosity = optimizer_verbosity)

    if state0 === nothing
        psi_start = @timeit to "bPEPS: symmetrize and normalize initial state" PEPSKit.peps_normalize(symmetrize!(deepcopy(psi0), symmetrization))
        env2_0, env3_0 = @timeit to "bPEPS: initial CTMRG environments" begin
            (bpeps_initial_environment(boundary_alg, InfiniteSquareNetwork(psi_start), χenv; window = truncation_window),
             bpeps_initial_environment(boundary_alg, InfiniteSquareNetwork(psi_start, T), χenv; window = truncation_window))
        end
    else
        psi_start, env2_0, env3_0 = state0
    end

    # one copy of the algorithm per network (block-Krylov state)
    costfun = generate_bpeps_cost_function(T; symmetrization, boundary_alg2 = deepcopy(boundary_alg), boundary_alg3 = deepcopy(boundary_alg),
                                           gradient_alg, differentiate_environment)
    retract, finalize! = bpeps_symmetrize_retract_and_finalize!(symmetrization, bpeps_retract)

    ncalls = Ref(0)
    iters_done = Ref(0)
    last_state = Ref{Any}(nothing)
    function counted_costfun(x)
        # also inside the line search: once an iteration is finished, we stop before a new call if the time is up
        iters_done[] > 0 && time() - t_start > max_seconds && throw(BpepsTimeLimit())
        ncalls[] += 1
        t_call = time()
        res = costfun(x)
        optimizer_verbosity >= 2 && @info "bPEPS: cost-function call $(ncalls[]) took $(round(time() - t_call; digits = 1)) s"
        return res
    end
    function finalize_limited!(x, f, g, numiter)
        x, f, g = finalize!(x, f, g, numiter)
        last_state[] = (x, f)
        iters_done[] = numiter
        numiter > 0 && time() - t_start > max_seconds && throw(BpepsTimeLimit())
        return x, f, g
    end

    local psi, env2, env3, cost, numfg
    timed_out = false
    try
        (psi, env2, env3), cost, _, numfg, _ = @timeit to "bPEPS: LBFGS optimization" optimize(
            counted_costfun, (psi_start, env2_0, env3_0), optimizer_alg;
            inner = PEPSKit.real_inner, retract, (transport!) = bpeps_transport!, (finalize!) = finalize_limited!,
        )
    catch err
        err isa BpepsTimeLimit || rethrow()
        (psi, env2, env3), cost = last_state[]
        numfg = ncalls[]
        timed_out = true
    end
    return (; psi, env2, env3, cost, numfg, timed_out)
end

# ⟨M⟩ = ⟨ψ|M|ψ⟩ / ⟨ψ|O|ψ⟩ on the converged state (M = the 2D tensor of the observable, bpeps_pepo_tensor(PEPO, obs) of ClusterExpansions)
function bpeps_observables(O, Ms, result)
    n3 = InfiniteSquareNetwork(result.psi, InfinitePEPO(O))
    den = @timeit to "bPEPS: observable denominator (O)" PEPSKit._contract_site((1, 1), n3, result.env3)
    return [(@timeit to "bPEPS: observable numerator (M)" PEPSKit.contract_local_tensor((1, 1, 1), M, n3, result.env3)) / den for M in Ms]
end

# ----------------------------------------------------------------------------------------------------------------------
# The interface for the scripts (the tensors come packed from the PEPO file)
# ----------------------------------------------------------------------------------------------------------------------
const STATES = Dict{String, Any}()     # (psi, env2, env3) of the last observation of every case, for the warm start

# One observation: optimize (warm start from STATES[key] if warm, otherwise from psi0p, the packed start tensor) and measure M.
# Returns (; cost, numfg, timed_out, m).
function run_observation(key::String, Op, Mp, psi0p, χpeps::Int, χenv::Int; warm::Bool, seed::Int, kwargs...)
    Random.seed!(seed)
    O, M = unpack_tensor(Op), unpack_tensor(Mp)
    psi0 = psi0p === nothing ? nothing : InfinitePEPS(unpack_tensor(psi0p))
    state0 = warm ? STATES[key] : nothing
    result = @timeit to "expectation_value (vbPEPS and observable)" begin
        r = bpeps_optimize(O, psi0, χpeps, χenv; state0, kwargs...)
        (; r, m = only(bpeps_observables(O, [M], r)))
    end
    STATES[key] = (result.r.psi, result.r.env2, result.r.env3)
    return (; cost = result.r.cost, numfg = result.r.numfg, timed_out = result.r.timed_out, m = ComplexF64(result.m))
end

# The timer as a plain Dict (TimerOutputs.todict); and a reset (at the start of every case)
timer_dict() = TimerOutputs.todict(to)
reset_timer() = (reset_timer!(to); empty!(STATES); nothing)
set_blas_threads(n::Int) = (BLAS.set_num_threads(n); BLAS.get_num_threads())
peak_memory_gib() = Sys.maxrss() / 2^30

end # module FastBpeps
