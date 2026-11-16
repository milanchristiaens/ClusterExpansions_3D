# ======================================================================================================================
# bpeps_helpers.jl -- HELPER file of bpeps_main.jl (the entry point: bpeps_expectation_value, bpeps_optimize, bpeps_observables).
# Sections: 1. optimization state (retract, transport)   2. initial CTMRG environment   3. timers inside Zygote   4. cost function
#           5. initial state (power method)   6. the 2D PEPO tensor (bpeps_pepo_tensor)
# The idea of the method and the leg convention are at the top of bpeps_main.jl.
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# 1. The optimization state is x = (peps, env2, env3): the PEPS that is optimized, and the CTMRG environments of the networks <psi|psi> (env2) and
#    <psi|O|psi> (env3) which are only kept to restart the CTMRG from the last solution. Only the PEPS is optimized: the retraction and the transport
#    of PEPSKit act on the first element, the environments are carried along.
# ----------------------------------------------------------------------------------------------------------------------

# OptimKit calls this to take a step, moves from x along the search direction η with step length α
function bpeps_retract((peps, env2, env3), η, α)
    # The peps_retract does the actual step on the PEPS tensors: peps´ = peps + α·η
    # It also returns a copy of env 2 and ξ which is the dir η
    (peps´, env2´), ξ = PEPSKit.peps_retract((peps, env2), η, α)
    # Take copy of env 3
    env3´ = deepcopy(env3)
    return (peps´, env2´, env3´), ξ
end

# LBFGS calls this: LBFGS stores old gradients and steps, this call moves them 
# from old point x to new point x' so they can be combined
function bpeps_transport!(ξ, (peps, env2, env3), η, α, (peps´, env2´, env3´))
    # It only forwards env2 to the standard PEPSKit function, env3 changes nothing
    return PEPSKit.peps_transport!(ξ, (peps, env2), η, α, (peps´, env2´))
end

# Returns the retract and finalize! function that symmetrize the PEPS and its gradient during the optimization.
# symm = nothing (no symmetry) or a PEPSKit symmetrization such as RotateReflect() (C4v symmetric PEPS; use together with boundary_alg_type = C4vCTMRG)
function bpeps_symmetrize_retract_and_finalize!(symm, retract = bpeps_retract, (finalize!) = OptimKit._finalize!)
    # If no symmetry
    isnothing(symm) && return retract, finalize!

    # if symmetry

    # This function is called by OptimKit after every iteration 
    function symmetrize_then_finalize!((peps, env2, env3), E, grad, numiter)
        # symmetrize the gradient, then finalize
        grad_symm = symmetrize!(grad, symm)
        return finalize!((peps, env2, env3), E, grad_symm, numiter)
    end

    function retract_then_symmetrize((peps, env2, env3), η, α)
        # retract, then symmetrize the retracted point and the directional derivative
        (peps´, env2´, env3´), ξ = retract((peps, env2, env3), η, α)
        peps´_symm = symmetrize!(peps´, symm)
        ξ_symm = symmetrize!(ξ, symm)
        return (peps´_symm, env2´, env3´), ξ_symm
    end
    return retract_then_symmetrize, symmetrize_then_finalize!
end


# ----------------------------------------------------------------------------------------------------------------------
# 2. Initial CTMRG environment: the contraction methods need different ones
# ----------------------------------------------------------------------------------------------------------------------

# For SimultaneousCTMRG or SequentialCTMRG, a standard random CTMRGEnv is built from the network and the environment space Venv
bpeps_initial_environment(::Union{SimultaneousCTMRG, SequentialCTMRG}, network, Venv) = CTMRGEnv(network, Venv)

# If C4v: one corner and one edge tensor, built like PEPSKit.initialize_random_c4v_env, but directly for a network (this is also needed for
# the PEPS-PEPO-PEPS network which PEPSKit.initialize_random_c4v_env does not accept)
function bpeps_initial_environment(::C4vCTMRG, network, Venv)
    T = scalartype(network)
    # Vstate is space of z directional leg
    Vstate = only(PEPSKit._north_edge_physical_spaces(network))

    # Define a random real diag matrix for the corner (C4v imposes this)
    corner₀ = DiagonalTensorMap(randn(real(T), Venv ← Venv))
    # Define a radom Hermitian edge tensor (C4v imposes this)
    edge₀ = PEPSKit._project_hermitian(randn(T, Venv ⊗ Vstate ← Venv))

    # Build and return the C4v environment from these 2 tensors
    return CTMRGEnv(corner₀, edge₀)
end


# ----------------------------------------------------------------------------------------------------------------------
# 3. Timers inside a function that Zygote differentiates.
#    `@timeit` uses try/finally, which Zygote cannot differentiate. The two calls below start and stop a timer of `to` and are hidden from Zygote with
#    ignore_derivatives, so they can be used inside the cost function. Use them in pairs: tok = _bpeps_tic("name"); ...; _bpeps_toc(tok)
#    (they use the internals push!/pop!/do_accumulate! of TimerOutputs v0.5, the same ones that the macro uses)
# ----------------------------------------------------------------------------------------------------------------------

# Start a timer with a given name
function _bpeps_tic(label::String)
    # Runs the block with AD switched off
    return PEPSKit.ignore_derivatives() do
        # if the timer is disbaled, return nothing
        to.enabled || return nothing

        # Make or find a child timer called label under the current timer
        # put it on the timer stack and return it as acc (nested timer)
        acc = Base.push!(to, label)
        
        # Return a token (acc, start time in ns, allocated bytes at the start, label)
        (acc, time_ns(), TimerOutputs.gc_bytes(), label)
    end
end

# Stops the timer of a given token
function _bpeps_toc(tok)
    # If timer was disabled 
    tok === nothing && return nothing

    # Runs the block with AD switched off
    PEPSKit.ignore_derivatives() do
        # Unpack the token
        acc, t0, b0, label = tok
        # add the elapsed time and the allocated bytes to the timer and raise its call count
        TimerOutputs.do_accumulate!(acc, t0, b0)
        # Remove the timer from the stack, later timers are no longer nested under it
        Base.pop!(to)
        # Store the max allocation seen under the label
        record_max_alloc!(label, TimerOutputs.gc_bytes() - b0)
        nothing
    end
    return nothing
end


# ----------------------------------------------------------------------------------------------------------------------
# 4. The cost function for OptimKit.optimize: x = (peps, env2, env3)  ->  (cost, gradient w.r.t. peps)
#    cost = -log|λ3/λ2| with λ = network_value(network, env) (the Baxter/CTM formula "site x corners / edges", per unit cell).
#
#    Two ways to take the gradient (keyword differentiate_environment):
#     Math: dE/dψ = ∂E/∂ψ + (∂E/∂env)·(∂env/∂ψ)
#    * false (default): the two environments are converged with leading_boundary OUTSIDE of the automatic differentiation, and only the explicit
#      dependence of the two network values on psi is differentiated (environments fixed). The omitted term is dE/denv * denv/dpsi. It is NOT
#      negligible: this is an approximate gradient; cheap, and the automatic differentiation of it is correct
#      (test/bpeps.jl), but LBFGS minimises a cost whose exact gradient it does not get. development/checks/check_bpeps_gradient.jl prints these numbers.
#    * true: the network values are evaluated with the output of hook_pullback, so that the
#      gradient also contains the fixed-point (geometric sum / Arnoldi) term through the CTMRG. Much more expensive, and the first call compiles
#      the pullback of a CTMRG iteration (Zygote: minutes on a normal machine). Not tested yet!
# ----------------------------------------------------------------------------------------------------------------------

# Builds and returns the cost function E for a fixed PEPO T
# Given a symmetry, a CTMRG algorithm, fixed point solver, 
# the boolean that decides which gradient calculation to use
function generate_bpeps_cost_function(
        T::InfinitePEPO;
        symmetrization = nothing,
        boundary_alg = SimultaneousCTMRG(),
        gradient_alg = FixedPointGradient(),
        differentiate_environment::Bool = false,
    )
    # -log|λ3/λ2| for given networks and environments
    function cost_from_values(n3, env3, n2, env2)
        # n3 is the triple layer network ⟨ψ|O|ψ⟩, infinite 2D square lattice of identical site tensors
        # in plane legs are thick:  χ_peps · D · χ_peps
        # n3 is exact
        # env3 is the set of corner and edge tensors for n3
        # env3 is approximation of n3, bond dimension = χ_env

        # Start a timer 
        tok = _bpeps_tic("bPEPS: network_value and cost")
        # Contract one unit cell with its environment
        λ3 = network_value(n3, env3)
        λ2 = network_value(n2, env2)
        # Calculate the cost function
        cost = -log(abs(λ3 / λ2))
        # Stop timer
        _bpeps_toc(tok)
        return cost
    end

    # (a) environments fixed: converge them first (no automatic differentiation), then differentiate only the network values
    function bpeps_costfun_fixed_env((peps, env2, env3))
        # Start timer for (a)
        return @timeit to "bPEPS: cost function (value + gradient)" begin
            # Calculate: n2 = ⟨ψ|ψ⟩ and n3 = ⟨ψ|T|ψ⟩ (both from current peps)
            n2 = InfiniteSquareNetwork(peps)
            n3 = InfiniteSquareNetwork(peps, T)

            # Perform CTMRG algorithm untill convergence, starting from the previously stored env (warm start)
            env2′, = @timeit to "bPEPS: n2 leading_boundary (CTMRG of <psi|psi>)" leading_boundary(env2, n2, boundary_alg)
            env3′, = @timeit to "bPEPS: n3 leading_boundary (CTMRG of <psi|O|psi>)" leading_boundary(env3, n3, boundary_alg)

            # Copy the converged environments into the old objects in place
            PEPSKit.update!(env2, env2′)    # keep the converged environments to restart the CTMRG from them at the next call
            PEPSKit.update!(env3, env3′)

            # Zygote.pullback(f, peps) runs f once, returns the cost E and a function back that relembers how E was computed
            # if back is called it runs the chain rule backwards through that computation
            # f = Cost function where we give as input: both networks rebuild inside it (Zygote can only follow dependencies that are created inside the differentiated function.) 
            # env3 and env2 are captured from outside, so they are constants!
            E, back = @timeit to "bPEPS: forward pass" Zygote.pullback(
                ψ -> cost_from_values(InfiniteSquareNetwork(ψ, T), env3, InfiniteSquareNetwork(ψ), env2), peps,
            )

            # Perform the backward pass with starting value 1
            gs = @timeit to "bPEPS: backward pass (gradient of the network values)" back(one(E))
            # Return gradient of PEPS ψ
            # Each entry of g says how much E changes if that number is changed slightly.
            g = only(gs)

            # Now we can symmetrize this gradient:
            # e.g. C4v: (4 rotations and 4 reflections), symmetrising means averaging over all 8 transformed copies:
            @timeit to "bPEPS: symmetrize gradient" symmetrize!(g, symmetrization)

            # Return the pair that OptimKit expects
            (E, g)
        end
    end

    # (b) This function is the one that gets differentiated, so only _bpeps_tic/_bpeps_toc allowed
    function forward(ψ, env2, env3)
        n2 = InfiniteSquareNetwork(ψ)
        # Start timer
        tok = _bpeps_tic("bPEPS: n2 leading_boundary (CTMRG of <psi|psi>)")
        # Run the CTMRG inside the forward pass
        env2′, = PEPSKit.hook_pullback(leading_boundary, env2, n2, boundary_alg; alg_rrule = gradient_alg)
        _bpeps_toc(tok)

        n3 = InfiniteSquareNetwork(ψ, T)
        tok = _bpeps_tic("bPEPS: n3 leading_boundary (CTMRG of <psi|O|psi>)")
        env3′, = PEPSKit.hook_pullback(leading_boundary, env3, n3, boundary_alg; alg_rrule = gradient_alg)
        _bpeps_toc(tok)

        # update the environments for reuse
        # .ignore_derivatives() makes it hidden from Zygote
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env2, env2′)
            PEPSKit.update!(env3, env3′)
        end

        # Evaluate the cost with ' env, which are connected to ψ through hook_pullback
        # This makes the gradient pass through CTMRG
        # Just like in a), again n2 and n3 are also 
        return cost_from_values(n3, env3′, n2, env2′)       # env2′ and env3′ (not env2, env3), otherwise the pullback through the CTMRG is never used
    end

    # Same structure as a), but this time with forward as the differentiated function
    function bpeps_costfun_differentiated_env((peps, env2, env3))
        return @timeit to "bPEPS: cost function (value + gradient)" begin
            # We use Zygote.pullback instead of withgradient, to be able to use 2 timers:
            # the forward pass (two CTMRG contractions) and the backward pass (fixed-point gradient) get their own timer
            E, back = @timeit to "bPEPS: forward pass" Zygote.pullback(ψ -> forward(ψ, env2, env3), peps)
            gs = @timeit to "bPEPS: backward pass (gradient through CTMRG)" back(one(E))
            g = only(gs)
            @timeit to "bPEPS: symmetrize gradient" symmetrize!(g, symmetrization)
            (E, g)
        end
    end

    # Return the chose function itself
    return differentiate_environment ? bpeps_costfun_differentiated_env : bpeps_costfun_fixed_env
end


# ----------------------------------------------------------------------------------------------------------------------
# 5. Start of the optimization: a random PEPS, improved by a few steps of the power method (approximate_state: apply O and truncate)
# ----------------------------------------------------------------------------------------------------------------------

# O: 2D PEPO tensor O[(DOWN, UP); (N, E, S, W)], T = InfinitePEPO(O), Vpeps = virtual space of the boundary PEPS (dimension χpeps)
function bpeps_initial_state(O, T::InfinitePEPO, Vpeps; maxiter_init::Int = 20)
    # Construct a random ComplexF64 boundary PEPS with 1 physical leg (codomain), 4 virtual legs (N, E, S, W) (domain)
    # all legs have space Vpeps
    ψ = initializePEPS(T, Vpeps)[1, 1]

    # Now define the truncation algorithm, we use local truncation, with limit = dim(Vpeps) = chi_peps
    trunc_alg = NoEnvTruncation(truncrank(dim(Vpeps)))

    # Perform the poweriteration
    for _ in 1:maxiter_init
        ψ, = approximate_state((ψ, O), trunc_alg)
        # normalizing: otherwise the norm of the power-method iterate grows or decays exponentially
        ψ /= norm(ψ)
    end

    # Wrap the bare tensor into a 1x1 InfinitePEPS structure
    return InfinitePEPS(ψ)
end


# ----------------------------------------------------------------------------------------------------------------------
# 6. The 2D PEPO tensor (used by bpeps_optimize and bpeps_observables in bpeps_main.jl)
# ----------------------------------------------------------------------------------------------------------------------

# This turns the 3D PEPO tensor (two physical legs, six virtual) into a 2D tensor 
# (where original physical elgs get traced with obs in middle and where UP and DOWN become physical legs) that can be stacked on ψ.
function bpeps_pepo_tensor(PEPO::AbstractTensorMap{A,S,2,6}, obs = nothing) where {A,S}
    # 6 virtual legs of PEPO are in order: U, N, E, D, S, W in domain

    if obs === nothing
        # Trace physical legs and place (DOWN, UP) in the 'physical legs' space
        @tensor O[-1 -2; -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -2 -3 -4 -1 -5 -6]

        # O has to be made Complex, for matching with initializePEPS
        return scalartype(O) <: Complex ? O : complex(O)
    end

    # If obs:
    @tensor M[-1 -2; -3 -4 -5 -6] := twist(PEPO, 2)[1 2; -2 -3 -4 -1 -5 -6] * obs[2; 1]
    return scalartype(M) <: Complex ? M : complex(M)
end
