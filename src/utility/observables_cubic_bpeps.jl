# ======================================================================================================================
# observables_cubic_bpeps.jl -- 3D observables with a VARIATIONAL BOUNDARY PEPS ("bPEPS", "vbpeps")
#
# Idea: the 3D PEPO (from the cluster expansion) is a transfer operator in the z-direction. Its dominant "eigenvector" is a 2D PEPS (the boundary PEPS
# psi, bond dimension χpeps) living on a plane. We do not find it with a power method but VARIATIONALLY: psi maximises the overlap per site
#       λ3 / λ2 = <psi| O |psi> / <psi|psi>          (cost function  -log|λ3/λ2|, minimised with LBFGS)
# where both 2D networks (λ2: PEPS-PEPS, λ3: PEPS-PEPO-PEPS) are contracted with a 2D CTMRG of PEPSKit (bond dimension χenv) and the gradient is
# found with a fixed-point (Arnoldi) gradient through the CTMRG. Afterwards the observables are 2D CTMRG contractions with one PEPO tensor replaced.
#
# MAIN ENTRY:  bpeps_expectation_value(PEPO, obss, bpepsspace, envspace; ...)   -- the analogue of MPSKit.expectation_value (CTMRG3D) in ctmrg_cubic.jl
#              (it calls bpeps_optimize and then bpeps_observables, you can also call these two yourself, e.g. to keep the state for a next beta)
#
# Names: the functions have the prefix bpeps_ so that they do not clash with the retract/transport functions of PEPSKit (peps_retract, peps_transport!)
#
# Leg convention of the 2D PEPO O (PEPSKit): O[(DOWN, UP); (N, E, S, W)] = the 3D PEPO with its two physical legs traced (twist(PEPO, 2) for fermions);
# DOWN and UP of the 3D PEPO are the "physical" legs of the 2D problem.
# Timers: nested timers "bPEPS: ..." write to the timer `to` of ClusterExpansions (see examples/profiling/ising_model_3D_timing_vbpeps.jl).
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
#      (test/bpeps.jl), but LBFGS minimises a cost whose exact gradient it does not get. development/check_bpeps_gradient.jl prints these numbers.
#    * true: the PEPSKit pattern (as in PEPSKit.fixedpoint): the network values are evaluated with the output of hook_pullback, so that the
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

    # (b) This function isthe one that gets differentiated, so only _bpeps_tic/_bpeps_toc allowed
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
# 6. Optimization and observables
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

# Find the boundary PEPS of the 3D PEPO.
# Returns (; psi, env2, env3, cost, numfg, normgradhistory); psi is an InfinitePEPS, env3 is the CTMRG environment of <psi|O|psi>.
#   bpepsspace, envspace: virtual space of the boundary PEPS (χpeps) and of the CTMRG environment (χenv)
#   boundary_alg_type: SimultaneousCTMRG (default, no symmetry needed) or C4vCTMRG (with symmetrization = RotateReflect(); needs a C4v symmetric PEPO)
#   state0: (psi, env2, env3) of an earlier call (warm start, e.g. the neighbouring β), instead of the power-method start
#   differentiate_environment: false = gradient with fixed environments (default, see section 4), true = gradient through the CTMRG (hook_pullback; uses gradient_tol, gradient_maxiter, gradient_krylovdim, gradient_eager)
function bpeps_optimize(
        PEPO::AbstractTensorMap{A,S,2,6}, bpepsspace, envspace;
        boundary_alg_type = SimultaneousCTMRG, symmetrization = nothing,
        boundary_tol = 1.0e-8, boundary_maxiter = 400, boundary_verbosity = 0,
        gradient_tol = 1.0e-6, gradient_maxiter = 5, gradient_krylovdim = 20, gradient_eager = true,
        optimizer_memory = 64, optimizer_tol = 1.0e-5, optimizer_maxiter = 75, optimizer_verbosity = 0,
        maxiter_init = 20, state0 = nothing, differentiate_environment = false,
    ) where {A,S}
    # Construct the 2D PEPO tensor
    O = @timeit to "bPEPS: build O (trace physical legs)" bpeps_pepo_tensor(PEPO)
    T = InfinitePEPO(O)

    # Define the different algorithms: gradient_alg only used if differentiate_environment = true
    boundary_alg = boundary_alg_type(; tol = boundary_tol, maxiter = boundary_maxiter, verbosity = boundary_verbosity)
    gradient_alg = FixedPointGradient(;
        solver_alg = KrylovKit.Arnoldi(; tol = gradient_tol, maxiter = gradient_maxiter, krylovdim = gradient_krylovdim, eager = gradient_eager),
    )
    optimizer_alg = LBFGS(optimizer_memory; gradtol = optimizer_tol, maxiter = optimizer_maxiter, verbosity = optimizer_verbosity)

    # Construct the Start boundary state
    # If no hot start was given:
    if state0 === nothing
        # Construct the initial boundary state (given O) with the power method 
        psi0 = @timeit to "bPEPS: initial state (power method, approximate_state)" bpeps_initial_state(O, T, bpepsspace; maxiter_init)
        psi0 = @timeit to "bPEPS: symmetrize and normalize initial state" PEPSKit.peps_normalize(symmetrize!(psi0, symmetrization))
        # Also construct initial CTMRG environments (with symmetry if needed): env2_0 for ⟨ψ|ψ⟩. env3_0 for ⟨ψ|O|ψ⟩
        env2_0, env3_0 = @timeit to "bPEPS: initial CTMRG environments" begin
            (bpeps_initial_environment(boundary_alg, InfiniteSquareNetwork(psi0), envspace),
             bpeps_initial_environment(boundary_alg, InfiniteSquareNetwork(psi0, T), envspace))
        end
    # Else use the hot start as initial guess
    else
        psi0, env2_0, env3_0 = state0
    end

    # Construct the finction x ↦ (cost, gradient) (true or false used here)
    costfun = generate_bpeps_cost_function(T; symmetrization, boundary_alg, gradient_alg, differentiate_environment)
    # Return the 2 callbacks that LBFG needs:
    # retract takes a step ψ → ψ + α·d, and it also carries the environments along.
    # finalize! runs after each iteration and re-symmetrises ψ if a symmetry is requested
    retract, finalize! = bpeps_symmetrize_retract_and_finalize!(symmetrization, bpeps_retract)

    # Run the optimisation, the variable to optimise is (psi, env2, env3)
    # This returns the variable, the cost, ...
    (psi, env2, env3), cost, _, numfg, normgradhistory = @timeit to "bPEPS: LBFGS optimization" optimize(
        costfun, (psi0, env2_0, env3_0), optimizer_alg;
        inner = PEPSKit.real_inner, retract, (transport!) = bpeps_transport!, (finalize!) = finalize!,
    )

    # Return the NamedTuple
    return (; psi, env2, env3, cost, numfg, normgradhistory)
end

# This measures ⟨obs⟩ = ⟨ψ|M|ψ⟩ / ⟨ψ|O|ψ⟩ on the converged state, where M is the obs and O is the original CE PEPO O tensor
function bpeps_observables(PEPO::AbstractTensorMap{A,S,2,6}, obss, result) where {A,S}
    # Firstly, rebuild the ⟨ψ|O|ψ⟩ network
    O = bpeps_pepo_tensor(PEPO)
    n3 = InfiniteSquareNetwork(result.psi, InfinitePEPO(O))

    # Calculate: ⟨ψ|O|ψ⟩, using result.env3
    den = @timeit to "bPEPS: observable denominator (O)" PEPSKit._contract_site((1, 1), n3, result.env3)
    
    # Loop over all observables
    expvals = []
    for obs in obss
        M = bpeps_pepo_tensor(PEPO, obs)
        # Calculate ⟨ψ|M|ψ⟩, Contract the psi's with M
        num = @timeit to "bPEPS: observable numerator (M)" PEPSKit.contract_local_tensor((1, 1, 1), M, n3, result.env3)
        push!(expvals, num / den)
    end
    return expvals
end

# This is a one-line wrapper. It calls bpeps_optimize(...; kwargs...) and passes the result straight to bpeps_observables.
function bpeps_expectation_value(PEPO::AbstractTensorMap{A,S,2,6}, obss, bpepsspace, envspace; kwargs...) where {A,S}
    result = bpeps_optimize(PEPO, bpepsspace, envspace; kwargs...)
    return bpeps_observables(PEPO, obss, result)
end
