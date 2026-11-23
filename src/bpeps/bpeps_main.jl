# The 3D observables with a variational boundary PEPS (bPEPS): the 3D PEPO is a transfer operator in the z-direction, whose dominant eigenvector is a 2D PEPS psi (bond dimension χpeps) that we find variationally, by minimising -log|<psi|O|psi> / <psi|psi>| with LBFGS (both 2D networks are contracted with a 2D CTMRG of PEPSKit, the gradient is a fixed-point gradient through the CTMRG).
# Entry point: bpeps_expectation_value(PEPO, obss, bpepsspace, envspace; ...), the analogue of MPSKit.expectation_value of CTMRG3D (it calls bpeps_optimize and bpeps_observables); the helper functions are in bpeps_helpers.jl. The 2D PEPO is O[(DOWN, UP); (N, E, S, W)], the 3D PEPO with its two physical legs traced.


# Thrown by the finalize! callback of bpeps_optimize when max_seconds is exceeded; bpeps_optimize catches it and returns the last LBFGS iterate
struct BpepsTimeLimit <: Exception end

# Find the boundary PEPS of the 3D PEPO.
# Returns (; psi, env2, env3, cost, numfg, normgradhistory, timed_out); psi is an InfinitePEPS, env3 is the CTMRG environment of <psi|O|psi>.
#   max_seconds: stop after the first LBFGS iteration that ends later than max_seconds after the start of this call (default Inf = no limit);
#     then the last iterate is returned with timed_out = true, numfg = number of cost-function calls and normgradhistory = nothing
#   bpepsspace, envspace: virtual space of the boundary PEPS (χpeps) and of the CTMRG environment (χenv)
#   boundary_alg_type: SimultaneousCTMRG (default, no symmetry needed) or C4vCTMRG (with symmetrization = RotateReflect(); needs a C4v symmetric PEPO)
#   state0: (psi, env2, env3) of an earlier call (warm start, e.g. the neighbouring β), instead of the power-method start
#   differentiate_environment: false = gradient with fixed environments (default, see section 4 of bpeps_helpers.jl), true = gradient through the CTMRG (hook_pullback; uses gradient_tol, gradient_maxiter, gradient_krylovdim, gradient_eager)
function bpeps_optimize(
        PEPO::AbstractTensorMap{A,S,2,6}, bpepsspace, envspace;
        boundary_alg_type = SimultaneousCTMRG, symmetrization = nothing,
        boundary_tol = 1.0e-8, boundary_maxiter = 400, boundary_verbosity = 0,
        gradient_tol = 1.0e-6, gradient_maxiter = 5, gradient_krylovdim = 20, gradient_eager = true,
        optimizer_memory = 64, optimizer_tol = 1.0e-5, optimizer_maxiter = 75, optimizer_verbosity = 0,
        maxiter_init = 20, state0 = nothing, differentiate_environment = false, max_seconds = Inf,
    ) where {A,S}

    # Start time, for max_seconds
    t_start = time()
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

    # For max_seconds: count the cost-function calls, and remember the last iterate (x, f) that finalize! has seen
    ncalls = Ref(0)
    last_state = Ref{Any}(nothing)
    counted_costfun(x) = (ncalls[] += 1; costfun(x))
    function finalize_limited!(x, f, g, numiter)
        x, f, g = finalize!(x, f, g, numiter)
        last_state[] = (x, f)
        # after a finished iteration (numiter > 0) we stop if the time is up
        numiter > 0 && time() - t_start > max_seconds && throw(BpepsTimeLimit())
        return x, f, g
    end

    # Run the optimisation, the variable to optimise is (psi, env2, env3)
    # This returns the variable, the cost, ...
    local psi, env2, env3, cost, numfg, normgradhistory
    timed_out = false
    try
        (psi, env2, env3), cost, _, numfg, normgradhistory = @timeit to "bPEPS: LBFGS optimization" optimize(
            counted_costfun, (psi0, env2_0, env3_0), optimizer_alg;
            inner = PEPSKit.real_inner, retract, (transport!) = bpeps_transport!, (finalize!) = finalize_limited!,
        )
    catch err
        err isa BpepsTimeLimit || rethrow()
        # time is up: we return the last iterate
        (psi, env2, env3), cost = last_state[]
        numfg = ncalls[]
        normgradhistory = nothing
        timed_out = true
    end

    # Return the NamedTuple
    return (; psi, env2, env3, cost, numfg, normgradhistory, timed_out)
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
