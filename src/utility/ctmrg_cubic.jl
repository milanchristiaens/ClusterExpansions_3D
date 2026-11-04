abstract type CTMRG3D_alg end

"""
1 O
6 A
12 E
8 C
tot = 27
"""

# Positions and leg order of the environment tensors: see the conventions at the top of ctmrg3D.jl.
# corners: all arrows outgoing
# edges: all arrows ingoing
# As: all arrows outgoing
# tensor O: UP, NORTH, EAST, DOWN, SOUTH, WEST. All arrows ingoing

# This function runs CTMRG sweeps on env, untill the corner spectrum stops changing, or untill maxiter is reached
function MPSKit.leading_boundary(env::CTMRG3D_alg)
    S_old = id(envspace(env))
    ϵs = []
    for i = 1:env.maxiter
        S_new = @timeit to "CTMRG3D: ctmrgstep (one full iteration)" ctmrgstep(env)
        push!(ϵs, @timeit to "CTMRG3D: convergence check (norm)" norm(S_new - S_old))
        if env.verbosity > 0
            @info "CTMRG3D step $i, ϵ = $(ϵs[end])"
        end
        if (ϵs[end] < env.tol) && (i > env.miniter)
            if env.verbosity > 1
                @info "CTMRG3D converged after $i iterations with ϵ = $(ϵs[end])"
            end
            return env, ϵs
        end
        S_old = copy(S_new)
    end
    if env.verbosity > 0
        @warn "CTMRG3D not converged after $(env.maxiter) iterations. ϵ = $(ϵs[end])"
    end
    return env, ϵs
end

# This is the usual entry point, given the 3D PEPO O tensor and the obs tensor, it builds the 3D CTMRG environment
# Converges it and returns one value per requested observabke
# ram_gib (= nothing) tells how much RAM we have to do this contraction (slicing)
# onesite_kwargs can have (kmax, allocator, margin, force, rules) info which is used in contract_onesite_dispatch() 
function MPSKit.expectation_value(PEPO::AbstractTensorMap{A,S,2,6}, obss, bpepsspace, envspace; ctm_alg = CTMRG3D, miniter = 5, maxiter = 150, tol = 1e-8, verbosity = 0, west_updates = :auto, west_tol = 0.01 * tol, ram_gib = nothing, onesite_kwargs = (;)) where {A,S}
    O = @timeit to "EV: build O (trace physical legs and flip)" begin
        @tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]
        O = flip(permute(O, ((),(1,2,3,4,5,6))), [4 5 6])
        O
    end

    trunc_bpeps = truncrank(dim(bpepsspace))
    trunc_env = truncrank(dim(envspace))
    env = @timeit to "EV: CTMRG3D init (27 tensors)" ctm_alg(O; trunc_env, trunc_bpeps, Vbp = bpepsspace, Vv = envspace, miniter, maxiter, tol, verbosity, west_updates, west_tol)
    env, ϵs = @timeit to "EV: leading_boundary (CTMRG3D iterations)" leading_boundary(env);
    expvals = []
    for obs = obss
        if obs == :spectrum
            TM = transfer_matrix(env)
            x0 = rand(scalartype(TM), domain(TM), ℂ^1)
            eigvals, eigvecs = eigsolve(x -> TM * x, x0, 5)
            m = marek_gap(eigvals)
        elseif obs == :universal
            m = calculate_universal_3D_spectrum(env)
        else
            M = @timeit to "EV: build M (insert observable)" begin
                @tensor M[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 2; -1 -2 -3 -4 -5 -6] * obs[2; 1]
                M = flip(permute(M, ((),(1,2,3,4,5,6))), [4 5 6])
                M
            end
            # Now we contract:
            # We go to file contract_onesite_auto.jl (to find best cuts, given budget), 
            # then we go to contract_onesite_sliced.jl (for performing the sliced contraction)
            # Then we call the current contraction kernel (which is stored in contract_onesite_orders.jl, under name TREE_CURRENT)
            num = @timeit to "EV: contract_onesite numerator (M)" contract_onesite_dispatch(env, M, ram_gib; verbosity, onesite_kwargs...)
            den = @timeit to "EV: contract_onesite denominator (O)" contract_onesite_dispatch(env, O, ram_gib; verbosity = 0, onesite_kwargs...)
            m = num / den
        end
        push!(expvals, m)
    end
    return expvals
end
