# ctmrg3D_main.jl -- MAIN file of the 3D CTMRG: the entry point MPSKit.expectation_value (builds O, converges the environment, contracts the observables) and leading_boundary.
# It calls the functions of its helper file ctmrg3D_helpers.jl (struct CTMRG3D, rotations, CTMRG steps, contract_onesite, ...), which is included right after this file.

abstract type CTMRG3D_alg end

"""
1 O
6 A
12 E
8 C
tot = 27
"""

# Positions and leg order of the environment tensors: see the conventions at the top of ctmrg3D_helpers.jl.
# corners: all arrows outgoing
# edges: all arrows ingoing
# As: all arrows outgoing
# tensor O: UP, NORTH, EAST, DOWN, SOUTH, WEST. All arrows ingoing

# This function runs CTMRG sweeps on env, untill the corner spectrum stops changing, or untill maxiter is reached,
# or untill env.max_seconds is used up (checked after every full iteration, so the limit can be overshot by at most one iteration).
# Returns (env, ϵs, timed_out); timed_out = true only if the time limit stopped it (env is then the environment of the last iteration, not converged).
function MPSKit.leading_boundary(env::CTMRG3D_alg)
    t_start = time()
    S_old = id(envspace(env))
    ϵs = []
    for i = 1:env.maxiter
        S_new = @timeit to "CTMRG3D: ctmrgstep (one full iteration)" ctmrgstep(env)
        push!(ϵs, @timeit to "CTMRG3D: convergence check (norm)" spectrum_change(S_new, S_old))
        if env.verbosity > 0
            @info "CTMRG3D step $i, ϵ = $(ϵs[end])"
        end
        if (ϵs[end] < env.tol) && (i > env.miniter)
            if env.verbosity > 1
                @info "CTMRG3D converged after $i iterations with ϵ = $(ϵs[end])"
            end
            return env, ϵs, false
        end
        if time() - t_start > env.max_seconds
            if env.verbosity > 0
                @warn "CTMRG3D stopped by the time limit ($(env.max_seconds) s) after $i iterations, not converged. ϵ = $(ϵs[end])"
            end
            return env, ϵs, true

        end
        S_old = copy(S_new)
    end
    if env.verbosity > 0
        @warn "CTMRG3D not converged after $(env.maxiter) iterations. ϵ = $(ϵs[end])"
    end
    return env, ϵs, false
end

# This is the usual entry point, given the 3D PEPO O tensor and the obs tensor, it builds the 3D CTMRG environment
# Converges it and returns one value per requested observabke
# ram_gib (= nothing) tells how much RAM we have to do this contraction (slicing)
# onesite_kwargs can have (kmax, allocator, margin, force, rules, cuts, tree, plans) info which is used in contract_onesite_dispatch()
# projectors: :svd or :oblique; projector_exponent: P of the :svd projectors U * S^P (1/2 = original)
# class_projectors: false (original) = one projector per new bond; true = one projector per class of translation-equivalent bonds (gauge fix,
#       see get_projectors_svd_class / get_projectors_oblique_class in ctmrg3D_helpers.jl), for :svd and :oblique
# env0: nothing = cold start (init = :ones, :random or :nishino_okunishi, seed init_seed);
#       an earlier converged CTMRG3D (e.g. of the previous β) = warm start from its C, E, A
# max_seconds: time limit of the CTMRG3D iterations (leading_boundary) for a given β, Inf = none. The contractions of the observables come on top of it.
# Contractions of observables are counted towards the case budget though!
# return_env = true (used in compare scripts, also for cold starts): returns (expvals, env, ϵs, timed_out) instead of expvals, so that env can be the env0 of the next call
# (timed_out = true if max_seconds stopped the iterations)
function MPSKit.expectation_value(PEPO::AbstractTensorMap{A,S,2,6}, obss, bpepsspace, envspace; ctm_alg = CTMRG3D, miniter = 5, maxiter = 150, tol = 1e-8, 
                                 verbosity = 0, west_updates = :auto, west_tol = 0.01 * tol, projectors = :svd, projector_exponent = 0.5, class_projectors = false, env0 = nothing, init = :ones, init_seed = 1, 
                                 max_seconds = Inf, return_env = false, ram_gib = nothing, onesite_kwargs = (;)) where {A,S}
    O = @timeit to "EV: build O (trace physical legs and flip)" begin
        @tensor O[-1 -2 -3 -4 -5 -6] := twist(PEPO, 2)[1 1; -1 -2 -3 -4 -5 -6]
        O = flip(permute(O, ((),(1,2,3,4,5,6))), [4 5 6])
        O
    end

    trunc_bpeps = truncrank(dim(bpepsspace))
    trunc_env = truncrank(dim(envspace))
    env = @timeit to "EV: CTMRG3D init (27 tensors)" ctm_alg(O; trunc_env, trunc_bpeps, Vbp = bpepsspace, Vv = envspace, miniter, maxiter, tol, verbosity, west_updates, west_tol, projectors, projector_exponent, class_projectors, env0, init, init_seed, max_seconds)
    env, ϵs, timed_out = @timeit to "EV: leading_boundary (CTMRG3D iterations)" leading_boundary(env);
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
            # If ram_gib = nothing: contract_onesite(env, M), the unsliced TREE_CURRENT kernel
            # Otherwise we go to contract_onesite_auto (file contract_onesite_slicing.jl), which finds a contraction tree and the best cuts, given the budget
            # (brute force on TREE_CURRENT, or a ready-made plan of onesite_plans.csv), and performs the sliced contraction
            num = @timeit to "EV: contract_onesite numerator (M)" contract_onesite_dispatch(env, M, ram_gib; verbosity, onesite_kwargs...)
            den = @timeit to "EV: contract_onesite denominator (O)" contract_onesite_dispatch(env, O, ram_gib; verbosity = 0, onesite_kwargs...)
            m = num / den
        end
        push!(expvals, m)
    end
    return return_env ? (expvals, env, ϵs, timed_out) : expvals
end
