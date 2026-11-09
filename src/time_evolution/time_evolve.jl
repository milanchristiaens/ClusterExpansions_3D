abstract type TimeEvolution end

struct TimeDependentTimeEvolution <: TimeEvolution
    β₀
    Δβ
    maxiter
    verbosity
    f₁
    f₂
end

struct StaticTimeEvolution <: TimeEvolution
    β₀
    βs_helper
    update_list
    verbosity
end

struct GroundStateTimeEvolution <: TimeEvolution
    β₀
    βs_helper
    update_list
    tol_energy
    verbosity
end

function StaticTimeEvolution(β₀, βs_helper, update_list; verbosity = 0)
    return StaticTimeEvolution(β₀, βs_helper, update_list, verbosity)
end

function UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 0)
    return StaticTimeEvolution(β₀, [Δβ], [1 for i = 1:maxiter], verbosity)
end

function UniformGroundStateTimeEvolution(β₀, Δβ, maxiter, tol_energy; verbosity = 0)
    return GroundStateTimeEvolution(β₀, [Δβ], [1 for i = 1:maxiter], tol_energy, verbosity)
end

function TimeDependentTimeEvolution(β₀, Δβ, maxiter; verbosity = 0, f₁ = β -> 1.0, f₂ = β -> 1.0)
    # Create a time evolution where the onesite and twosite operators are given by f₁(β) * onesite and f₂(β) * twosite_op, respectively.
    return TimeDependentTimeEvolution(β₀, Δβ, maxiter, verbosity, f₁, f₂)
end

function infinite_temperature_PEPO(T, pspace, vspace)
    return permute(id(T, pspace ⊗ vspace ⊗ vspace), ((1,4),(5,6,2,3)))
end

function evolution_operator(ce_alg::ClusterExpansion, time_alg::TimeDependentTimeEvolution, β::Number; T_conv = ComplexF64)
    β == 0.0 && return infinite_temperature_PEPO(T_conv, domain(ce_alg.onesite_op)[1], ce_alg.spaces(0))
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, time_alg.Δβ, time_alg.f₂(β) * ce_alg.twosite_op, time_alg.f₁(β) * ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
    O_clust_full = convert(TensorMap, O_clust_full)
    O = zeros(T_conv, codomain(O_clust_full), domain(O_clust_full))
    for (f_full, f_conv) in zip(blocks(O_clust_full), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O
end

function evolution_operator(ce_alg::ClusterExpansion, β::Number; T_conv = ComplexF64)
    β == 0.0 && return infinite_temperature_PEPO(T_conv, domain(ce_alg.onesite_op)[1], ce_alg.spaces(0))
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; nn_term = ce_alg.nn_term, spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops, svd = ce_alg.svd)
    O_clust_full = convert(TensorMap, O_clust_full)
    O = zeros(T_conv, codomain(O_clust_full), domain(O_clust_full))
    for (f_full, f_conv) in zip(blocks(O_clust_full), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O # Don't normalize, otherwise Atsushi will be mad.
end

function evolution_operator(td_alg::GenericTrotterDecomposition, β::Number; T_conv = ComplexF64)
    β == 0.0 && return infinite_temperature_PEPO(T_conv, domain(td_alg.onesite_op)[1], td_alg.spaces(0))
    U_onesite = get_Trotter_onesite(td_alg.onesite_op, td_alg.g, β)
    U_twosite = get_Trotter_twosite(td_alg.twosite_op, td_alg.spaces(1), β)
    @tensor O_Trotter[-1 -2; -3 -4 -5 -6] := U_onesite[-1; 1] * U_twosite[1 2; -3 -4 -5 -6] * U_onesite[2; -2]
    return O_Trotter
end

function evolution_operator(td_alg::TwositeTrotterDecomposition, β::Number; T_conv = ComplexF64)
    β == 0.0 && return infinite_temperature_PEPO(T_conv, domain(td_alg.onesite_op)[1], td_alg.spaces(0))
    O_Trotter = get_Trotter_twosite(td_alg.twosite_op, td_alg.spaces(1), β)
    return O_Trotter
end

function MPSKit.time_evolve(
    ce_alg::Union{ClusterExpansion,TrotterDecomposition},
    time_alg::StaticTimeEvolution,
    trunc_alg::EnvTruncation,
    observable;
    finalize! = nothing,
    A0 = nothing,
    skip_first::Bool = false,
    saving::Bool = true,
    saving_frequency::Int = 1,
    normalizing::Bool = true,
    trunc_alg_start = nothing
)
    if isnothing(trunc_alg_start)
        As = AbstractTensorMap[evolution_operator(ce_alg, β) for β = time_alg.βs_helper]
    else
        As = AbstractTensorMap[approximate_state(evolution_operator(ce_alg, β), trunc_alg_start; rot_invariance = trunc_alg.rot_invariance)[1] for β = time_alg.βs_helper]
    end
    times = copy(time_alg.βs_helper)
    if isnothing(A0)
        A = evolution_operator(ce_alg, time_alg.β₀)
    else
        A = A0
    end
    if skip_first
        obs = nothing
        expvals = []
    else
        obs = observable(A, 0)
        expvals = [obs]
    end
    push!(As, copy(A))
    push!(times, time_alg.β₀)
    errors = []
    for (i,ind) in enumerate(time_alg.update_list)
        if ind <= length(As)
            A, ϵ = approximate_state((A, As[ind]), trunc_alg; rot_invariance = trunc_alg.rot_invariance)
        elseif ind == i
            A, ϵ = approximate_state((A, A), trunc_alg; rot_invariance = trunc_alg.rot_invariance)
        else
            error("Cannot perform time evolution without saving intermediaire steps for this time algorithm")
        end
        if normalizing
            A /= norm(A)
        end
        obs = observable(A, i)
        push!(errors, ϵ)
        push!(times, times[end] + times[ind])
        if saving
            push!(expvals, obs)
            if mod(i, saving_frequency) == 0
                push!(As, copy(A))
            end
        end
        if time_alg.verbosity > 1
            @info "Time evolution step $(i) with β = $(times[end]), obs = $(obs)"
            @info "Bond dimension is now $(dim(domain(A)[1]))"
            if time_alg.verbosity > 2
                @info "Current norm is $(norm(A))"
            end
        end
        if !isnothing(finalize!)
            finalize!(As, expvals, i)
        end
    end
    if saving
        return times[length(time_alg.βs_helper)+1:end], expvals, As[length(time_alg.βs_helper)+1:end], errors
    else
        return times[end], obs, A, errors
    end
end

function PEPSKit.fixedpoint(
    ce_alg::ClusterExpansion,
    time_alg::GroundStateTimeEvolution,
    trunc_alg::EnvTruncation,
    observable;
    finalize! = nothing,
    A0 = nothing,
    skip_first::Bool = false
)
    As = AbstractTensorMap[evolution_operator(ce_alg, β) for β = time_alg.βs_helper]
    times = copy(time_alg.βs_helper)
    if isnothing(A0)
        A = evolution_operator(ce_alg, time_alg.β₀)
    else
        A = A0
    end

    push!(As, copy(A))
    push!(times, time_alg.β₀)
    
    if skip_first
        expvals = []
    else
        expvals = [observable(A)]
    end

    for (i,ind) in enumerate(time_alg.update_list)
        A, _ = approximate_state((A, As[ind]), trunc_alg; rot_invariance = trunc_alg.rot_invariance)
        A /= norm(A)
        obs = observable(A)
        push!(times, times[end] + times[ind])
        push!(expvals, obs)
        push!(As, copy(A))
        if !isnothing(finalize!)
            A = finalize!(As, expvals, i)
        end

        if time_alg.verbosity > 1
            @info "Time evolution step $(i) with β = $(times[end]), obs = $(obs)"
            @info "Bond dimension is now $(dim(domain(A)[1]))"
            if time_alg.verbosity > 2
                @info "Current norm is $(norm(A))"
            end
        end
        if i > 2 && abs(expvals[end][1] - expvals[end-1][1]) < time_alg.tol_energy
            if time_alg.verbosity > 1
                @info "Ground state search converged after $(i) iterations. Energy is $(expvals[end][1])"
            end
            return As[end], expvals[end]
        end
    end
    if time_alg.verbosity > 0
        @warn "Ground state search did not converge after $(length(time_alg.update_list)) iterations. Energy is $(expvals[end][1])"
    end
    return As[end], expvals[end]
end