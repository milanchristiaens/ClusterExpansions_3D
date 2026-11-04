# This file compares the CTMRG3D option `west_updates` (:auto, 1, 2, 4) on the 3D Ising model near βc:
# magnetization, needed CTMRG iterations, number of partial_ctmrgstep calls and timings.
# One extra case: :auto with PEPO bond dimension D = 5.
#
# Output: table on screen + CSV in data/benchmarks/.

using TensorKit, TensorKitTensors, MPSKit, PEPSKit, ClusterExpansions, TimerOutputs, Random, LinearAlgebra, Printf
using ClusterExpansions: contract_onesite_dispatch

Random.seed!(123456789); setprecision(128)

# Defining some common settings
J, g = 1.0, 0.0
chipeps, chienv = 4, 6
pepsspace, envspace = ℂ^chipeps, ℂ^chienv
obs = SpinOperators.σᶻ(Float64, Trivial)
Δβ_iter = 0.005
# Close to phase transition: βc = 0.221 654 626(5)  (Ferrenberg, Xu and Landau (2018)).
β_obs = [0.20, 0.22, 0.24]
ctm_miniter, ctm_maxiter, ctm_tol = 3, 150, 1e-7
ram_gib = 4
# Sometimes it is handy to perform each simulation multiple times to get more statistics:
n_timing_reps = 1        # >1: report the minimum time over repeated runs (magnetization result stays same)

cases = [
    (name = ":auto, D=5", D = 5, west_updates = :auto),
    (name = ":auto, D=4", D = 4, west_updates = :auto),
    (name = "1, D=4",     D = 4, west_updates = 1),
    (name = "2, D=4",     D = 4, west_updates = 2),
    (name = "4, D=4",     D = 4, west_updates = 4),
    (name = "6, D=4",     D = 4, west_updates = 6),
    (name = "10, D=4",     D = 4, west_updates = 10),
]

# Construct the PEPO at each β_obs, once per D
ce_alg = ising_operators(J, g, 0.0; T = BigFloat, symmetry = "Oh", verbosity = 0)
ΔO = evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64)

function evolve_states(D)
    trunc_alg = NoEnvTruncation(truncrank(D); verbosity = 0)
    # From time_evolve_cubic file: we construct O at β=0
    O = evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)
    states = []
    β_prev = 0.0
    for β in β_obs
        nsteps = round(Int, (β - β_prev) / Δβ_iter)
        @assert abs(nsteps - (β - β_prev) / Δβ_iter) < 1e-10 "β_obs values must be multiples of Δβ_iter"
        for _ in 1:nsteps
            O, _ = approximate_state((O, ΔO), trunc_alg; rot_invariance = false)
        end
        push!(states, O)
        β_prev = β
    end
    return states
end

# Now we save the PEPO O tensor in states (one for every D)
states = Dict{Int, Any}()
for D in unique(c.D for c in cases)
    t = @elapsed (states[D] = evolve_states(D))
    @info "PEPO states for D = $D built in $(round(t; digits = 1)) s"
end

# Now we define some helpers (same recipe as MPSKit.expectation_value)
function traced_O(P)
    @tensor O[-1 -2 -3 -4 -5 -6] := twist(P, 2)[1 1; -1 -2 -3 -4 -5 -6]
    return flip(permute(O, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])
end
function traced_M(P, obs)
    @tensor M[-1 -2 -3 -4 -5 -6] := twist(P, 2)[1 2; -1 -2 -3 -4 -5 -6] * obs[2; 1]
    return flip(permute(M, ((), (1, 2, 3, 4, 5, 6))), [4 5 6])
end

# Total number of calls of the timer entry `label`, wherever it is nested in the TimerOutput tree
function ncalls_of(t::TimerOutput, label)
    n = 0

    # For all direct children of t (k = label string, v = the child TimerOutput)
    for (k, v) in t.inner_timers
        # If k is the label, we add the child's call count
        k == label && (n += TimerOutputs.ncalls(v))
        # Recursively go into deeper levels (but k has still to be matching with label)
        n += ncalls_of(v, label)
    end
    return n
end

# P is the PEPO tensor for D=4 or for D=5) (with physical legs still open)
# west_updates is :auto, 1, 2 or 4
function run_case(P, west_updates; maxiter = ctm_maxiter, reps = n_timing_reps)
    # Trace physical legs: (one time with obs in physical legs)
    O = traced_O(P)
    M = traced_M(P, obs)

    m = NaN; iters = 0; conv = false; npart = 0
    t_ctm = Inf; t_con = Inf

    # For all asked reps (only timing differs):
    for _ in 1:reps
        # First time the construction of the fresh env from O
        reset_timer!(ClusterExpansions.to)
        env = CTMRG3D(O; trunc_env = truncrank(chienv), trunc_bpeps = truncrank(chipeps), Vbp = pepsspace, Vv = envspace,
                      miniter = ctm_miniter, maxiter, tol = ctm_tol, verbosity = 0, west_updates)
        t0 = time_ns()

        # Now time the CTMRG iterations/convergence
        # Also store the ϵ's!!
        env, ϵs = MPSKit.leading_boundary(env)
        t1 = (time_ns() - t0) / 1e9
        # Count the number of partial ctmrgsteps that were used (same as total number of indivual West updates)
        npart = ncalls_of(ClusterExpansions.to, "partial_ctmrgstep")
        
        # Time the contraction of env
        t0 = time_ns()
        num = contract_onesite_dispatch(env, M, ram_gib)
        den = contract_onesite_dispatch(env, O, ram_gib)
        t2 = (time_ns() - t0) / 1e9
        # Store the expectatioan value of magnetization
        m = num / den
        
        # The total number of ϵ's is equal to the number of full CTMRG iterations performed
        iters = length(ϵs)
        # Define boolean which is True if last ϵ is below the tolerance, which means it indeed converged in the given max_iter
        conv = ϵs[end] < ctm_tol

        # If n_timing_reps>1, then we only store the lowest t_ctm and the lowest t_con
        t_ctm = min(t_ctm, t1); t_con = min(t_con, t2)
    end
    return (; m, iters, conv, npart, t_ctm, t_con)
end

#  warm-up (compilation), result discarded
run_case(states[cases[1].D][1], cases[1].west_updates; maxiter = 2, reps = 1)

#  main loop
results = []
for (ib, β) in enumerate(β_obs)
    for c in cases
        r = run_case(states[c.D][ib], c.west_updates)
        push!(results, (; β, c.name, c.D, c.west_updates, r...))
        @info "β = $β, $(c.name): m = $(r.m), $(r.iters) iterations, $(r.npart) partial steps"
    end
end

# Printing of the table
println()
for β in β_obs
    rs = filter(r -> r.β == β, results)
    ref = only(filter(r -> r.D == 4 && r.west_updates == 10, rs))      # reference: 10 West updates, D = 4
    println("β = $β   (|Δm| is relative to the 'west_updates = 10, D = 4' run; not defined for D = 5)")
    @printf("%-12s %-20s %-10s %-6s %-9s %-10s %-10s %-10s\n",
            "case", "m", "|Δm|", "iters", "converged", "partial", "t_CTM [s]", "t_contr [s]")
    for r in rs
        dm = r.D == 4 ? @sprintf("%.2e", abs(r.m - ref.m)) : "n/a"
        @printf("%-12s %-20.15f %-10s %-6d %-9s %-10d %-10.2f %-10.2f\n",
                r.name, r.m, dm, r.iters, r.conv ? "yes" : "NO", r.npart, r.t_ctm, r.t_con)
    end
    println()
end

# Saving the CSV
outdir = normpath(joinpath(@__DIR__, "..", "data", "benchmarks"))
mkpath(outdir)
csvpath = joinpath(outdir, "west_updates_comparison.csv")
open(csvpath, "w") do io
    println(io, "beta,case,D,west_updates,m,iterations,converged,partial_steps,t_ctm_s,t_contract_s")
    for r in results
        println(io, join((r.β, r.name, r.D, r.west_updates, r.m, r.iters, r.conv, r.npart, r.t_ctm, r.t_con), ","))
    end
end
@info "wrote $csvpath"
