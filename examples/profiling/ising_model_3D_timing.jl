# Profiling of the 3D Ising simulation of `ising_model_3D.jl`.
#
# Same physics and same default parameters as ising_model_3D.jl (CE + CTMRG3D and SU + CTMRG3D);
# the differences are: (1) `ClusterExpansions.@timeit to "name" expr` timers from TimerOutputs.jl (the timer `to` lives
# in the ClusterExpansions module; more timers are placed inside the package code), (2) a small
# warm-up run so that compilation time is not measured, (3) a timing report, a CSV file and two
# plots at the end.

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using TimerOutputs
using Plots
using Random
using LinearAlgebra

Random.seed!(123456789)

const to = ClusterExpansions.to # the timer that the package code writes to

function evolve_CE_timed(obs, Δβ_iter, Δβ_obs, maxiter_obs, D, chipeps, chienv; J = 1.0, g = 0.0, spin_symmetry = Trivial, ctm_miniter = 3, ctm_maxiter = 80)
    if spin_symmetry == Trivial
        pepsspace = ℂ^chipeps
        envspace = ℂ^chienv
    else
        pepsspace = Vect[Z2Irrep](0 => chipeps - div(chipeps, 2), 1 => div(chipeps, 2))
        envspace = Vect[Z2Irrep](0 => chienv - div(chienv, 2), 1 => div(chienv, 2))
    end

    nsteps = round(Int, Δβ_obs / Δβ_iter)
    @assert abs(nsteps - Δβ_obs / Δβ_iter) < 1e-10 "Δβ_obs and Δβ_iter are incomptatible"

    trunc_alg = NoEnvTruncation(truncrank(D))

    ce_alg = ClusterExpansions.@timeit to "CE: ising_operators (setup)" ising_operators(J, g, 0.0; T = BigFloat, symmetry = nothing)
    ΔO = ClusterExpansions.@timeit to "CE: evolution_operator_cubic for ΔO" ClusterExpansions.evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64)
    O = ClusterExpansions.@timeit to "CE: evolution_operator_cubic for O at beta=0" ClusterExpansions.evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)

    ms = []
    for i = 1:maxiter_obs
        ClusterExpansions.@timeit to "CE: time evolution (nsteps x approximate_state)" begin
            for _ = 1:nsteps
                O, = ClusterExpansions.@timeit to "approximate_state (stack and truncate)" approximate_state((O, ΔO), trunc_alg; rot_invariance = false)
            end
        end
        m, = ClusterExpansions.@timeit to "expectation_value (CTMRG3D and observable)" MPSKit.expectation_value(O, [obs], pepsspace, envspace; miniter = ctm_miniter, maxiter = ctm_maxiter, tol = 1e-7, verbosity = 0)
        push!(ms, m)
    end
    βs = [Δβ_obs * i for i in 1:maxiter_obs]
    return βs, ms
end

function evolve_SU_timed(obs, Δβ_iter, Δβ_obs, maxiter_obs, D, chipeps, chienv; J = 1.0, g = 0.0, spin_symmetry = Trivial, T = Float64, ctm_miniter = 3, ctm_maxiter = 80)
    P = SpinOperators.spin_space(spin_symmetry)
    dm = DensityMatrixSpace(T, P)
    unitcell = (2, 2, 2) # Simple Update needs 2x2x2 unit cell.

    twosite_op = rmul!(PEPSKit.σᶻᶻ(T, spin_symmetry), -J) + rmul!(PEPSKit.σˣ(T, spin_symmetry) ⊗ id(P) + id(P) ⊗ PEPSKit.σˣ(T, spin_symmetry), -J * g / 6)
    H = ClusterExpansions.@timeit to "SU: build ThermalCubicHamiltonian" ThermalCubicHamiltonian(dm, twosite_op; unitcell)

    if spin_symmetry == Trivial
        pepsspace = ℂ^chipeps
        envspace = ℂ^chienv
        vspace = ℂ^D
    else
        pepsspace = Vect[Z2Irrep](0 => chipeps - div(chipeps, 2), 1 => div(chipeps, 2))
        envspace = Vect[Z2Irrep](0 => chienv - div(chienv, 2), 1 => div(chienv, 2))
        vspace = Vect[Z2Irrep](0 => D - div(D, 2), 1 => div(D, 2))
    end

    nsteps = round(Int, Δβ_obs / Δβ_iter)
    @assert abs(nsteps - Δβ_obs / Δβ_iter) < 1e-10 "Δβ_obs and Δβ_iter are incomptatible"

    peps = ClusterExpansions.@timeit to "SU: infinite_temperature_peps3D" infinite_temperature_peps3D(dm, vspace; unitcell, noise = 0.0)

    ms = []
    for i = 1:maxiter_obs
        peps, wtdiff = ClusterExpansions.@timeit to "SU: simpleupdate3D (nsteps sweeps)" simpleupdate3D(
            peps, H, [Δβ_iter];
            tol = -Inf, maxiter = nsteps, verbosity = 0, trscheme = FixedSpaceTruncation(),
        )
        PEPO = ClusterExpansions.@timeit to "SU: cubic_pepo_purified" real(cubic_pepo_purified(peps, dm)[1, 1, 1])
        m, = ClusterExpansions.@timeit to "expectation_value (CTMRG3D and observable)" MPSKit.expectation_value(PEPO, [obs], pepsspace, envspace; miniter = ctm_miniter, maxiter = ctm_maxiter, tol = 1e-7, verbosity = 0)
        push!(ms, m)
    end
    βs = [2 * Δβ_obs * i for i in 1:maxiter_obs] # Factor 2 is due to the purification
    return βs, ms
end

## Reading the TimerOutput back (TimerOutputs only prints it, so we extract our own rows)
# TimerOutputs.todict(to) returns nested dictionaries, one per section:
# Dict("n_calls" => ..., "time_ns" => ..., "allocated_bytes" => ..., "inner_timers" => Dict("child name" => Dict(same keys...), ...))
# Now we want to turn this into rows

# Depth-first rows, siblings sorted by decreasing total time. Times in seconds, memory in MiB.
# `self` is the time not covered by any child section. Uses `TimerOutputs.todict`.
# e.g. 
# TOTAL (10 s)
# ├─ A (6 s)
# │   ├─ A1 (4 s)
# │   └─ A2 (1 s)
# └─ B (3 s)
#     └─ B1 (2 s)
# Then we get: TOTAL, A, A1, A2, B, B1
function timer_rows(to::TimerOutput)
    # Start empty list, each section will become one named tuple in it
    rows = NamedTuple[]
    
    # Small helper function: Calculates the total time (ns) of all direct children of section d, if no children = 0
    inner_ns(d) = sum((c["time_ns"] for c in values(d["inner_timers"])); init = 0)
    
    # Recursive helper that visits all children of d
    function walk(d, prefix, depth, parent_ns)
        # d = studied parent, prefix = path so far, depth is the level (0=top)
        # parent_ns = total parent's time

        # Now we loop over all children, in order of first the longest child time to lowest child time
        for (name, c) in sort(collect(d["inner_timers"]); by = kv -> kv[2]["time_ns"], rev = true)
            # collect() : turns children dictionary into a list of: (name child, child data) pairs
            # name = child name and c is the child's data

            # This creates a string corresponding to the full path: e.g. "TOTAL CE run > expectation_value > ..."
            path = isempty(prefix) ? name : prefix * " > " * name
            
            # We add to rows a named tuple: (path = path, name = name, depth = depth, ...)
            push!(rows, (;
                path, name, depth,
                ncalls = c["n_calls"],          # Number of calls
                total = c["time_ns"] / 1.0e9,                   # total time of child
                self = (c["time_ns"] - inner_ns(c)) / 1.0e9,    # total self time of child
                alloc_MiB = c["allocated_bytes"] / 2^20,        # Allocated memory in MiB (2^20 Bytes)
                pct_parent = parent_ns > 0 ? 100 * c["time_ns"] / parent_ns : NaN, # Share of parent's time in %
                max_alloc_MiB = get(ClusterExpansions.MAXALLOC, name, NaN) / 2^20, # Largest allocation of ONE call of a section with this name (MiB)
            ))

            # Make it resursive, we visit this section's own children, one level deeper, with this section as new parent
            # With the updated path/prefix
            walk(c, path, depth + 1, c["time_ns"])
        end
    end

    # Now we convert the whole time to dictionaries, start the walk at root with ampty prefix and depth 0
    top = TimerOutputs.todict(to)
    # We have to pass a parent_ns, but top doesn't really have a parent, so it shouldn't have self time
    # So we can simply use time of children inner_ns(top)
    walk(top, "", 0, inner_ns(top))
    return rows
end

# Compact overview of the first levels of the tree (print_timer itself has no depth limit)
function print_overview(rows; maxdepth = 2)
    # Print a header
    println(rpad("section", 70), lpad("calls", 8), lpad("total [s]", 11), lpad("self [s]", 10), lpad("% parent", 9), lpad("alloc [MiB]", 13))
    println("-"^121)

    # Loop over all rows that are not deeper than maxdepth
    for r in rows
        r.depth > maxdepth && continue

        # Print one row, indented via ("  "^r.depth)
        println(
            rpad("  "^r.depth * r.name, 70), lpad(r.ncalls, 8), lpad(round(r.total; digits = 3), 11),
            lpad(round(r.self; digits = 3), 10), lpad(round(r.pct_parent; digits = 1), 9),
            lpad(round(r.alloc_MiB; digits = 1), 13)
        )
    end
    return nothing
end


function write_timer_csv(path, rows)
    # Open file for writing at path
    open(path, "w") do io
        # Header line
        println(io, "path,depth,ncalls,total_s,self_s,alloc_MiB,pct_parent,max_alloc_MiB")        # Print one line per row, sometimes we use " ", because section names can contain commas or >
        # Comma's in between
        for r in rows
            println(io, '"', r.path, '"', ',', r.depth, ',', r.ncalls, ',', r.total, ',', r.self, ',', r.alloc_MiB, ',', r.pct_parent, ',', r.max_alloc_MiB)        
        end
    end
    return path
end

# include() runs another Julia file as if its code were pasted here
# It just defines the plot functions in the session
include(joinpath(@__DIR__, "timing_plots.jl"))

## Parameters (same as in ising_model_3D.jl)
T = Float64
spin_symmetry = Trivial
J = 1.0
g = 0.0

D_CE = 4
D_SU = 2
chipeps = 4
chienv = 6

obs = SpinOperators.σᶻ(T, spin_symmetry)

Δβ_iter = 0.005
Δβ_obs = 0.01
maxiter_obs = 30 # CE uses maxiter_obs, SU uses  maxiter_obs/ 2, so make sure it is even number

# Return as info the number of Julia threads (how many tasks Julia code can run in parallel in this session (default=1)
# And the number of threads that BLAS (LinearAlgebra, SVD, QR, ...) uses, it has its own thread pool, seperate from Julia's threads (default>1)
# Numbers from another run, another computer, or a cluster are only comparable if the thread counts are written next to them.
@info "Threads.nthreads() = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())"


## Warm-up: compile everything with tiny settings (not measured)
@info "Warm-up run (compilation), not timed"
evolve_CE_timed(obs, Δβ_iter, Δβ_iter, 1, D_CE, chipeps, chienv; J, g, spin_symmetry, ctm_miniter = 1, ctm_maxiter = 1)
evolve_SU_timed(obs, Δβ_iter, Δβ_iter, 1, D_SU, chipeps, chienv; J, g, spin_symmetry, ctm_miniter = 1, ctm_maxiter = 1)


## Measured run: CE + CTMRG3D
reset_timer!(to)
ClusterExpansions.reset_max_alloc!()
βs, ms = ClusterExpansions.@timeit to "TOTAL CE run" evolve_CE_timed(obs, Δβ_iter, Δβ_obs, maxiter_obs, D_CE, chipeps, chienv; J, g, spin_symmetry)
rows_CE = timer_rows(to)
println("\n===== CE + CTMRG3D: overview (depth <= 2) =====")
print_overview(rows_CE; maxdepth = 2)
println("\n===== CE + CTMRG3D: full tree (TimerOutputs) =====")
print_timer(to; sortby = :time)
write_timer_csv("timing_CE.csv", rows_CE)
@info "Peak memory of the Julia process so far (includes memory outside the GC): $(round(Sys.maxrss() / 2^20; digits = 1)) MiB"
plt_CE_phases = plot_phases(rows_CE, "CE + CTMRG3D")
plt_CE_self = plot_self_time(rows_CE, "CE + CTMRG3D")
plt_CE_ctm = plot_self_time(rows_CE, "CE + CTMRG3D"; under = "leading_boundary", top = 25)
display(plt_CE_phases)
display(plt_CE_self)
display(plt_CE_ctm)
savefig(plt_CE_phases, "timing_CE_phases.png")
savefig(plt_CE_self, "timing_CE_self.png")
savefig(plt_CE_ctm, "timing_CE_ctmrg.png")


## Measured run: SU + CTMRG3D
reset_timer!(to)
ClusterExpansions.reset_max_alloc!()
βs_SU, ms_SU = ClusterExpansions.@timeit to "TOTAL SU run" evolve_SU_timed(obs, Δβ_iter, Δβ_obs, maxiter_obs, D_SU, chipeps, chienv; J, g, spin_symmetry)
rows_SU = timer_rows(to)
println("\n===== SU + CTMRG3D: overview (depth <= 2) =====")
print_overview(rows_SU; maxdepth = 2)
println("\n===== SU + CTMRG3D: full tree (TimerOutputs) =====")
print_timer(to; sortby = :time)
write_timer_csv("timing_SU.csv", rows_SU)
@info "Peak memory of the Julia process so far (includes memory outside the GC): $(round(Sys.maxrss() / 2^20; digits = 1)) MiB"
plt_SU_phases = plot_phases(rows_SU, "SU + CTMRG3D")
plt_SU_self = plot_self_time(rows_SU, "SU + CTMRG3D")
plt_SU_ctm = plot_self_time(rows_SU, "SU + CTMRG3D"; under = "leading_boundary", top = 25)
display(plt_SU_phases)
display(plt_SU_self)
display(plt_SU_ctm)
savefig(plt_SU_phases, "timing_SU_phases.png")
savefig(plt_SU_self, "timing_SU_self.png")
savefig(plt_SU_ctm, "timing_SU_ctmrg.png")
