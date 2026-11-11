# Profiling of the 3D Ising simulation with CE + CTMRG3D (g = 0): where does the time and the memory go?
# The counterpart with a variational boundary PEPS is ising_model_3D_CE_vbPEPS_profiling.jl (this folder).
# Comparing several settings or algorithms: comparing/. Plots from the saved CSV files without rerunning: tools/plot_timing_from_csv.jl
#
# Same physics and settings as ../3D/ising_model_3D_CE_vs_SU.jl (CE + CTMRG3D part; the SU part is not profiled here);
# the differences are: (1) `ClusterExpansions.@timeit to "name" expr` timers from TimerOutputs.jl (the timer `to` lives
# in the ClusterExpansions module; more timers are placed inside the package code), (2) a small
# warm-up run so that compilation time is not measured, (3) a timing report, a CSV file and three
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

## Settings (the same names and the same β grid in all 3D examples)
T = Float64
spin_symmetry = Trivial
J = 1.0
g = 0.0

Δβ_iter = 0.005
# Each value must be a multiple of Δβ_iter.
β_obs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

D_CE = 4
chipeps = 4
chienv = 6

obs = SpinOperators.σᶻ(T, spin_symmetry)


function evolve_CE_timed(obs, β_obs, Δβ_iter, D, chipeps, chienv; J = 1.0, g = 0.0, spin_symmetry = Trivial, ctm_miniter = 3, ctm_maxiter = 80)
    if spin_symmetry == Trivial
        pepsspace = ℂ^chipeps
        envspace = ℂ^chienv
    else
        pepsspace = Vect[Z2Irrep](0 => chipeps - div(chipeps, 2), 1 => div(chipeps, 2))
        envspace = Vect[Z2Irrep](0 => chienv - div(chienv, 2), 1 => div(chienv, 2))
    end

    trunc_alg = NoEnvTruncation(truncrank(D))

    ce_alg = ClusterExpansions.@timeit to "CE: ising_operators (setup)" ising_operators(J, g, 0.0; T = Complex{BigFloat}, symmetry = nothing)
    ΔO = ClusterExpansions.@timeit to "CE: evolution_operator_cubic for ΔO" ClusterExpansions.evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64)
    O = ClusterExpansions.@timeit to "CE: evolution_operator_cubic for O at beta=0" ClusterExpansions.evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64)

    βs_all = [0.0; β_obs]    # start point + observation points
    ms = []
    for i in eachindex(β_obs)
        # time steps from the previous β to this one
        Δ = βs_all[i+1] - βs_all[i]
        nsteps = round(Int, Δ / Δβ_iter)
        @assert abs(nsteps - Δ / Δβ_iter) < 1e-10 "β_obs values must be multiples of Δβ_iter"
        ClusterExpansions.@timeit to "CE: time evolution (nsteps x approximate_state)" begin
            for _ = 1:nsteps
                O, = ClusterExpansions.@timeit to "approximate_state (stack and truncate)" approximate_state((O, ΔO), trunc_alg; rot_invariance = false)
            end
        end
        m, = ClusterExpansions.@timeit to "expectation_value (CTMRG3D and observable)" MPSKit.expectation_value(O, [obs], pepsspace, envspace; miniter = ctm_miniter, maxiter = ctm_maxiter, tol = 1e-7, verbosity = 0)
        push!(ms, m)
    end
    return β_obs, ms
end


# include() runs another Julia file as if its code were pasted here
# timer_rows, print_overview and write_timer_csv (read the TimerOutput back, print an
# overview, write the CSV) are in tools/timing_tools.jl
include(joinpath(@__DIR__, "tools", "timing_tools.jl"))

# It just defines the plot functions in the session
include(joinpath(@__DIR__, "tools", "timing_plots.jl"))
outdir = normpath(joinpath(@__DIR__, "..", "..", "data", "profiling"))
mkpath(outdir)

# Return as info the number of Julia threads (how many tasks Julia code can run in parallel in this session (default=1)
# And the number of threads that BLAS (LinearAlgebra, SVD, QR, ...) uses, it has its own thread pool, seperate from Julia's threads (default>1)
# Numbers from another run, another computer, or a cluster are only comparable if the thread counts are written next to them.
@info "Threads.nthreads() = $(Threads.nthreads()), BLAS threads = $(BLAS.get_num_threads())"


## Warm-up: compile everything with tiny settings (not measured)
@info "Warm-up run (compilation), not timed"
evolve_CE_timed(obs, [Δβ_iter], Δβ_iter, D_CE, chipeps, chienv; J, g, spin_symmetry, ctm_miniter = 1, ctm_maxiter = 1)


## Measured run: CE + CTMRG3D
reset_timer!(to)
ClusterExpansions.reset_max_alloc!()
βs, ms = ClusterExpansions.@timeit to "TOTAL CE run" evolve_CE_timed(obs, β_obs, Δβ_iter, D_CE, chipeps, chienv; J, g, spin_symmetry)
rows_CE = timer_rows(to)
println("\n===== CE + CTMRG3D: overview (depth <= 2) =====")
print_overview(rows_CE; maxdepth = 2)
println("\n===== CE + CTMRG3D: full tree (TimerOutputs) =====")
print_timer(to; sortby = :time)
write_timer_csv(joinpath(outdir, "timing_CE.csv"), rows_CE)
@info "Peak memory of the Julia process so far (includes memory outside the GC): $(round(Sys.maxrss() / 2^20; digits = 1)) MiB"
plt_CE_phases = plot_phases(rows_CE, "CE + CTMRG3D")
plt_CE_self = plot_self_time(rows_CE, "CE + CTMRG3D")
plt_CE_ctm = plot_self_time(rows_CE, "CE + CTMRG3D"; under = "leading_boundary", top = 25)
display(plt_CE_phases)
display(plt_CE_self)
display(plt_CE_ctm)
savefig(plt_CE_phases, joinpath(outdir, "timing_CE_phases.png"))
savefig(plt_CE_self, joinpath(outdir, "timing_CE_self.png"))
savefig(plt_CE_ctm, joinpath(outdir, "timing_CE_ctmrg.png"))
