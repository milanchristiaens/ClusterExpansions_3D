# 3D Ising model: magnetization against β for CE + CTMRG3D and SU + CTMRG3D, together with the critical β from QMC.
# Use this example to see that the cluster expansion (CE) reproduces the simple update (SU) result and the phase transition.
# More output while running: ../verbosity/        Timings of one algorithm: ../profiling/

using TensorKit
using TensorKitTensors
using MPSKit
using PEPSKit
using ClusterExpansions
using Plots
using Random
using LinearAlgebra

Random.seed!(123456789)
setprecision(128)

## Settings (the same names and the same β grid in all 3D examples)
T = Float64
spin_symmetry = Trivial
J = 1.0
g = 0.0                  # 0.0, 2.0 or 4.5 (we only know the critical β for these values, see the end of the file)

Δβ_iter = 0.005          # time step of the evolution
# β values at which we observe. Increasing; every value must be a multiple of Δβ_iter.
# SU evolves a state at τ = β/2 (purification), so there every step in β/2 must be a multiple of Δβ_iter.
β_obs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Bond dimension of the density operator, and the two environment bond dimensions of CTMRG3D (these are quite low)
D_CE = 4
D_SU = 2
chipeps = 4
chienv = 6

# Which observable do you want to calculate?
obs = SpinOperators.σᶻ(T, spin_symmetry)


function evolve_CE(obs, β_obs, Δβ_iter, D, chipeps, chienv; J = 1.0, g = 0.0, spin_symmetry = Trivial)
    if spin_symmetry == Trivial
        pepsspace = ℂ^chipeps
        envspace = ℂ^chienv
    else
        pepsspace = Vect[Z2Irrep](0 => chipeps - div(chipeps,2), 1 => div(chipeps,2))
        envspace = Vect[Z2Irrep](0 => chienv - div(chienv,2), 1 => div(chienv,2))
    end

    # define time-evolution (truncation) algorithm
    trunc_alg = NoEnvTruncation(truncrank(D))

    # define cluster-expansion algorithm. p = 3 is default.
    ce_alg = ising_operators(J, g, 0.0; T = BigFloat, symmetry = "Oh")
    ΔO = evolution_operator_cubic(ce_alg, Δβ_iter; T_conv = Float64);

    # start with density operator at β = 0
    O = evolution_operator_cubic(ce_alg, 0.0; T_conv = Float64);

    βs_all = [0.0; β_obs]    # start point + observation points
    ms = []
    for i in eachindex(β_obs)
        @info "At step $i / $(length(β_obs)), β = $(β_obs[i])"
        # Do the time evolution from the previous β to this one
        Δ = βs_all[i+1] - βs_all[i]
        nsteps = round(Int, Δ / Δβ_iter)
        @assert abs(nsteps - Δ / Δβ_iter) < 1e-10 "β_obs values must be multiples of Δβ_iter"
        for _ = 1:nsteps
            O, = approximate_state((O,ΔO), trunc_alg; rot_invariance = false)
        end
        # Calculate the CTMRG environment and the expectation value
        m, = MPSKit.expectation_value(O, [obs], pepsspace, envspace; miniter=3, maxiter=80, tol=1e-7, verbosity=0)
        push!(ms, m)
        @info "magn is $(ms[end])"
    end
    return β_obs, ms
end

function evolve_SU(obs, β_obs, Δβ_iter, D, chipeps, chienv; J = 1.0, g = 0.0, spin_symmetry = Trivial, T = Float64)
    # Some necessary stuff for SU
    P = SpinOperators.spin_space(spin_symmetry)
    dm = DensityMatrixSpace(T, P)
    unitcell = (2, 2, 2) # Simple Update needs 2x2x2 unit cell.

    twosite_op = rmul!(PEPSKit.σᶻᶻ(T, spin_symmetry), -J) + rmul!(PEPSKit.σˣ(T, spin_symmetry) ⊗ id(P) + id(P) ⊗ PEPSKit.σˣ(T, spin_symmetry), -J*g/6)
    H = ThermalCubicHamiltonian(dm, twosite_op; unitcell)

    if spin_symmetry == Trivial
        pepsspace = ℂ^chipeps
        envspace = ℂ^chienv
        vspace = ℂ^D
    else
        pepsspace = Vect[Z2Irrep](0 => chipeps - div(chipeps,2), 1 => div(chipeps,2))
        envspace = Vect[Z2Irrep](0 => chienv - div(chienv,2), 1 => div(chienv,2))
        vspace = Vect[Z2Irrep](0 => D - div(D,2), 1 => div(D,2))
    end

    # Start with infinite temperature
    peps = infinite_temperature_peps3D(dm, vspace; unitcell, noise=0.0)

    # The purification doubles β: the state at τ is the density operator at β = 2τ, so we observe at τ = β/2
    τs_all = [0.0; β_obs ./ 2]
    ms = []
    for i in eachindex(β_obs)
        @info "At step $i / $(length(β_obs)), β = $(β_obs[i])"
        # Do the SU time evolution from the previous τ to this one
        Δτ = τs_all[i+1] - τs_all[i]
        nsteps = round(Int, Δτ / Δβ_iter)
        @assert abs(nsteps - Δτ / Δβ_iter) < 1e-10 "β_obs/2 steps must be multiples of Δβ_iter"
        peps, wtdiff = simpleupdate3D(
            peps, H, [Δβ_iter];
            tol=-Inf, maxiter=nsteps, verbosity=0, trscheme=FixedSpaceTruncation(),
        )
        # make the purification
        PEPO = real(cubic_pepo_purified(peps, dm)[1, 1, 1])

        # Calculate the CTMRG environment and the expectation value
        m, = MPSKit.expectation_value(PEPO, [obs], pepsspace, envspace; miniter=3, maxiter=80, tol=1e-7, verbosity=0)
        push!(ms, m)
        @info "magn is $(ms[end])"

    end
    return β_obs, ms
end


βs, ms = evolve_CE(obs, β_obs, Δβ_iter, D_CE, chipeps, chienv; J, g, spin_symmetry)
βs_SU, ms_SU = evolve_SU(obs, β_obs, Δβ_iter, D_SU, chipeps, chienv; J, g, spin_symmetry, T)

# The exact results for the critical temperature
βc = if g == 0.0
    0.2216546
elseif g == 2.0
    1 / 4.212488872894219
elseif g == 4.5
    1 / 2.6092116221586945
else
    error("Unknown g value")
end

# Make plot
plt = plot()
scatter!(plt, βs, abs.(ms), label = "CE + CTMRG")
scatter!(plt, βs_SU, abs.(ms_SU), label = "SU + CTMRG")
vline!(plt, [βc], label = "QMC")
xlabel!("β")
ylabel!("Magnetization")
display(plt)
