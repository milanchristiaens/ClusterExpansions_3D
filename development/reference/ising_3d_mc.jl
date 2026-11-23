# Monte Carlo reference for the classical 3D Ising model (simple cubic L×L×L, periodic, J = 1, no field) with the Wolff cluster algorithm.
# For every L and β we measure ⟨|m|⟩ (with a binning error), sqrt⟨m²⟩ and the energy per site, and append them to OUT_CSV.
using Random, Printf, Statistics
include(joinpath(@__DIR__, "ising_3d_reference.jl"))

const LS     = parse.(Int, split(get(ENV, "MC_LS", "8,16,32"), r"[,_ ]+"; keepempty = false))
const BETAS  = parse.(Float64, split(get(ENV, "MC_BETAS", "0.15,0.19,0.21,0.22,0.225,0.23,0.25,0.35"), r"[,_ ]+"; keepempty = false))
const SWEEPS = parse(Int, get(ENV, "MC_SWEEPS", "20000"))    # measurements; one measurement after Wolff clusters with ≈ N spins flipped in total
const THERM  = parse(Int, get(ENV, "MC_THERM", "2000"))      # the same unit, thrown away
const SEED   = parse(Int, get(ENV, "MC_SEED", "1"))
const OUT_CSV = get(ENV, "MC_OUT", joinpath(@__DIR__, "..", "..", "data", "reference", "ising3d_mc.csv"))
const NBINS  = 20

# we flip one Wolff cluster starting from a random site; returns its size
function wolff_cluster!(s::Array{Int8,3}, padd, rng, stack::Vector{CartesianIndex{3}})
    L = size(s, 1)
    i0 = CartesianIndex(rand(rng, 1:L), rand(rng, 1:L), rand(rng, 1:L))
    s0 = s[i0]; s[i0] = -s0
    empty!(stack); push!(stack, i0); n = 1
    while !isempty(stack)
        i = pop!(stack)
        for d in 1:3, δ in (-1, 1)
            j = CartesianIndex(ntuple(k -> k == d ? mod1(i[k] + δ, L) : i[k], 3))
            if s[j] == s0 && rand(rng) < padd
                s[j] = -s0; push!(stack, j); n += 1
            end
        end
    end
    return n
end

function bond_sum(s::Array{Int8,3})
    L = size(s, 1); b = 0
    @inbounds for z in 1:L, y in 1:L, x in 1:L
        v = Int(s[x, y, z])
        b += v * (s[mod1(x + 1, L), y, z] + s[x, mod1(y + 1, L), z] + s[x, y, mod1(z + 1, L)])
    end
    return b
end

function run_mc(L, β; sweeps = SWEEPS, therm = THERM, seed = SEED)
    rng = Xoshiro(hash((seed, L, β)))
    N = L^3; s = ones(Int8, L, L, L); padd = 1 - exp(-2β); stack = CartesianIndex{3}[]
    step() = (flipped = 0; while flipped < N; flipped += wolff_cluster!(s, padd, rng, stack); end)
    for _ in 1:therm; step(); end
    ms = zeros(sweeps); es = zeros(sweeps)
    for k in 1:sweeps
        step()
        ms[k] = abs(sum(Int, s)) / N
        es[k] = -bond_sum(s) / N
    end
    binerr(x) = (b = [mean(x[r]) for r in Iterators.partition(eachindex(x), length(x) ÷ NBINS)][1:NBINS]; std(b) / sqrt(NBINS))
    return (; abs_m = mean(ms), abs_m_err = binerr(ms), rms_m = sqrt(mean(abs2, ms)), energy = mean(es), energy_err = binerr(es))
end

mkpath(dirname(OUT_CSV))
isfile(OUT_CSV) || open(io -> println(io, "L,beta,abs_m,abs_m_err,rms_m,energy,energy_err,sweeps,seconds"), OUT_CSV, "w")
println("Wolff MC 3D Ising: L = $LS, β = $BETAS, $SWEEPS measurements after $THERM thermalisation steps, seed $SEED -> $OUT_CSV")
for L in LS, β in BETAS
    t = @elapsed r = run_mc(L, β)
    @printf("L %3d  β %.4f  ⟨|m|⟩ = %.5f ± %.5f  sqrt⟨m²⟩ = %.5f  e = %.5f ± %.5f  (Talapov-Blöte L=∞: %.5f)  %.1f s\n",
            L, β, r.abs_m, r.abs_m_err, r.rms_m, r.energy, r.energy_err, talapov_blote_magnetization(β), t)
    open(io -> println(io, join((L, β, r.abs_m, r.abs_m_err, r.rms_m, r.energy, r.energy_err, SWEEPS, round(t, digits = 2)), ",")), OUT_CSV, "a")
end
