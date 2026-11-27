# Quantum Monte Carlo reference for the 3D transverse-field Ising model H = -J Σ σᶻσᶻ - g Σ σˣ (simple cubic L×L×L, periodic, J = 1) at inverse temperature β.
# We use the continuous-imaginary-time Swendsen-Wang cluster algorithm: every world line is cut into segments (at its kinks and at new cuts with rate g),
# neighbouring segments with equal spins are joined with rate 2J, and every cluster is flipped with probability 1/2. At g = 0 this is the classical SW algorithm.
using Random, Printf, Statistics

const G      = parse(Float64, get(ENV, "QMC_G", "2.0"))
const LS     = parse.(Int, split(get(ENV, "QMC_LS", "8,12,16"), r"[,_ ]+"; keepempty = false))
const BETAS  = parse.(Float64, split(get(ENV, "QMC_BETAS", "0.22,0.23,0.24"), r"[,_ ]+"; keepempty = false))
const SWEEPS = parse(Int, get(ENV, "QMC_SWEEPS", "20000"))     # measured sweeps (one sweep = one Swendsen-Wang update of all segments)
const THERM  = parse(Int, get(ENV, "QMC_THERM", "2000"))       # sweeps thrown away
const SEED   = parse(Int, get(ENV, "QMC_SEED", "1"))
const OUT_CSV = get(ENV, "QMC_OUT", joinpath(@__DIR__, "..", "..", "data", "reference", "tfim3d_qmc.csv"))
const NBINS  = 20
const J = 1.0

# The world lines: for site i the kink times kinks[i] (sorted, in [0, β)) and the spin s0[i] just after τ = 0
mutable struct Config
    L::Int
    β::Float64
    s0::Vector{Int8}
    kinks::Vector{Vector{Float64}}
end
Config(L, β) = Config(L, β, ones(Int8, L^3), [Float64[] for _ in 1:L^3])

site(x, y, z, L) = 1 + mod(x, L) + L * mod(y, L) + L^2 * mod(z, L)

# union-find with path halving
function findroot!(p, a)
    while p[a] != a
        p[a] = p[p[a]]
        a = p[a]
    end
    return a
end
function union!(p, a, b)
    ra, rb = findroot!(p, a), findroot!(p, b)
    ra != rb && (p[ra] = rb)
    return nothing
end

# Poisson times with this rate in [0, β), sorted
function poisson_times(rng, rate, β)
    t = Float64[]
    rate > 0 || return t
    τ = randexp(rng) / rate
    while τ < β
        push!(t, τ)
        τ += randexp(rng) / rate
    end
    return t
end

# One sweep. Segment k of site i is [cuts[k], cuts[k+1]); the last one wraps around τ = β and is the same as the segment before cuts[1].
# Returns (m, n_kinks): the magnetization per site averaged over imaginary time and the total number of kinks after the update.
function sweep!(c::Config, rng, cuts, offs, spin, parent)
    L, β, N = c.L, c.β, c.L^3
    nseg = 0
    for i in 1:N
        cuts[i] = sort!(vcat(c.kinks[i], poisson_times(rng, G, β)))
        offs[i] = nseg
        nseg += max(length(cuts[i]), 1)
    end
    resize!(spin, nseg); resize!(parent, nseg)
    for i in 1:N
        ct, k, s = cuts[i], c.kinks[i], c.s0[i]
        n = max(length(ct), 1)
        # segment n (wrapping) starts at the last cut and holds τ = 0; the others follow the kinks
        sk = s
        j = 1
        for a in eachindex(ct)
            if j <= length(k) && k[j] == ct[a]
                sk = -sk; j += 1
            end
            spin[offs[i] + a] = sk
        end
        isempty(ct) && (spin[offs[i] + 1] = s)
        for a in 1:n
            parent[offs[i] + a] = offs[i] + a
        end
    end
    # segment of site i at time τ, and its spin
    seg(i, τ) = (a = searchsortedlast(cuts[i], τ); offs[i] + (a == 0 ? max(length(cuts[i]), 1) : a))
    for z in 0:L-1, y in 0:L-1, x in 0:L-1
        i = site(x, y, z, L)
        for j in (site(x + 1, y, z, L), site(x, y + 1, z, L), site(x, y, z + 1, L))
            for τ in poisson_times(rng, 2J, β)
                a, b = seg(i, τ), seg(j, τ)
                spin[a] == spin[b] && union!(parent, a, b)
            end
        end
    end
    # flip every cluster with probability 1/2
    flip = rand(rng, Bool, nseg)
    for a in 1:nseg
        r = findroot!(parent, a)
        flip[r] && (spin[a] = -spin[a])
    end
    # new kinks and spins at τ = 0; magnetization
    msum, nk = 0.0, 0
    for i in 1:N
        ct = cuts[i]
        n = max(length(ct), 1)
        if isempty(ct)
            c.s0[i] = spin[offs[i] + 1]; empty!(c.kinks[i])
            msum += spin[offs[i] + 1] * β
            continue
        end
        c.s0[i] = spin[offs[i] + n]
        newk = Float64[]
        prev = spin[offs[i] + n]
        for a in 1:n
            sa = spin[offs[i] + a]
            sa != prev && push!(newk, ct[a])
            prev = sa
            t1 = a < n ? ct[a + 1] : ct[1] + β
            msum += sa * (t1 - ct[a])
        end
        c.kinks[i] = newk
        nk += length(newk)
    end
    return msum / (N * β), nk
end

function binning_error(x)
    nb = min(NBINS, length(x))
    bl = length(x) ÷ nb
    means = [mean(@view x[(k - 1) * bl + 1:k * bl]) for k in 1:nb]
    return std(means) / sqrt(nb)
end

function run_point(L, β, seed)
    rng = Xoshiro(seed)
    c = Config(L, β)
    N = L^3
    cuts, offs = Vector{Vector{Float64}}(undef, N), zeros(Int, N)
    spin, parent = Int8[], Int[]
    for _ in 1:THERM
        sweep!(c, rng, cuts, offs, spin, parent)
    end
    ms, m2s, m4s, xs = zeros(SWEEPS), zeros(SWEEPS), zeros(SWEEPS), zeros(SWEEPS)
    for k in 1:SWEEPS
        m, nk = sweep!(c, rng, cuts, offs, spin, parent)
        ms[k], m2s[k], m4s[k] = abs(m), m^2, m^4
        xs[k] = nk / (N * β * G)          # ⟨σˣ⟩ = ⟨number of kinks⟩ / (N β g)
    end
    m2, m4 = mean(m2s), mean(m4s)
    return (; mabs = mean(ms), dmabs = binning_error(ms), mrms = sqrt(m2), binder = 1 - m4 / (3 * m2^2), sx = G > 0 ? mean(xs) : NaN,
            dsx = G > 0 ? binning_error(xs) : NaN)
end

function main()
    mkpath(dirname(OUT_CSV))
    isfile(OUT_CSV) || open(io -> println(io, "g,L,beta,sweeps,therm,seed,m_abs,dm_abs,m_rms,binder,sx,dsx,time_s"), OUT_CSV, "w")
    points = [(L, β) for L in LS for β in BETAS]
    results = Vector{Any}(undef, length(points))
    Threads.@threads :dynamic for k in eachindex(points)
        L, β = points[k]
        t = @elapsed r = run_point(L, β, SEED + 1000 * k)
        results[k] = (; L, β, r..., t)
        @printf("g %.3f L %3d β %.4f: |m| %.6f ± %.6f  m_rms %.6f  U %.4f  σx %.6f  (%.0f s)\n", G, L, β, r.mabs, r.dmabs, r.mrms, r.binder, r.sx, t)
        flush(stdout)
    end
    open(OUT_CSV, "a") do io
        for r in results
            println(io, join((G, r.L, r.β, SWEEPS, THERM, SEED, r.mabs, r.dmabs, r.mrms, r.binder, r.sx, r.dsx, round(r.t; digits = 1)), ","))
        end
    end
    return nothing
end

main()
