# This is the test file for checking the error of the CE for different P, in case of Classical Ising model

using Test
using TensorKit
using ClusterExpansions
using PEPSKit
import PEPSKit: rmul!, σᶻᶻ, σˣ


# Define the numer type as BigFloat (default precision = 256 bits)
# but change it to 128 bits (otherwise too long)
T = Complex{BigFloat}
setprecision(128)
# Classical Ising
J = T(1.0)
g = T(0.0)

# Define: −J·σᶻᶻ and −g·J·σˣ (=0)
twosite_op = rmul!(σᶻᶻ(T), -J)
onesite_op = rmul!(σˣ(T), g * -J)

# Calculates the exponential exact: exp(−β·sum(H)), our system is torus with unit cell LxL
# if L=2: 4 sites, 1=(1,1), 2=(2,1), 3=(1,2), 4=(2,2), 2 bounds per 'connection' (1-2, but also 2-1 via 'outside')
function exponentiate_hamiltonian_periodic(T, onesite_op, twosite_op, β, L)
    N = L^2
    pspace = domain(twosite_op)[1]
    H = []

    # Loop over all sites in unit cell
    for x₁ = 1:L
        for y₁ = 1:L
            # define the site number: 1=(1,1), 2=(2,1), 3=(1,2), 4=(2,2) (L=2)
            i = L*(y₁-1) + x₁
            # Take outer tensor product to create: 1 1 ..  onesite_op ... 1 1
            term = ncon([onesite_op, [id(pspace) for _ = 1:N-1]...], [[-i, -N-i], [[-k, -N-k] for k = setdiff(1:N, i)]...], [false for _ = 1:N])
            # Split into N domain legs and N codomain legs
            push!(H, permute(term, (Tuple(1:N), Tuple(N+1:2*N))))

            # Define the bond in the x direction
            # We have original x₁,y₁ site and we want to go to right, there we have x₂,y₂ site
            x₂ = mod1(x₁+1, L)
            y₂ = mod1(y₁, L)
            # Find out which site number it corresponds to
            (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)
            # Define the twosite_op on that bond, rest is identities
            term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
            push!(H, permute(term, (Tuple(1:N), Tuple(N+1:2*N))))
            
            # Do same for bond in y-dimension (going up)
            x₂ = mod1(x₁, L)
            y₂ = mod1(y₁+1, L)
            (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)
            term = ncon([twosite_op, [id(pspace) for _ = 1:N-2]...], [[-i, -j, -N-i, -N-j], [[-k, -N-k] for k = setdiff(1:N, (i,j))]...], [false for _ = 1:N-1])
            push!(H, permute(term, (Tuple(1:N), Tuple(N+1:2*N))))
        end
    end
    # Now we calculate the exact exponential
    exp_H = exp(-β*sum(H))
    return exp_H
end


# Contract the periodic O tensor of CE
function contract_PEPO_periodic(O, L)
    N = L^2
    # Create a zero array of size Nx6, in which all 4+2 contraction indices for each of the N sites will come
    indices = fill(0, N, 6)
    # Counter for new labels
    bond = 1

    # Loop over all sites
    for x₁ = 1:L
        for y₁ = 1:L
            # Find neighbour in right direction
            x₂ = mod1(x₁+1, L)
            y₂ = mod1(y₁, L)
            (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)

            # If that bond/leg (out, in, NESW) has no index yet, give it bond
            if indices[i,4] == 0
                # To the original site (E bond)
                indices[i,4] = bond
                # Also to the 'right' site (W bond)
                indices[j,6] = bond
                bond += 1
            end
            
            # Do same for upper neighbour
            x₂ = mod1(x₁, L)
            y₂ = mod1(y₁+1, L)
            (i,j) = (L*(y₁-1) + x₁, L*(y₂-1) + x₂)
            if indices[i,5] == 0
                indices[i,5] = bond
                indices[j,3] = bond
                bond += 1
            end

            # Physical legs (out, in) get negative labels
            indices[i,1] = -i
            indices[i,2] = -N-i
        end
    end

    # Contract
    indices = [[indices[n,i] for i = 1:6] for n = 1:N]
    term = ncon([O for _ = 1:N], indices, [false for _ = 1:N])

    # N out, N in
    return permute(term, (Tuple(1:N), Tuple(N+1:2*N)))
end


bmin = -1
bmax = -2
pmax = 4
nodp = 4 # number of data points for β
βs = [T(10.0)^(x) for x in LinRange(bmin, bmax, nodp)]  # 10⁻¹ … 10⁻²
ps = [i for i = 2:pmax]                                 # 2, 3, 4
errors = zeros(length(βs), length(ps))
L = 2

# Perform one test for C4v and one with no symmetry on CE PEPO O.
@testset "Order of the cluster expansion with symmetry = $(symmetry)" for symmetry in ["C4v", nothing]
    # For each P
    for (j,p) = enumerate(ps)
        # Build the CE algorithm for classical Ising
        ce_alg = ising_operators(J, g, 0.0; p, verbosity = 0, symmetry, T)
        
        # For each β
        for (i,β) = enumerate(βs)
            @info "Starting with p = $p and β = $(round(Float64(β), digits = 4))"
            # Return O for that P and β combination (no i time evolution, just simply O(β) direct)
            O_clust_full = evolution_operator(ce_alg, β)
            exp_approx = contract_PEPO_periodic(O_clust_full, L)

            # Calculate exp(-βH) exact
            exp_exact = exponentiate_hamiltonian_periodic(T, onesite_op, twosite_op, β, L)
            
            # Save relative error
            error = norm(exp_exact-exp_approx)/norm(exp_approx)
            errors[i,j] = error
        end
    end

    # Now that we have all errors, calculate for a given P (for all neighbour pair of β)
    # the log-log slope of the error, we want to find β^p
    for p = 2:pmax
        for n = 1:nodp-1
            @test log(errors[n+1,p-1]/errors[n,p-1]) / log(βs[n+1]/βs[n]) / p ≈ 1 atol=5e-2
        end
    end
end
