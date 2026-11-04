# test/contract_onesite_auto.jl -- tests (TINY sizes, no timings) for src/utility/contract_onesite_auto.jl, on the mock environment.
# The contraction tests compile the static kernel (a minute or two with `julia -O1`); the model-only tests take seconds.

using Test, TensorKit, LinearAlgebra
using ClusterExpansions
const CE = ClusterExpansions

@testset "choose cuts: cost model only (8,5,5), ComplexF64" begin
    d = CE.onesite_dims(8, 5, 5)
    p0 = CE.onesite_choose_cuts(d; ram_gib = 256)
    @test p0.k == 0 && p0.fits && p0.slices == 1 && p0.flops_rel == 1.0
    p1 = CE.onesite_choose_cuts(d; ram_gib = 64)
    @test p1.k == 1 && p1.fits && p1.est_gib <= 0.8 * 64 && p1.flops_rel >= 1.0
    p2 = CE.onesite_choose_cuts(d; ram_gib = 16)
    @test p2.k == 2 && p2.fits && p2.est_gib <= 0.8 * 16

    # LEFT: least FLOPS possible for 2 cuts (which you need to get under 16Gb ram)
    # RIGHT: lowest memory peak possible for 2 cuts
    # Flops of left should be <= flops of right
    @test p2.flops_rel <= CE.onesite_cost(CE.TREE_CURRENT, d; cuts = CE.onesite_best_cuts(CE.TREE_CURRENT, d, 2).cuts).flops / CE.onesite_cost(CE.TREE_CURRENT, d).flops + 1e-12
    
    # Check if contractions of Float64 elements (8 bytes per number) indeed need <= the cuts needed for Complex (16 bytes per element)
    @test CE.onesite_choose_cuts(d; ram_gib = 64, bytes = 8).k <= p1.k
    
    # a budget that cannot be met: should give fits=False, but still result in contraction order
    pn = CE.onesite_choose_cuts(d; ram_gib = 2, kmax = 2)
    @test !pn.fits && pn.k == 2

    # We only allow cuts up to 5 and :reuse and :manual allocator
    @test_throws ErrorException CE.onesite_choose_cuts(d; ram_gib = 64, kmax = 6)
    @test_throws ErrorException CE.onesite_choose_cuts(d; ram_gib = 64, allocator = :other)
    
    # plan_of for a given set
    q = CE.onesite_plan_of(d, [:A9, :L5]; ram_gib = 64)
    @test q.cuts == [:A9, :L5] && q.k == 2 && q.fits
end

@testset "contract_onesite_auto on the mock environment" begin
    env, M = CE.onesite_mock_env(ℂ^3, ℂ^2, ℂ^2)
    d = CE.onesite_dims(env, M)
    z0 = CE.contract_onesite_kernel(env.C, env.E, env.A, M)
    
    # Put the baseline to 0 for now
    rules0 = (baseline_gib = 0.0, reuse = 2.7, manual = 1.6)                   # no Julia baseline at this tiny size
    # Calculate the model peak of unsliced network
    peak0_gib = CE.onesite_cost(CE.TREE_CURRENT, d).peak * 16 / 2^30

    # enough RAM: no cut, same as the unsliced kernel
    @test CE.contract_onesite_auto(env, M; ram_gib = 1.0e3, verbosity = 0) ≈ z0
    # the dispatcher with a number goes to the same code
    @test CE.contract_onesite_dispatch(env, M, 1.0e3; verbosity = 0) ≈ z0

    # a budget that is a third of what the unsliced network needs: cuts are chosen, the value does not change
    ram = (peak0_gib / 3) * 2.7 / 0.8
    @test CE.onesite_choose_cuts(d; ram_gib = ram, rules = rules0).k >= 1
    for allocator in (:reuse, :manual)
        z = CE.contract_onesite_auto(env, M; ram_gib = ram, rules = rules0, allocator, verbosity = 0)
        @test z ≈ z0
    end

    # forced cuts (no search for best cuts, just give them as input)
    z = CE.contract_onesite_auto(env, M; ram_gib = 1.0e3, cuts = [:L5, :L4], verbosity = 0)
    @test z ≈ z0

    # Float64 tensors (8-byte elements)
    envr, Mr = CE.onesite_mock_env(ℂ^3, ℂ^2, ℂ^2; T = Float64)
    zr0 = CE.contract_onesite_kernel(envr.C, envr.E, envr.A, Mr)
    @test CE.contract_onesite_auto(envr, Mr; ram_gib = ram, rules = rules0, verbosity = 0) ≈ zr0

    # nothing fits and no force: error
    @test_throws ErrorException CE.contract_onesite_auto(env, M; ram_gib = 1.0e-9, rules = rules0, kmax = 1, verbosity = 0)
end


@testset "contract_onesite_auto, Z2 graded mock" begin
    Venv = Vect[Z2Irrep](0 => 2, 1 => 2); Vpeps = Vect[Z2Irrep](0 => 2, 1 => 1); VD = Vect[Z2Irrep](0 => 1, 1 => 1)
    env, M = CE.onesite_mock_env(Venv, Vpeps, VD)
    d = CE.onesite_dims(env, M)
    z0 = CE.contract_onesite_kernel(env.C, env.E, env.A, M)
    rules0 = (baseline_gib = 0.0, reuse = 2.7, manual = 1.6)
    peak0_gib = CE.onesite_cost(CE.TREE_CURRENT, d).peak * 16 / 2^30
    
    # If only one third of memory available, need slicing
    z = CE.contract_onesite_auto(env, M; ram_gib = (peak0_gib / 3) * 2.7 / 0.8, rules = rules0, verbosity = 0)
    @test z ≈ z0
    @test CE.onesite_choose_cuts(d; ram_gib = (peak0_gib / 3) * 2.7 / 0.8, rules = rules0).k >= 1
end
