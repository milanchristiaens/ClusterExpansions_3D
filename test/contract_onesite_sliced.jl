# test/contract_onesite_sliced.jl -- correctness tests (TINY sizes, no timings) for
#   src/utility/contract_onesite_orders.jl and src/utility/contract_onesite_sliced.jl, on a mock environment (no CTMRG3D needed).
# NOTE: the first call of every distinct 27-tensor kernel compiles for minutes (a few kernels are compiled here); `julia -O1` helps.
using Test, TensorKit, LinearAlgebra
using ClusterExpansions
const CE = ClusterExpansions

@testset "cost model" begin
    @test CE.onesite_selfcheck(verbose = false)
    d = CE.onesite_dims(8, 5, 5)
    @test length(d) == 54 == length(CE.ONESITE_LABELS)
    @test CE.onesite_best_cuts(CE.TREE_CURRENT, d, 2).cuts == [:A9, :L5]
    @test CE.onesite_best_cuts(CE.TREE_CURRENT, d, 2; cap = 1.2 * 2^30 / 16).cuts == [:L4, :L5]
    @test CE.onesite_cost(CE.TREE_FAST, d).flops < CE.onesite_cost(CE.TREE_CURRENT, d).flops
    @test CE.onesite_cost(CE.TREE_FAST, d).peak > CE.onesite_cost(CE.TREE_CURRENT, d).peak
    @test_throws ErrorException CE.onesite_parse_tree("((1, 2), 3)")                   # leaves missing
    t = CE.onesite_parse_tree(string(CE.TREE_FAST))
    @test t == CE.TREE_FAST
    path = tempname()
    CE.onesite_write_dims(path, d)
    @test countlines(path) == 55
end

@testset "mock environment, dense" begin
    env, M = CE.onesite_mock_env(ℂ^3, ℂ^2, ℂ^2)
    @test CE.onesite_dims(env, M) == CE.onesite_dims(3, 2, 2)
    z0 = CE.contract_onesite_kernel(env.C, env.E, env.A, M)
    @test isfinite(abs(z0)) && abs(z0) > 0

    # generated kernels agree with the two static ones
    kcur = CE.onesite_make_kernel(CE.TREE_CURRENT); kfast = CE.onesite_make_kernel(CE.TREE_FAST)
    @test kcur(env.C, env.E, env.A, M) ≈ z0
    @test kfast(env.C, env.E, env.A, M) ≈ z0
    @test CE.contract_onesite_kernel_fast(env.C, env.E, env.A, M) ≈ z0

    # slicing: static kernels, a tree as kernel, number of slices
    for cuts in ([:L5 => 3], [:L5 => 3, :L4 => 2], [:U1 => 3, :U10 => 3, :U6 => 2])
        @test CE.check_slicing(env, M, cuts)[1]
        @test CE.check_slicing(env, M, cuts; kernel = :fast)[1]
    end
    # a tree as kernel (compiles one more kernel); re-ordered children for the sliced network
    tr = CE.onesite_reorder(CE.TREE_FAST, CE.onesite_dims(3, 2, 2); cuts = [:U1, :U10])
    @test CE.check_slicing(env, M, [:U1 => 3, :U10 => 3]; kernel = tr)[1]
    @test CE.onesite_nslices(env, M, [:L5 => 3, :L4 => 2]) == 6
    @test CE.onesite_nslices(env, M, [:L5 => 2]) == 2                                  # coarser pieces
    @test CE.check_slicing(env, M, [:L5 => 2, :L4 => 1])[1]
    @test CE.onesite_full_cuts(env, M, [:L5, :L4, :A8]) == [:L5 => 3, :L4 => 2, :A8 => 2]
    @test CE.check_slicing(env, M, Pair{Symbol,Int}[])[1]
end

@testset "allocators" begin
    env, M = CE.onesite_mock_env(ℂ^3, ℂ^2, ℂ^2)
    cuts = [:L5 => 3, :L4 => 2]
    z0 = CE.contract_onesite_kernel(env.C, env.E, env.A, M)

    ra = CE.ReuseAllocator()
    z1 = CE.contract_onesite_sliced(env, M, cuts; allocator = ra)
    @test z1 ≈ z0
    @test ra.live_bytes == 0                       # everything handed out came back
    @test ra.hits > 0 && ra.misses > 0              # later slices re-used the first slice's blocks
    @test ra.pooled_bytes > 0
    CE.release!(ra); @test ra.pooled_bytes == 0

    rb = CE.ReuseAllocator(max_pool_bytes = 0)       # pool disabled: still correct
    @test CE.contract_onesite_sliced(env, M, cuts; allocator = rb) ≈ z0
    @test rb.hits == 0

    rc = CE.ReuseAllocator(max_total_bytes = 1)      # forces eviction on every miss: still correct
    @test CE.contract_onesite_sliced(env, M, cuts; allocator = rc) ≈ z0
    CE.release!(rc)

    ta = CE.TrackingAllocator(CE.ReuseAllocator())
    @test CE.contract_onesite_sliced(env, M, cuts; allocator = ta) ≈ z0
    @test ta.live_bytes == 0 && ta.peak_bytes > 0 && ta.nalloc > 0
    peak_sliced = ta.peak_bytes
    tb = CE.TrackingAllocator(CE.ManualAllocator())
    @test CE.contract_onesite_kernel(env.C, env.E, env.A, M; allocator = tb) ≈ z0
    @test tb.live_bytes == 0
    @test tb.peak_bytes > peak_sliced                # slicing L5, L4 lowers the peak of the temporaries
    CE.release!(ta.inner)
end

@testset "mock environment, Z2 graded" begin
    Venv = Vect[Z2Irrep](0 => 2, 1 => 2); Vpeps = Vect[Z2Irrep](0 => 2, 1 => 1); VD = Vect[Z2Irrep](0 => 1, 1 => 1)
    env, M = CE.onesite_mock_env(Venv, Vpeps, VD)
    @test CE.check_slicing(env, M, [:L5 => 4, :L4 => 3])[1]
    @test CE.check_slicing(env, M, [:U1 => 4, :U10 => 2]; kernel = :fast)[1]
end
