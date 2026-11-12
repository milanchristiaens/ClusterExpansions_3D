# test/contract_onesite_slicing.jl -- tests (TINY sizes, no timings) for src/utility/contract_onesite_slicing.jl, on a mock environment (no CTMRG3D needed).
# NOTE: the first call of every distinct 27-tensor kernel compiles for minutes (TREE_CURRENT, OPT_TRUE and one generated tree are compiled here;
# every other test re-uses them, the generated kernels are cached); `julia -O1` helps.

using Test, TensorKit, LinearAlgebra
using ClusterExpansions
const CE = ClusterExpansions

# Reference results of the cost model at (χenv, χpeps, D) = (8, 5, 5), calculated independently with the python model of the old cotengra script:
# name, tree, cuts, flops, peak (elements), slices
const SELFCHECK = [
    ("current (no cuts)", CE.TREE_CURRENT, Symbol[], 2272611776512.0, 2601600000.0, 1.0),
    ("current + L5", CE.TREE_CURRENT, Symbol[:L5], 2276800938496.0, 326600000.0, 8.0),
    ("current + L5,L4", CE.TREE_CURRENT, Symbol[:L5, :L4], 2295962042880.0, 66600000.0, 40.0),
    ("current + L5,L4,A8", CE.TREE_CURRENT, Symbol[:L5, :L4, :A8], 2323064230400.0, 14600000.0, 200.0),
    ("opt_true (no cuts)", CE.OPT_TRUE, Symbol[], 1512621097600.0, 16000000000.0, 1.0),
    ("opt_true + U10,U6,U9", CE.OPT_TRUE, Symbol[:U10, :U6, :U9], 1553400616000.0, 80000000.0, 200.0),
    ("opt_true + U1,U10", CE.OPT_TRUE, Symbol[:U1, :U10], 1549349683200.0, 250000000.0, 64.0),
]

# ----------------------------------------------------------------------------------------------------------------------
# Cost model and choosing cuts: only the cost model is used, so this takes seconds
# ----------------------------------------------------------------------------------------------------------------------
@testset "cost model" begin
    d = CE.onesite_dims(8, 5, 5)
    @test length(d) == 54 == length(CE.ONESITE_LABELS)

    # The Julia cost model reproduces the reference numbers
    for (name, tree, cuts, f, p, s) in SELFCHECK
        c = CE.onesite_cost(tree, d; cuts)
        @testset "$name" begin
            @test c.flops ≈ f rtol = 1e-9
            @test c.peak ≈ p rtol = 1e-9
            @test c.slices ≈ s rtol = 1e-9
        end
    end

    # Find best 2 cuts
    @test CE.onesite_best_cuts(CE.TREE_CURRENT, d, 2).cuts == [:A9, :L5]
    # Find sliced contraction under 1.2 GiB RAM cap, with lowest flops.
    @test CE.onesite_best_cuts(CE.TREE_CURRENT, d, 2; cap = 1.2 * 2^30 / 16).cuts == [:L4, :L5]

    # Check that OPT_TRUE tree indeed results in lower flops, but higher memory cost in our model
    @test CE.onesite_cost(CE.OPT_TRUE, d).flops < CE.onesite_cost(CE.TREE_CURRENT, d).flops
    @test CE.onesite_cost(CE.OPT_TRUE, d).peak > CE.onesite_cost(CE.TREE_CURRENT, d).peak

    # Trees as text: leaves missing is an error, converting a tree to a string and back gives the same tree
    @test_throws ErrorException CE.onesite_parse_tree("((1, 2), 3)")
    @test CE.onesite_parse_tree(string(CE.OPT_TRUE)) == CE.OPT_TRUE

    # Reordering the children keeps the flops, and gives exactly the peak of the best order (and never a larger one than the written order)
    for (tree, cuts) in ((CE.TREE_CURRENT, [:L5, :L4]), (CE.OPT_TRUE, [:U1, :U10]), (CE.TREE_CURRENT, Symbol[]))
        t2 = CE.onesite_reorder(tree, d; cuts)
        cb = CE.onesite_cost(tree, d; cuts, order = :best)
        c2 = CE.onesite_cost(t2, d; cuts)
        @test c2.peak == cb.peak && c2.flops == cb.flops
        @test c2.peak <= CE.onesite_cost(tree, d; cuts).peak
        @test sort(collect(CE._os_leaves(t2))) == collect(1:27)
    end
end

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

    # The tree of a plan is the one that has to be contracted: it has the predicted peak and the same leaves
    for p in (p0, p1, p2)
        @test sort(collect(CE._os_leaves(p.tree))) == collect(1:27)
        @test CE.onesite_cost(p.tree, d; cuts = p.cuts).peak * 16 / 2^30 ≈ p.peak_gib
    end

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

# ----------------------------------------------------------------------------------------------------------------------
# Plan file (csv) and OMEinsumContractionOrders.jl: only the cost model is used, so this takes seconds
# ----------------------------------------------------------------------------------------------------------------------
@testset "plan file and OMEinsumContractionOrders" begin
    d = CE.onesite_dims(8, 5, 5)

    # Trees found by OMEinsumContractionOrders are valid trees of our network, and our cost model agrees with the one of the package
    # (this checks that the leaf numbers of the two descriptions of the network are the same)
    found = CE.onesite_find_trees(d; n_targets = 1, ntrials = 1, niters = 5)
    @test !isempty(found)
    for f in found
        @test sort(collect(CE._os_leaves(f.tree))) == collect(1:27)
        @test !CE._os_has_outer(f.tree)
        @test CE.onesite_parse_tree(string(f.tree)) == f.tree
        @test log2(CE.onesite_cost(f.tree, d).flops) ≈ f.tc rtol = 1e-9
    end

    # Plans for a few budgets (candidates: TREE_CURRENT, OPT_TRUE and the found trees)
    rows = CE.onesite_find_plans(8, 5, 5; ram_gibs = (16, 256), n_targets = 1, ntrials = 1, niters = 5, verbosity = 0)
    @test !isempty(rows)
    for r in rows
        p = CE.onesite_plan_of(d, r.cuts; ram_gib = r.ram_gib, tree = r.tree)
        @test p.fits && p.flops ≈ r.flops
        @test r.k == length(r.cuts) && isnan(r.time_s)
    end

    # Writing and reading: the same rows come back, writing twice adds nothing
    path = tempname() * ".csv"
    @test CE.onesite_write_plans(path, rows) == length(rows)
    @test CE.onesite_write_plans(path, rows) == 0
    back = CE.onesite_read_plans(path)
    @test length(back) == length(rows)
    @test all(back[i].tree == rows[i].tree && back[i].cuts == rows[i].cuts && back[i].chi_env == 8 for i in eachindex(rows))
    @test countlines(path) == length(rows) + 1                    # header + one line per plan
    @test CE.onesite_read_plans(joinpath(tempdir(), "does_not_exist.csv")) == []
    @test isempty(CE.onesite_plans_for(CE.onesite_dims(3, 2, 2), path))        # plans are only used for exactly their dimensions
    @test length(CE.onesite_plans_for(d, path)) == length(rows)

    # Choosing: select = :measured ignores plans without time_s, so the brute-force plan on TREE_CURRENT is used;
    # select = :model takes the plan with the fewest predicted flops, which beats TREE_CURRENT here (OPT_TRUE has fewer flops)
    pm = CE.onesite_choose_plan(d; ram_gib = 64, plans = path)
    @test startswith(pm.source, "TREE_CURRENT") && pm.k == 1
    pf = CE.onesite_choose_plan(d; ram_gib = 64, plans = path, select = :model)
    @test startswith(pf.source, "plan file") && pf.fits && pf.flops < pm.flops
    # plans = nothing ignores the file, a given tree or given cuts too
    @test startswith(CE.onesite_choose_plan(d; ram_gib = 64, plans = nothing, select = :model).source, "TREE_CURRENT")
    @test startswith(CE.onesite_choose_plan(d; ram_gib = 64, plans = path, select = :model, tree = CE.OPT_TRUE).source, "given tree")
    @test CE.onesite_choose_plan(d; ram_gib = 64, plans = path, select = :model, cuts = [:L5]).cuts == [:L5]
    @test_throws ErrorException CE.onesite_choose_plan(d; ram_gib = 64, select = :other)

    # Fill in a measured time in the line of the second plan (column 13): select = :measured now uses it (if it fits), and writing the rows
    # again does not overwrite the time
    lines = readlines(path)
    f = split(lines[3], ';'); f[13] = "123.5"; lines[3] = join(f, ';')
    write(path, join(lines, "\n") * "\n")
    @test CE.onesite_read_plans(path)[2].time_s == 123.5
    ps = CE.onesite_choose_plan(d; ram_gib = rows[2].ram_gib, plans = path)
    @test startswith(ps.source, "plan file") && ps.tree == rows[2].tree && ps.cuts == rows[2].cuts
    @test CE.onesite_write_plans(path, rows) == 0
    @test CE.onesite_read_plans(path)[2].time_s == 123.5

    # The measurements that development/contraction_plans/benchmark_contract_onesite.jl writes back (21-column layout: time_s, time_real_s, allocator ... note).
    # A row that was never measured has empty columns; older lines with 13 or 20 columns are still read (time_real_s is then empty).
    r0 = CE.onesite_read_plans(path)[1]
    @test isnan(r0.rss_gib) && r0.note == "" && isnan(r0.time_real_s) && length(split(readlines(path)[2], ';')) == 21
    path13 = tempname()
    write(path13, join(split(readlines(path)[1], ';')[1:13], ';') * "\n" * join(split(readlines(path)[3], ';')[1:13], ';') * "\n")
    @test CE.onesite_read_plans(path13)[1].time_s == 123.5 && isnan(CE.onesite_read_plans(path13)[1].check) && isnan(CE.onesite_read_plans(path13)[1].time_real_s)
    # a 20-column line of the older layout: the measurement columns are 14..20
    f20 = split(readlines(path)[3], ';')[1:13]; append!(f20, ["manual", "1.5", "0.5", "0.25", "7", "1e-13", "old"])
    path20 = tempname(); write(path20, join(f20, ';') * "\n")
    o20 = CE.onesite_read_plans(path20)[1]
    @test o20.time_s == 123.5 && isnan(o20.time_real_s) && o20.allocator == "manual" && o20.rss_gib == 1.5 && o20.minflt == 7 && o20.check == 1e-13 && o20.note == "old"

    upd = (chi_env = rows[1].chi_env, chi_peps = rows[1].chi_peps, D = rows[1].D, tree = rows[1].tree, cuts = rows[1].cuts,
           time_s = 4.5, allocator = "reuse", rss_gib = 2.25, pool_gib = NaN, sys_s = 0.5, minflt = 1000.0, check = 1.0e-12, note = "a; b")
    # a MOCK update: time_s and the seven measurement columns (the row has no real time yet)
    @test CE.onesite_update_plans(path, [upd]) == 1
    r1 = CE.onesite_read_plans(path)[1]
    @test r1.time_s == 4.5 && isnan(r1.time_real_s) && r1.allocator == "reuse" && r1.rss_gib == 2.25 && isnan(r1.pool_gib) && r1.sys_s == 0.5 && r1.minflt == 1000 && r1.check == 1.0e-12 && r1.note == "a, b"
    @test CE.onesite_read_plans(path)[2].time_s == 123.5            # the other rows are untouched
    @test length(CE.onesite_read_plans(path)) == length(rows)
    @test CE.onesite_write_plans(path, rows) == 0                     # still no duplicates, the measurements stay
    @test CE.onesite_read_plans(path)[1].time_s == 4.5

    # a REAL update: time_real_s, and the seven measurement columns now belong to the real run; time_s stays
    @test CE.onesite_update_plans(path, [merge(upd, (time_s = 9.0, allocator = "manual", rss_gib = 3.5, note = "real"))]; env = :real) == 1
    r1 = CE.onesite_read_plans(path)[1]
    @test r1.time_s == 4.5 && r1.time_real_s == 9.0 && r1.allocator == "manual" && r1.rss_gib == 3.5 && r1.note == "real"
    @test CE.onesite_plan_time(r1) == 9.0                              # the real time counts, not the (smaller) mock time
    # a later MOCK update changes time_s only; the measurements of the real run are kept
    @test CE.onesite_update_plans(path, [merge(upd, (time_s = 5.5, allocator = "reuse", rss_gib = 1.0, note = "mock again"))]) == 1
    r1 = CE.onesite_read_plans(path)[1]
    @test r1.time_s == 5.5 && r1.time_real_s == 9.0 && r1.allocator == "manual" && r1.rss_gib == 3.5 && r1.note == "real"
    # an update without a finite time (skipped or failed run) only writes the note, and keeps the times and the measurements
    @test CE.onesite_update_plans(path, [merge(upd, (time_s = NaN, note = "skipped: too big"))]; env = :real) == 1
    r1 = CE.onesite_read_plans(path)[1]
    @test r1.time_s == 5.5 && r1.time_real_s == 9.0 && r1.rss_gib == 3.5 && r1.note == "skipped: too big"
    # a mock update of an OLD-layout line (13 columns) writes the 21-column layout
    @test CE.onesite_update_plans(path13, [merge(upd, (cuts = rows[2].cuts, tree = rows[2].tree, time_s = 7.0))]) == 1
    @test length(split(readlines(path13)[end], ';')) == 21 && CE.onesite_read_plans(path13)[1].time_s == 7.0
    @test_throws ErrorException CE.onesite_update_plans(path, [upd]; env = :other)
    @test_throws ErrorException CE.onesite_update_plans(path, [merge(upd, (cuts = [:L5, :L4, :A8, :U1, :U2, :U3],))])    # this row is not in the file

    # Choosing with a real time: plan 1 has time_s = 5.5 and time_real_s = 9.0, plan 2 has time_s = 123.5 only. The effective times are 9.0 and 123.5,
    # so plan 1 wins; with a real time of 200 for plan 1 plan 2 (123.5) wins, although the mock time of plan 1 (5.5) is smaller
    @test CE.onesite_plan_time(CE.onesite_read_plans(path)[2]) == 123.5
    b1 = CE.onesite_choose_plan(d; ram_gib = max(rows[1].ram_gib, rows[2].ram_gib), plans = path)
    @test startswith(b1.source, "plan file") && b1.tree == rows[1].tree && b1.cuts == rows[1].cuts
    CE.onesite_update_plans(path, [merge(upd, (time_s = 200.0,))]; env = :real)
    b2 = CE.onesite_choose_plan(d; ram_gib = max(rows[1].ram_gib, rows[2].ram_gib), plans = path)
    @test startswith(b2.source, "plan file") && b2.tree == rows[2].tree && b2.cuts == rows[2].cuts
end

# ----------------------------------------------------------------------------------------------------------------------
# Slicing machinery on the mock environment: this compiles the 27-tensor kernels
# ----------------------------------------------------------------------------------------------------------------------
@testset "mock environment, dense" begin
    env, M = CE.onesite_mock_env(ℂ^3, ℂ^2, ℂ^2)
    @test CE.onesite_dims(env, M) == CE.onesite_dims(3, 2, 2)
    z0 = CE.contract_onesite_kernel(env.C, env.E, env.A, M)

    # Calculate the contractrion value z0, (it should be /=0 because we will use ≈ which is relative)
    @test isfinite(abs(z0)) && abs(z0) > 0

    # the two precompiled trees give the precompiled kernels, and they agree with each other
    @test CE.onesite_make_kernel(CE.TREE_CURRENT) === CE.contract_onesite_kernel
    @test CE.onesite_make_kernel(CE.OPT_TRUE) === CE.contract_onesite_kernel_opt_true
    @test CE.contract_onesite_kernel_opt_true(env.C, env.E, env.A, M) ≈ z0

    # For different cuts (on current tree and OPT_TRUE tree),
    # test whether the result corresponds to z
    for cuts in ([:L5 => 3], [:L5 => 3, :L4 => 2], [:U1 => 3, :U10 => 3, :U6 => 2])
        @test CE.check_slicing(env, M, cuts)[1]
        @test CE.check_slicing(env, M, cuts; kernel = :opt_true)[1]
    end

    # Test the reordening of the children, and use it as kernel tr (a tree that has to be generated and compiled), and check if it corresponds to z
    tr = CE.onesite_reorder(CE.OPT_TRUE, CE.onesite_dims(3, 2, 2); cuts = [:U1, :U10])
    @test CE.check_slicing(env, M, [:U1 => 3, :U10 => 3]; kernel = tr)[1]
    # a generated kernel is made only once per tree
    @test CE.onesite_make_kernel(tr) === CE.onesite_make_kernel(tr)

    # Check the number of slices given a cut
    @test CE.onesite_nslices(env, M, [:L5 => 3, :L4 => 2]) == 6

    # Test the coarse pieces (cutting D=3 into 2 pieces)
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
    @test ra.hits > 0 && ra.misses > 0              # later slices re-used the first slice's blocks (first slice was seen as misses)
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
    # Create Z2 graded mock environment
    Venv = Vect[Z2Irrep](0 => 2, 1 => 2); Vpeps = Vect[Z2Irrep](0 => 2, 1 => 1); VD = Vect[Z2Irrep](0 => 1, 1 => 1)
    env, M = CE.onesite_mock_env(Venv, Vpeps, VD)

    # Check if slicing this results in same answer as unsliced contraction
    @test CE.check_slicing(env, M, [:L5 => 4, :L4 => 3])[1]
    @test CE.check_slicing(env, M, [:U1 => 4, :U10 => 2]; kernel = :opt_true)[1]
end

# ----------------------------------------------------------------------------------------------------------------------
# contract_onesite_auto: the main entry
# ----------------------------------------------------------------------------------------------------------------------
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

    # A plan from a plan file (the reordered OPT_TRUE tree + 2 cuts, the generated kernel of the previous testset is used again):
    # with kmax = 0 the brute-force plan has no cuts, which does not fit a budget that just fits the plan of the file, so the plan of the file is used
    tr = CE.onesite_reorder(CE.OPT_TRUE, d; cuts = [:U1, :U10])
    prow = CE.onesite_plan_of(d, [:U1, :U10]; ram_gib = 1.0e3, rules = rules0, tree = tr)
    ram_row = prow.est_gib / 0.8 * 1.05
    @test CE.onesite_cost(CE.TREE_CURRENT, d).peak * 16 / 2^30 * 2.7 > ram_row * 0.8            # TREE_CURRENT without cuts does not fit that budget
    path = tempname() * ".csv"
    row = (chi_env = 3, chi_peps = 2, D = 2, tree = tr, cuts = [:U1, :U10], source = "test row", k = 2, slices = prow.slices,
           flops = prow.flops, peak_gib = prow.peak_gib, ram_gib = ram_row, bytes = 16, time_s = 1.0)
    CE.onesite_write_plans(path, [row])
    plan = CE.onesite_choose_plan(d; ram_gib = ram_row, rules = rules0, plans = path, kmax = 0)
    @test startswith(plan.source, "plan file") && plan.tree == tr && plan.fits
    @test CE.contract_onesite_auto(env, M; ram_gib = ram_row, rules = rules0, plans = path, kmax = 0, verbosity = 0) ≈ z0
    # the same with a row without time: not used in the default mode, so nothing fits and it is an error; with select = :model it is used
    row_unmeasured = merge(row, (time_s = NaN,))
    path2 = tempname() * ".csv"
    CE.onesite_write_plans(path2, [row_unmeasured])
    @test_throws ErrorException CE.contract_onesite_auto(env, M; ram_gib = ram_row, rules = rules0, plans = path2, kmax = 0, verbosity = 0)
    @test CE.contract_onesite_auto(env, M; ram_gib = ram_row, rules = rules0, plans = path2, select = :model, kmax = 0, verbosity = 0) ≈ z0
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