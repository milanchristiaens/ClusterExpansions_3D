"""
find_orders_cotengra.py  --  contraction orders + slicing (cut) sets for the 27-tensor network of `contract_onesite`, with cotengra.

WHAT IT DOES
  * describes the closed network of `contract_onesite` (src/utility/ctmrg3D.jl): 27 tensors, 54 bonds (leaf ids 1..27 = textual order of the
    tensors in `contract_onesite`; bond labels = the index names of the @tensor expression);
  * asks cotengra for contraction trees that minimise flops / peak size / a mix ("combo");
  * for every tree and every target size, lets cotengra choose bonds to SLICE (cut) with three techniques:
        slice          greedy slicing of a FIXED tree                       (tree.slice_)
        slice+reconf   slicing interleaved with local re-optimisation       (tree.slice_and_reconfigure_)
        joint          one HyperOptimizer run that searches tree and slices together (slicing_opts / slicing_reconf_opts)
  * writes one line per candidate to a TSV file that `development/benchmark_contract_onesite.jl` reads, and prints a table.

NOTES
  * Sizes are in ELEMENTS (x16 bytes for ComplexF64).  `--targets 2^26` = at most 2^26 elements in the largest intermediate of a slice.
  * cotengra's search is stochastic and `--repeats` small: re-run, or raise --repeats, before trusting a "best" order.
  * cotengra's numbers assume dense tensors and ignore TensorKit's permutation copies, allocation cost and GEMM efficiency.
    The flops-optimal tree can be SLOWER in wall time than the memory-lean one, always time candidates in Julia!
  * The leaf order inside a pair is cotengra's, not necessarily the cheapest; the Julia side re-orders children to minimise peak
    (`onesite_reorder`) unless told otherwise.
"""

import argparse, math, sys, warnings            # argparse: used for command-line options
import cotengra as ctg                          # Contengra is the main python algorithm used

warnings.filterwarnings("ignore", message=".*kahypar.*")        # KaHyPar is a C++ library for this, which contengra uses as one of its path finders
                                                                # In this case it is not necessary (only 3x3x3), so we don't need to install it

# (name, labels): leaf id = position + 1; textual order of `contract_onesite`.  E = (a b; c d), A = (centre; 4 legs), O = M (the PEPO tensor)
TENSORS = [
    ("C1", ["U1", "A1", "U3"]),
    ("E1", ["U2", "A2", "U1", "U4"]),
    ("C2", ["U5", "A3", "U2"]),
    ("E5", ["A1", "M1", "B1", "M3"]),
    ("C5", ["B1", "L1", "L3"]),
    ("A2", ["NORTH", "A2", "M2", "B2", "M1"]),
    ("E6", ["A3", "M5", "B3", "M2"]),
    ("E9", ["L1", "B2", "L2", "L4"]),
    ("C6", ["B3", "L5", "L2"]),
    ("E2", ["U10", "A6", "U5", "U7"]),
    ("C3", ["U12", "A9", "U10"]),
    ("A1", ["UP", "U9", "U7", "U4", "U6"]),
    ("E4", ["U3", "A4", "U8", "U6"]),
    ("E3", ["U11", "A8", "U12", "U9"]),
    ("C4", ["U8", "A7", "U11"]),
    ("A5", ["WEST", "A4", "M3", "B4", "M8"]),
    ("E8", ["A7", "M8", "B7", "M11"]),
    ("E12", ["L8", "B4", "L3", "L6"]),
    ("C8", ["B7", "L8", "L11"]),
    ("O", ["UP", "NORTH", "EAST", "DOWN", "SOUTH", "WEST"]),
    ("A3", ["EAST", "A6", "M10", "B6", "M5"]),
    ("A4", ["SOUTH", "A8", "M11", "B8", "M12"]),
    ("E7", ["A9", "M12", "B9", "M10"]),
    ("A6", ["DOWN", "L4", "L7", "L9", "L6"]),
    ("E11", ["L12", "B8", "L11", "L9"]),
    ("E10", ["L5", "B6", "L10", "L7"]),
    ("C7", ["B9", "L12", "L10"]),
]
# The two reference trees (nested pairs of leaf ids) 
# Lander result (leads to low memory peak, needs more FLOPS than opt=true)
TREE_CURRENT = (1, (2, (3, ((((16, (17, (18, 19))), (20, (21, (22, (23, (24, (25, (26, 27)))))))), ((10, 11), (12, (13, (14, 15))))), ((4, 5), (6, (7, (8, 9))))))))
# opt = true result (leads to high memory peak)
TREE_FAST = (((((20, (((24, ((((25, 19), 27), ((17, 15), (((23, 11), 22), 14))), (((4, (13, 1)), 16), (18, 5)))), 8), (9, 26))), 6), 21), ((12, 10), (2, 3))), 7)

# List of index/leg tuples
INPUTS = [tuple(l) for _, l in TENSORS]
# sorted set of all 54 bond labels
LABELS = sorted({x for l in INPUTS for x in l})
# Check: 27 tensors, 54 labels, every label/leg in exact 2 tensors
assert len(INPUTS) == 27 and len(LABELS) == 54 and all(sum(x in l for l in INPUTS) == 2 for x in LABELS)

# Given bond label: is this a chi_env bond, a chi_peps bond or a D bond?
def bond_class(lab):
    # n[0] returns the class (A, E, C, O) of the tensors which touch lab
    # kinds is a set like {"C","E"}
    kinds = frozenset(n[0] for n, l in TENSORS if lab in l)
    # Return a number depending on which kind
    return {frozenset("CE"): 0, frozenset("AE"): 1, frozenset("AO"): 2}[kinds]   # 0: chi_env, 1: chi_peps, 2: D

# Given an array chi for bond dimensions for each bond class
# Create a Dict in which: label -> dimension
def size_dict_from_dims(chi):
    return {lab: chi[bond_class(lab)] for lab in LABELS}

# Read the file written by onesite_write_dims, and extract the Dict: label/leg -> dimension
def size_dict_from_file(path):
    d = {}

    # Read each line one by one
    for line in open(path):
        # Skip blank lines
        line = line.strip()
        # Skip # lines
        if line and not line.startswith("#"):
            # Extract: label dim. And make a Dict with label -> dim
            k, v = line.split(); d[k] = int(float(v))
    # If any bond is missing: exit with error meassage
    miss = set(LABELS) - set(d)
    if miss: sys.exit(f"dims file lacks bonds: {sorted(miss)}")
    return {k: d[k] for k in LABELS}

# Turn 2^26 into 67108864 and also accepts 1e8
def parse_size(s):
    s = s.strip()
    if "^" in s:
        a, b = s.split("^"); return int(a) ** int(b)
    return int(float(s))

# Contengra uses SSA representation: Inputs are 0...26 and new contractions get id 27, 28, ...
# # Instead of the above nested loops
# So here we go from nested to SSA
def nested_to_ssa(t):
    path = []; nxt = [len(INPUTS)]

    # Define a recursive function, which walks the tree depth first
    # If leaf, it returns t-1
    # If pair: it recurses left then right, appends (a,b ) and returns the new id
    def rec(t):
        if isinstance(t, int): return t - 1
        a = rec(t[0]); b = rec(t[1]); path.append((a, b)); nxt[0] += 1; return nxt[0] - 1
    rec(t); return tuple(path)

# Go back to nested: node maps: id → subtree
def ssa_to_nested(path):
    node = {i: i + 1 for i in range(len(INPUTS))}; nxt = len(INPUTS)
    for a, b in path:
        node[nxt] = (node.pop(a), node.pop(b)); nxt += 1
    # Unpack the single remaining entree
    (root,) = node.values(); return root

# Build a ContractionTree from the INPUTS (list of legs) and the nested representation (tranformed to ssa)
def tree_of(nested, size_dict, cuts=()):
    tree = ctg.ContractionTree.from_path(INPUTS, (), size_dict, ssa_path=nested_to_ssa(nested))

    # cuts() let's you evaluate a hand-picked cut set, for slicing
    for x in cuts: tree.remove_ind_(x)
    return tree

# Returns some data about the tree
def metrics(tree):
    # contraction_cost: total number of FLOPS (including slice multiplicity)
    # peak_size() = largest total size of all tensors that are alive at the same time (also inclouding input tensors)
    # peak_size() matters for memory
    # max_size() = size of single largest tensor that appears during contraction (we can only limit this, so choose this low enough)
    # multiplicity = number of slices (1 if unsliced)
    return dict(flops=float(tree.contraction_cost()), peak=float(tree.peak_size()), maxsize=float(tree.max_size()),
                slices=int(getattr(tree, "multiplicity", 1)))

def main():
    # STEP 1: read the options
    # argparse turns what you type after the script name into Python values
    ap = argparse.ArgumentParser()
    # Define different options
    ap.add_argument("--dims", default="8,5,5", help="chi_env,chi_peps,D (default 8,5,5)")
    ap.add_argument("--dims-file", default=None, help="text file with lines 'label dim' (54 bonds); overrides --dims")
    ap.add_argument("--targets", default="2^28,2^26,2^24", help="slicing targets: max elements of the largest intermediate")
    ap.add_argument("--minimize", default="flops,combo,size", help="comma list from flops,combo,size,write for the tree searches")
    ap.add_argument("--repeats", type=int, default=128, help="HyperOptimizer repeats per search (default 128)")
    ap.add_argument("--out", default="orders.tsv")
    a = ap.parse_args()
    # e.g. --dims 6,4,4 becomes a.dims == "6,4,4"
    # --repeats 32 becomes a.repeats == 32
    # Default is: dims 8,5,5, three slicing targets, three tree objectives, 128 repeats and the output file orders.tsv

    # STEP 2: turn the options into numbers
    # If a dimensions file was given, read it, otherwise "8,5,5" is transformed into a dict
    sd = size_dict_from_file(a.dims_file) if a.dims_file else size_dict_from_dims(tuple(int(x) for x in a.dims.split(",")))
    # Convert "2^28,2^26,2^24" into [268435456, 67108864, 16777216], this is target size which limits max_size
    targets = [parse_size(x) for x in a.targets.split(",")]

    # STEP 3: Collect the starting trees
    sources = {"current": TREE_CURRENT, "myfast": TREE_FAST}
    # For each objective: flops, combo, size. Contengra searches for a tree
    for m in a.minimize.split(","):
        # We define HyperOptimizer algorithm
        opt = ctg.HyperOptimizer(minimize=m, max_repeats=a.repeats, parallel=False, progbar=False)
        # We use it to find the best contraction paths
        # The result is added to sourcec as ctg-flops, ctg-combo and ctg-size
        sources["ctg-" + m] = ssa_to_nested(opt.search(INPUTS, (), sd).get_ssa_path())
        # At this point we should have 5 trees
    # Find the base metric, from the current contraction order, all other quantities are shown relative to these
    base = metrics(tree_of(TREE_CURRENT, sd))

    # STEP 4: slice every tree
    rows = []  # (name, cuts, nested tree, metrics)
    # This function adds one candidate to rows, as a tuple of (name, cuts, nested tree, metrics) 
    def add(name, tree):
        rows.append((name, sorted(tree.sliced_inds), ssa_to_nested(tree.get_ssa_path()), metrics(tree)))
    
    # For each of the 5 current trees:
    for sname, nested in sources.items():
        # Store the unscliced version, by using Dict sd from leg -> bound dimension
        add(f"{sname}/unsliced", tree_of(nested, sd))

        # For all max size targets, run 2 slicing methods on different copies of the tree, and store the results
        for T in targets:
            # slice_: pick bonds to cut greedly untill the largest intermediate is below the target (tree shape stays the same)
            t = tree_of(nested, sd); t.slice_(target_size=T);               add(f"{sname}/slice/T{int(math.log2(T))}", t)
            # slice_and_reconfigure_: do same as above, but also re-optimises parts of the tree while cutting, shape may change
            t = tree_of(nested, sd); t.slice_and_reconfigure_(target_size=T); add(f"{sname}/slice+reconf/T{int(math.log2(T))}", t)
        # THis gives 1+3x2=7 rows per tree, 35 in total thus

    # Third technique: Here the slicing already happens during the search, save result, 38 rows
    for T in targets:
        opt = ctg.HyperOptimizer(minimize="flops", max_repeats=a.repeats, parallel=False, progbar=False,
                                 slicing_opts={"target_size": T}, slicing_reconf_opts={"target_size": T})
        add(f"ctg-joint/T{int(math.log2(T))}", opt.search(INPUTS, (), sd))

    # STEP 5: write a file (which Julia later has to read elsewhere in the code) and print result
    with open(a.out, "w") as f:
        # First 2 header lines, which Julia will skip
        f.write("# name\tcuts\ttree\tctg_flops\tctg_peak_elems\tslices\n")
        f.write("# size_dict: " + " ".join(f"{k}={sd[k]}" for k in LABELS) + "\n")
        # Each row, becomes a line, with tab seperated columns 
        for name, cuts, nested, m in rows:
            f.write(f"{name}\t{','.join(cuts)}\t{nested!r}\t{m['flops']:.6e}\t{m['peak']:.6e}\t{m['slices']}\n")
    
    # Now also print the same
    print(f"{'candidate':30s} {'cuts':28s} {'flops/flops(cur)':>16s} {'peak elems':>11s} {'peak GiB(c128)':>14s} {'slices':>7s}")
    for name, cuts, nested, m in sorted(rows, key=lambda r: (r[3]["peak"], r[3]["flops"])):
        print(f"{name:30s} {','.join(cuts)[:28]:28s} {m['flops']/base['flops']:16.4f} {m['peak']:11.3e} {m['peak']*16/2**30:14.4f} {m['slices']:7d}")
    print(f"\nwrote {len(rows)} candidates to {a.out}.  reference (current, unsliced): flops {base['flops']:.4e}, peak {base['peak']:.4e}")

if __name__ == "__main__":
    main()
