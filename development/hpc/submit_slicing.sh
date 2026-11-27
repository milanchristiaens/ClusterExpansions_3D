#!/bin/bash
# Submits the slicing jobs: the plan benchmark (29 GiB plans on a quarter node) and slicing_cost.jl for TREE_CURRENT at both budgets.
# The benchmark needs the 29 GiB plans in the plan file first (find_orders_ome.jl, job_plans.pbs); slicing_cost.jl with TREE_CURRENT needs no plan file.
set -euo pipefail

# ---------------- settings ----------------
BENCH_PROFILES="hpc29"                         # plan benchmark: hpc29 (29 GiB plans, quarter node); hpc (3 GiB plans, 4 cores) already runs in job_plans.pbs
SLICING_JOBS="current:3 current:29"            # slicing_cost.jl, tree:RAM budget (GiB); trees: current, opt_true, ome_flops, ome_scN (see slicing_cost.jl)
CORES_29=16                                    # cores and threads (Julia and BLAS) of the 29 GiB jobs; the 3 GiB jobs use 4
# ------------------------------------------

for p in $BENCH_PROFILES; do
    cores=4; [ "$p" = hpc29 ] && cores=$CORES_29
    qsub -N "bench_$p" -l nodes=1:ppn=$cores -v "PROFILE=$p,THREADS=$cores" development/hpc/job_benchmark_plans.pbs
done
for j in $SLICING_JOBS; do
    t=${j%:*}; r=${j#*:}
    cores=4; [ "$r" = 29 ] && cores=$CORES_29
    qsub -N "slicing_${t}_$r" -l nodes=1:ppn=$cores -v "TREE=$t,RAM=$r,THREADS=$cores" development/hpc/job_slicing_cost.pbs
done
echo "Submitted. Follow with: squeue -u \$USER"
