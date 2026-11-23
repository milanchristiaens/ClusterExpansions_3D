#!/bin/bash
# Submits the Hortense jobs of the compare scripts (job_ctmrg_compare.pbs, job_vbpeps_compare.pbs) for PROFILE pilot, timing or hpc.
# All settings are below; nothing is given on the command line.
set -euo pipefail

# ---------------- settings ----------------
PROFILE=hpc                                     # pilot, timing or hpc
ALGORITHMS="ctmrg vbpeps"                       # which compare scripts get jobs: ctmrg and/or vbpeps (timing: only vbpeps)
CTMRG_DIMS="16_6_3"                             # pilot candidates (chienv_chipeps_DCE) of CTMRG3D
CTMRG_CORES=5                                   # cores of every CTMRG3D job
VBPEPS_DIMS="27_3_3 32_3_3 36_3_3"              # pilot candidates of vbPEPS
CTMRG_CASES="svd_cold_auto svd_warm_auto oblique_cold_auto oblique_warm_auto svd_cold_w1 svd_warm_w1 oblique_cold_w1 oblique_warm_w1 octant_cold octant_warm"
# other CTMRG3D cases (CASES_EXTRA of the compare script): svdclass_* and obliqueclass_* (class_projectors = true)
VBPEPS_CASES="false_cold false_warm"
VBPEPS_IMPLS_TIMING="pepskit betterpepskit"     # timing: pepskit = PEPSKit 0.8.1 (slow reference), betterpepskit = fast
VBPEPS_IMPL_RUN="betterpepskit"                 # implementation of vbPEPS in the pilot and the full run
VBPEPS_CORES=3                                  # cores of the vbPEPS pilot and full run (the julia_threads of the profiles :pilot and :hpc)
VBPEPS_TIMING_CORES=4                           # cores of both implementations in the timing
PEPO_D_CE=3                                     # the PEPO file of this D_CE must exist (data/pepos/, job_pepos.pbs)
# walltimes: budget of the profile + compilation + plots, with some margin
CTMRG_WALLTIME_PILOT=03:30:00                   # 2 cases × 80 min + compilation
CTMRG_WALLTIME_HPC=14:30:00                     # 12.8 h (5 cores) + compilation; 17:30:00 for 4 cores, 12:30:00 for 6
VBPEPS_WALLTIME_PILOT=03:00:00                  # 2 cases × 1 h + compilation (betterpepskit: no Zygote compilation of the CTMRG)
VBPEPS_WALLTIME_TIMING=04:30:00                 # 2 cases × 30 min + compilation (pepskit: about 2 h for `true`)
VBPEPS_WALLTIME_HPC=23:00:00                    # 21.3 h + compilation
# ------------------------------------------

ls data/pepos/ising_3D_*_D${PEPO_D_CE}_*.jld2 > /dev/null 2>&1 || { echo "No PEPO file of D_CE = $PEPO_D_CE in data/pepos: run qsub development/hpc/job_pepos.pbs first"; exit 1; }
RUN=$(date +%Y%m%d_%H%M)
for alg in $ALGORITHMS; do
    job="development/hpc/job_${alg}_compare.pbs"
    if [ "$alg" = ctmrg ]; then
        if [ "$PROFILE" = pilot ]; then
            for d in $CTMRG_DIMS; do qsub -N "ctmrg_pilot_$d" -l nodes=1:ppn=$CTMRG_CORES -l walltime=$CTMRG_WALLTIME_PILOT -v "PROFILE=pilot,CORES=$CTMRG_CORES,DIMS=$d,RUN=${RUN}_pilot_$d" "$job"; done
        elif [ "$PROFILE" = hpc ]; then
            for c in $CTMRG_CASES; do qsub -N "ctmrg_$c" -l nodes=1:ppn=$CTMRG_CORES -l walltime=$CTMRG_WALLTIME_HPC -v "PROFILE=hpc,CORES=$CTMRG_CORES,CASE=$c,RUN=$RUN" "$job"; done
        elif [ "$PROFILE" != timing ]; then echo "unknown PROFILE $PROFILE (use pilot, timing or hpc)"; exit 1; fi
    elif [ "$alg" = vbpeps ]; then
        i=$VBPEPS_IMPL_RUN; c=$VBPEPS_CORES
        if [ "$PROFILE" = pilot ]; then
            for d in $VBPEPS_DIMS; do
                qsub -N "vbpeps_pilot_$d" -l nodes=1:ppn=$c -l walltime=$VBPEPS_WALLTIME_PILOT -v "PROFILE=pilot,IMPL=$i,CORES=$c,DIMS=$d,RUN=${RUN}_pilot_$d" "$job"
            done
        elif [ "$PROFILE" = timing ]; then
            c=$VBPEPS_TIMING_CORES
            for i in $VBPEPS_IMPLS_TIMING; do
                qsub -N "vbpeps_timing_$i" -l nodes=1:ppn=$c -l walltime=$VBPEPS_WALLTIME_TIMING -v "PROFILE=timing,IMPL=$i,CORES=$c,RUN=${RUN}_timing_$i" "$job"
            done
        elif [ "$PROFILE" = hpc ]; then
            for k in $VBPEPS_CASES; do
                qsub -N "vbpeps_${k}_$i" -l nodes=1:ppn=$c -l walltime=$VBPEPS_WALLTIME_HPC -v "PROFILE=hpc,IMPL=$i,CORES=$c,CASE=$k,RUN=${RUN}_$i" "$job"
            done
        else echo "unknown PROFILE $PROFILE (use pilot, timing or hpc)"; exit 1; fi
    else echo "unknown algorithm $alg (use ctmrg or vbpeps)"; exit 1; fi
done
echo "Submitted ($PROFILE, run $RUN). See the jobs with: qstat"
