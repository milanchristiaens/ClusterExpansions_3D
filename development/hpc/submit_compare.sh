#!/bin/bash
# submit_compare.sh -- submits the Hortense jobs of the compare scripts (job_ctmrg_compare.pbs, job_vbpeps_compare.pbs); round 2 of the Hortense guide is
# PROFILE=hpc (round 1 has its own script, submit_round1.sh, which also submits the pilots and the timing). Run it on the login node,
# in the clone folder, after `module swap cluster/dodrio/cpu_milan_rhel9`, and after the PEPO files exist (job_pepos.pbs):
#   bash development/hpc/submit_compare.sh
#   PROFILE=pilot:  dimension pilot, one job per candidate in CTMRG_DIMS / VBPEPS_DIMS (each in its own run folder <date>_<time>_pilot_<dims>);
#                   vbPEPS with betterpepskit on VBPEPS_CORES cores
#   PROFILE=timing: vbPEPS only, fast against slow at the dims of the full run (HPC_DIMS in the vbPEPS compare script, set them to the choice of the
#                   pilot before): one job per implementation (pepskit, betterpepskit), both on VBPEPS_TIMING_CORES cores (pepskit needs them for its memory)
#   PROFILE=hpc:    full run, one job per case (8 CTMRG3D, 2 vbPEPS), all in parallel and in one run folder <date>_<time> (vbPEPS: <date>_<time>_<impl>);
#                   the dimensions are the dims of the hpc profile in the compare scripts (set them to the choice of the pilot before)
# All settings are below; nothing is given on the command line.
set -euo pipefail

# ---------------- settings ----------------
PROFILE=hpc                                     # pilot, timing or hpc (round 2: hpc)
ALGORITHMS="ctmrg vbpeps"                       # which compare scripts get jobs: ctmrg and/or vbpeps (timing: only vbpeps)
CTMRG_DIMS="16_5_3 16_6_3 18_6_3"               # pilot candidates (chienv_chipeps_DCE) of CTMRG3D; their 3 GiB plans must be timed (job_benchmark_plans.pbs)
VBPEPS_DIMS="27_3_3 32_3_3 36_3_3"              # pilot candidates of vbPEPS
CTMRG_CASES="svd_cold_auto svd_warm_auto oblique_cold_auto oblique_warm_auto svd_cold_w1 svd_warm_w1 oblique_cold_w1 oblique_warm_w1"
VBPEPS_CASES="false_cold false_warm"
VBPEPS_IMPLS_TIMING="pepskit betterpepskit"     # timing: pepskit = PEPSKit 0.8.1 (slow reference), betterpepskit = fast
VBPEPS_IMPL_RUN="betterpepskit"                 # implementation of vbPEPS in the pilot and the full run
VBPEPS_CORES=2                                  # cores of the vbPEPS pilot and full run (3 if the pilot shows that 2 have too little memory; then also
                                                # julia_threads = 3 in the profiles :pilot and :hpc of the vbPEPS compare script, and its budgets)
VBPEPS_TIMING_CORES=4                           # cores of both implementations in the timing
PEPO_D_CE=3                                     # the PEPO file of this D_CE must exist (data/pepos/, job_pepos.pbs)
# walltimes: budget of the profile + compilation + plots, with some margin (only the time really used is charged)
CTMRG_WALLTIME_PILOT=03:30:00                   # 2 cases × 80 min + compilation
CTMRG_WALLTIME_HPC=17:30:00                     # 16 h + compilation
VBPEPS_WALLTIME_PILOT=03:00:00                  # 2 cases × 1 h + compilation (betterpepskit: no Zygote compilation of the CTMRG)
VBPEPS_WALLTIME_TIMING=04:30:00                 # 2 cases × 30 min + compilation (pepskit: about 2 h for `true`)
VBPEPS_WALLTIME_HPC=34:00:00                    # 32 h + compilation
# ------------------------------------------

ls data/pepos/ising_3D_*_D${PEPO_D_CE}_*.jld2 > /dev/null 2>&1 || { echo "No PEPO file of D_CE = $PEPO_D_CE in data/pepos: run qsub development/hpc/job_pepos.pbs first"; exit 1; }
RUN=$(date +%Y%m%d_%H%M)
for alg in $ALGORITHMS; do
    job="development/hpc/job_${alg}_compare.pbs"
    if [ "$alg" = ctmrg ]; then
        if [ "$PROFILE" = pilot ]; then
            for d in $CTMRG_DIMS; do qsub -N "ctmrg_pilot_$d" -l walltime=$CTMRG_WALLTIME_PILOT -v "PROFILE=pilot,DIMS=$d,RUN=${RUN}_pilot_$d" "$job"; done
        elif [ "$PROFILE" = hpc ]; then
            for c in $CTMRG_CASES; do qsub -N "ctmrg_$c" -l walltime=$CTMRG_WALLTIME_HPC -v "PROFILE=hpc,CASE=$c,RUN=$RUN" "$job"; done
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
