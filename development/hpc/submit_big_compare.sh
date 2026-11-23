#!/bin/bash
# Submits the four big compare jobs (CTMRG3D :svd, :oblique, octant and vbPEPS) at once, each with its own dimensions, on the exact PEPO with cold start.
# A job gets its dims, β, iteration cap and time budget as environment variables of the compare scripts.
set -euo pipefail

# ---------------- settings ----------------
BETAS="0.195_0.20_0.205_0.21_0.215_0.22_0.225_0.23_0.25"   # the β of every observation (multiples of 0.005, all in the PEPO files)
CORES_CTMRG=4                                  # cores of the three CTMRG3D jobs
CORES_VBPEPS=3                                 # cores of the vbPEPS job
BUDGET_CTMRG_S=57600                           # time budget per case = 64 core hours on 4 cores (the contraction of every β counts for it)
BUDGET_VBPEPS_S=76800                          # 64 core hours on 3 cores (an observation may use at most 1.2 × budget / number of β = 10 240 s)
WALLTIME_CTMRG=18:00:00                        # budget + compilation + the overshoot of one iteration (only the time really used is charged)
WALLTIME_VBPEPS=23:00:00
# name:dims (chienv_chipeps_DCE; D_CE = 2 for the exact PEPO):cap (CTMRG3D: iterations, vbPEPS: LBFGS iterations)
SVD="18_6_2:200"                               # CTMRG3D :svd, svd_cold_auto
OBLIQUE="17_6_2:275"                           # CTMRG3D :oblique, oblique_cold_auto (does not converge; the cap sets the cost)
OCTANT="16_6_2:350"                            # CTMRG3DOctant, octant_cold
VBPEPS="32_7_2:100"                            # vbPEPS (BetterPEPSKit, fixed environment), false_cold
# ------------------------------------------

squeue -u "$USER" -h -o %j | grep -qx pepos && { echo "The pepos job is still queued or running: wait until it has finished (the PEPO files need the new β)."; exit 1; }
RUN=$(date +%Y%m%d_%H%M)
common="CE_COMPARE_PEPO=exact,CE_COMPARE_BETAS=$BETAS"
submit_ctmrg() {   # name profile case dims:cap
    local name=$1 profile=$2 case=$3 dims=${4%:*} cap=${4#*:}
    qsub -N "big_$name" -l nodes=1:ppn=$CORES_CTMRG -l walltime=$WALLTIME_CTMRG \
        -v "PROFILE=$profile,CORES=$CORES_CTMRG,CASE=$case,DIMS=$dims,RUN=${RUN}_big_$name,$common,CE_COMPARE_MAXITER=$cap,CE_COMPARE_BUDGET_S=$BUDGET_CTMRG_S,CE_COMPARE_VERBOSITY=1" \
        development/hpc/job_ctmrg_compare.pbs
}
submit_ctmrg svd pilot svd_cold_auto "$SVD"
submit_ctmrg oblique pilot oblique_cold_auto "$OBLIQUE"
submit_ctmrg octant pilot_octant octant_cold "$OCTANT"
qsub -N big_vbpeps -l nodes=1:ppn=$CORES_VBPEPS -l walltime=$WALLTIME_VBPEPS \
    -v "PROFILE=pilot,IMPL=betterpepskit,CORES=$CORES_VBPEPS,CASE=false_cold,DIMS=${VBPEPS%:*},RUN=${RUN}_big_vbpeps,$common,CE_COMPARE_MAXITER=${VBPEPS#*:},CE_COMPARE_BUDGET_S=$BUDGET_VBPEPS_S" \
    development/hpc/job_vbpeps_compare.pbs
echo "Submitted run ${RUN}_big_*. Follow with: squeue -u \$USER ; grep -h OBS big_*.o*"
