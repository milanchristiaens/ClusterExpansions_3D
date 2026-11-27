#!/bin/bash
# Submits the big compare jobs at once (CTMRG3D :svd and :oblique, the three CTMRG3DOctant variants and vbPEPS), each with its own dimensions, on the exact PEPO.
# All cases are warm (cold near βc, the warm rule of the compare scripts); vbPEPS also observes the β of VBPEPS_EXTRA_BETAS.
# A job gets its dims, β, iteration cap and time budget as environment variables of the compare scripts.
set -euo pipefail

# ---------------- settings ----------------
BETAS="0.195_0.20_0.205_0.21_0.215_0.22_0.225_0.23_0.24_0.25"   # the β of every observation (increasing)
VBPEPS_BETAS="0.195_0.20_0.205_0.21_0.215_0.22_0.225_0.2275_0.23_0.24_0.25"   # vbPEPS: also 0.2275 (in the exact PEPO file)
CORES_CTMRG=4                                  # cores of the CTMRG3D jobs
CORES_VBPEPS=3                                 # cores of the vbPEPS job
BUDGET_CTMRG_S=64000                           # time budget per case: 57 600 s (64 core hours on 4 cores) for 9 β, times 10/9 for 10 β
BUDGET_VBPEPS_S=93867                          # 76 800 s (64 core hours on 3 cores) for 9 β, times 11/9 for 11 β
WALLTIME_CTMRG=20:00:00                        # budget + compilation + the overshoot of one iteration (only the time really used is charged)
WALLTIME_VBPEPS=40:00:00                       # an observation may go over its share (MIN_OBS_FACTOR), so the case up to about 1.5 × the budget
OCT_EIGENSOLVER=subspace                       # eigensolver of ρ_F of the octant jobs: subspace, block_lanczos, dense_block_lanczos or dense_eigen
# job name:case:dims (chienv_chipeps_DCE; D_CE = 2 for the exact PEPO):cap (CTMRG3D: iterations, vbPEPS: LBFGS iterations)
CTMRG_JOBS="svd:svd_warm_auto:18_6_2:200 oblique:oblique_warm_auto:17_6_2:275 octantunweighted:octantunweighted_warm:16_6_2:350 octantsigmaweighted:octantsigmaweighted_warm:16_6_2:350 octantleadweighted:octantleadweighted_warm:16_6_2:350"
VBPEPS_JOB="vbpeps:false_warm:28_6_2:100"      # vbPEPS (BetterPEPSKit, fixed environment)
# ------------------------------------------

squeue -u "$USER" -h -o %j | grep -qx pepos && { echo "The pepos job is still queued or running: wait until it has finished (the PEPO files need the new β)."; exit 1; }
RUN=$(date +%Y%m%d_%H%M)
common="CE_COMPARE_PEPO=exact"
for job in $CTMRG_JOBS; do
    IFS=: read -r name case dims cap <<< "$job"
    profile=pilot; [[ $name == octant* ]] && profile=pilot_octant
    qsub -N "big_$name" -l nodes=1:ppn=$CORES_CTMRG -l walltime=$WALLTIME_CTMRG \
        -v "PROFILE=$profile,CORES=$CORES_CTMRG,CASE=$case,DIMS=$dims,RUN=${RUN}_big_$name,$common,CE_COMPARE_BETAS=$BETAS,CE_COMPARE_MAXITER=$cap,CE_COMPARE_BUDGET_S=$BUDGET_CTMRG_S,CE_COMPARE_VERBOSITY=1,CE_COMPARE_OCTANT_EIGENSOLVER=$OCT_EIGENSOLVER" \
        development/hpc/job_ctmrg_compare.pbs
done
IFS=: read -r name case dims cap <<< "$VBPEPS_JOB"
qsub -N "big_$name" -l nodes=1:ppn=$CORES_VBPEPS -l walltime=$WALLTIME_VBPEPS \
    -v "PROFILE=pilot,IMPL=betterpepskit,CORES=$CORES_VBPEPS,CASE=$case,DIMS=$dims,RUN=${RUN}_big_$name,$common,CE_COMPARE_BETAS=$VBPEPS_BETAS,CE_COMPARE_MAXITER=$cap,CE_COMPARE_BUDGET_S=$BUDGET_VBPEPS_S" \
    development/hpc/job_vbpeps_compare.pbs
echo "Submitted run ${RUN}_big_*. Follow with: squeue -u \$USER ; grep -h OBS big_*.o*"
