#!/bin/bash
# Submits big CTMRG3DOctant runs with P_f from the enlarged edge: octantedge_cold (face_projector = :edge_env), octantedgeplain_cold (:edge),
# octantedgelead_cold (:edge_lead); JOBS chooses which.
# Each job has the settings of the octant job of submit_big_compare.sh (exact PEPO, cold start, the same β, dims, iteration cap and 64 core hours),
# so they compare directly with big_octant.
set -euo pipefail

# ---------------- settings ----------------
BETAS="0.195_0.20_0.205_0.21_0.215_0.22_0.225_0.23_0.25"   # the β of every observation (as in submit_big_compare.sh)
CORES=4                                        # cores of every job
BUDGET_S=57600                                 # time budget per case = 64 core hours on 4 cores (the contraction of every β counts for it)
WALLTIME=18:00:00                              # budget + compilation + the overshoot of one iteration
DIMS_CAP="16_6_2:350"                          # dims (chienv_chipeps_DCE; D_CE = 2 for the exact PEPO):iteration cap, as the octant job
JOBS="octantedgelead:octantedgelead_cold"     # job name:case; also possible: octantedge:octantedge_cold octantedgeplain:octantedgeplain_cold
OCT_STAGNATION=0                               # stagnation window of the octant cases (0 = off), for example 100
# ------------------------------------------

RUN=$(date +%Y%m%d_%H%M)
dims=${DIMS_CAP%:*}
cap=${DIMS_CAP#*:}
for job in $JOBS; do
    name=${job%:*}
    case=${job#*:}
    qsub -N "big_$name" -l nodes=1:ppn=$CORES -l walltime=$WALLTIME \
        -v "PROFILE=pilot_octant,CORES=$CORES,CASE=$case,DIMS=$dims,RUN=${RUN}_big_$name,CE_COMPARE_PEPO=exact,CE_COMPARE_BETAS=$BETAS,CE_COMPARE_MAXITER=$cap,CE_COMPARE_BUDGET_S=$BUDGET_S,CE_COMPARE_VERBOSITY=1,CE_COMPARE_OCTANT_STAGNATION=$OCT_STAGNATION" \
        development/hpc/job_ctmrg_compare.pbs
done
echo "Submitted run ${RUN}_big_* ($JOBS). Follow with: squeue -u \$USER ; grep -h OBS big_octantedge*.o*"
