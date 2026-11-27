#!/bin/bash
# Submits the bond-dimension scan: every method of the big compare (submit_big_compare.sh) at seven smaller sets of (χenv, χpeps), with the same β,
# warm rule, iteration caps, cores and budget as its top set (exact PEPO, D_CE = 2). Run names end in _scan_<method>_<dims>.
# The sets follow the cost table of the guide (estimated cost of the top set: about 1/2, 1/4, ..., down to 1/100 or less).
set -euo pipefail

# ---------------- settings ----------------
BETAS="0.195_0.20_0.205_0.21_0.215_0.22_0.225_0.23_0.24_0.25"
VBPEPS_BETAS="0.195_0.20_0.205_0.21_0.215_0.22_0.225_0.2275_0.23_0.24_0.25"
CORES_CTMRG=4
CORES_VBPEPS=3
BUDGET_CTMRG_S=64000                           # the budgets of the top sets: the scan jobs need much less, they are only charged for what they use
BUDGET_VBPEPS_S=93867
WALLTIME_CTMRG=20:00:00
WALLTIME_VBPEPS=40:00:00
OCT_EIGENSOLVER=subspace
# method:case:cap:sets (sets chienv_chipeps_DCE separated by +); CTMRG3D: 2 <= χenv/χpeps <= 3; vbPEPS: χenv >= 0.64 χpeps²
SVD="svd:svd_warm_auto:200:16_6_2+14_6_2+15_5_2+14_5_2+12_5_2+12_4_2+8_3_2"
OBLIQUE="oblique:oblique_warm_auto:275:14_6_2+14_5_2+12_5_2+12_4_2+10_4_2+9_3_2+6_2_2"
OCTANT_SETS="14_6_2+14_5_2+12_5_2+12_4_2+10_4_2+8_3_2+6_2_2"
OCTANTS="octantunweighted:octantunweighted_warm:350:$OCTANT_SETS octantsigmaweighted:octantsigmaweighted_warm:350:$OCTANT_SETS octantleadweighted:octantleadweighted_warm:350:$OCTANT_SETS"
VBPEPS="vbpeps:false_warm:100:28_5_2+22_5_2+17_5_2+20_4_2+16_4_2+12_4_2+8_3_2"
METHODS="$SVD $OBLIQUE $OCTANTS $VBPEPS"       # leave out a method to skip it
# ------------------------------------------

RUN=$(date +%Y%m%d_%H%M)
n=0
for m in $METHODS; do
    IFS=: read -r name case cap sets <<< "$m"
    for dims in ${sets//+/ }; do
        run="${RUN}_scan_${name}_${dims}"
        if [ "$name" = vbpeps ]; then
            qsub -N "scan_${name}_$dims" -l nodes=1:ppn=$CORES_VBPEPS -l walltime=$WALLTIME_VBPEPS \
                -v "PROFILE=pilot,IMPL=betterpepskit,CORES=$CORES_VBPEPS,CASE=$case,DIMS=$dims,RUN=$run,CE_COMPARE_PEPO=exact,CE_COMPARE_BETAS=$VBPEPS_BETAS,CE_COMPARE_MAXITER=$cap,CE_COMPARE_BUDGET_S=$BUDGET_VBPEPS_S" \
                development/hpc/job_vbpeps_compare.pbs
        else
            profile=pilot; [[ $name == octant* ]] && profile=pilot_octant
            qsub -N "scan_${name}_$dims" -l nodes=1:ppn=$CORES_CTMRG -l walltime=$WALLTIME_CTMRG \
                -v "PROFILE=$profile,CORES=$CORES_CTMRG,CASE=$case,DIMS=$dims,RUN=$run,CE_COMPARE_PEPO=exact,CE_COMPARE_BETAS=$BETAS,CE_COMPARE_MAXITER=$cap,CE_COMPARE_BUDGET_S=$BUDGET_CTMRG_S,CE_COMPARE_VERBOSITY=1,CE_COMPARE_OCTANT_EIGENSOLVER=$OCT_EIGENSOLVER" \
                development/hpc/job_ctmrg_compare.pbs
        fi
        n=$((n + 1))
    done
done
echo "Submitted $n scan jobs of run ${RUN}_scan_*. Follow with: squeue -u \$USER"
