#!/bin/bash
# Submits the BP comparison (models_3D_CE_bp_compare.jl) and its vbPEPS part (models_3D_CE_vbpeps_compare.jl): one job per model and case, cold start
# at every β, 64 core hours each. Needs the model PEPO files (job_models_pepos.pbs).
# bh_a only gets the :svd cases and no vbPEPS (its O is not cube symmetric, see model_pepos.jl).
set -euo pipefail

# ---------------- settings ----------------
CORES=4                                        # an octant case that runs out of memory goes again on 5 cores
WALLTIME=18:00:00                              # 64 core hours on 4 cores = 16 h, plus compilation
MAXITER=300
# model:dims (chienv_chipeps_DCE; D_CE of the model):cases (separated by +)
# svdclass_P0 (P = 0 with class projectors) next to every svd_P0: svd_P0 did not converge (no progress in ϵ)
ALL_CASES="svd_P05+svd_P0+svdclass_P0+octantunweighted+octantsigmaweighted+octantleadweighted"
JOBS="tfim_g2:16_5_3:$ALL_CASES tfim_g3.5:16_5_3:$ALL_CASES tfim_g4.5:16_5_3:$ALL_CASES bh_b:16_5_3:$ALL_CASES bh_a:12_4_4:svd_P05+svd_P0+svdclass_P0"
# vbPEPS (BetterPEPSKit, fixed environment, cold start at every β): model:dims; walltime 1.5 × the budget of 16 h (every β gets at least half its fair share) plus compilation
VBPEPS_JOBS="tfim_g2:16_5_3 tfim_g3.5:16_5_3 tfim_g4.5:16_5_3 bh_b:16_5_3"
WALLTIME_VBPEPS=26:00:00
# ------------------------------------------

RUN=$(date +%Y%m%d_%H%M)
for j in $JOBS; do
    IFS=: read -r model dims cases <<< "$j"
    for c in ${cases//+/ }; do
        qsub -N "m_${model}_$c" -l nodes=1:ppn=$CORES -l walltime=$WALLTIME \
            -v "MODEL=$model,DIMS=$dims,CASE=$c,CORES=$CORES,RUN=${RUN}_${model},CE_COMPARE_MAXITER=$MAXITER" development/hpc/job_models_compare.pbs
    done
done
for j in $VBPEPS_JOBS; do
    IFS=: read -r model dims <<< "$j"
    qsub -N "v_${model}" -l nodes=1:ppn=$CORES -l walltime=$WALLTIME_VBPEPS \
        -v "MODEL=$model,DIMS=$dims,CASE=vbpeps_cold,CORES=$CORES,RUN=${RUN}_${model}" development/hpc/job_models_vbpeps.pbs
done
echo "Submitted run ${RUN}_*. Follow with: squeue -u \$USER ; grep -h OBS m_*.o* v_*.o*"
