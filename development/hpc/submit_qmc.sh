#!/bin/bash
# Submits the QMC reference of the transverse-field Ising model (tfim_3d_qmc.jl): one job per g, the β of the models of model_pepos.jl, several L.
set -euo pipefail

# ---------------- settings ----------------
LS="8_12_16_24_32"
SWEEPS=20000
THERM=2000
CORES=8
# g:β list
JOBS="2.0:0.205_0.21_0.215_0.22_0.225_0.23_0.235_0.24_0.25_0.27 3.5:0.24_0.25_0.26_0.27_0.275_0.28_0.285_0.29_0.30_0.33 4.5:0.30_0.31_0.32_0.34_0.36_0.37_0.38_0.39_0.41_0.45"
# ------------------------------------------

for j in $JOBS; do
    g=${j%%:*}; b=${j#*:}
    qsub -N "qmc_g$g" -l nodes=1:ppn=$CORES -v "G=$g,LS=$LS,BETAS=$b,SWEEPS=$SWEEPS,THERM=$THERM,CORES=$CORES" development/hpc/job_qmc.pbs
done
echo "Submitted. Results in data/reference/tfim3d_qmc.csv"
