#!/bin/bash
# submit_round1.sh -- submits ALL jobs of round 1b of the Hortense guide at once (they run in parallel). Run it on the login node, in the clone folder,
# after `module swap cluster/dodrio/cpu_milan_rhel9` and after round 1a (job_pepos.pbs) has finished:
#   bash development/hpc/submit_round1.sh
# The jobs:
#   plans          job_plans.pbs: plan validation (mock, real) and the full plan benchmark, one after the other (they write the same plan file)
#   check_init     job_check_ctmrg3d.pbs CHECK=init      (CTMRG3D cold starts: :ones, :random, :nishino_okunishi)
#   check_exponent job_check_ctmrg3d.pbs CHECK=exponent  (CTMRG3D projector exponent P = 0, 1/2, 1)
#   check_ad       job_check_vbpeps_diff_env.pbs         (vbPEPS: fixed against differentiated environment)
#   decompositions job_decompositions.pbs                (dense against block-Krylov, time per CTMRG step)
#   timing         job_vbpeps_compare.pbs PROFILE=timing (Sander's vbPEPS (PEPSKit) against Lander's (BetterPEPSKit) at (27,3,3), 4 cores each)
#   pilots         job_ctmrg_compare.pbs and job_vbpeps_compare.pbs PROFILE=pilot (one job per candidate)
# All settings are below; nothing is given on the command line. Set a line of JOBS to "no" to leave that job out.
set -euo pipefail

# ---------------- settings ----------------
JOBS_PLANS=yes
JOBS_CHECKS=yes                                 # check_init, check_exponent, check_ad
JOBS_DECOMPOSITIONS=yes
JOBS_TIMING=yes
JOBS_PILOTS=yes
CTMRG_DIMS="16_5_3 16_6_3 18_6_3"               # pilot candidates of CTMRG3D
VBPEPS_DIMS="27_3_3 32_3_3 36_3_3"              # pilot candidates of vbPEPS (Lander's)
VBPEPS_CORES=2                                  # cores of the vbPEPS pilots (as the full run)
TIMING_CORES=4                                  # cores of both implementations in the timing (PEPSKit needs them for its memory)
PEPO_D_CE=3                                     # the PEPO file of this D_CE must exist (data/pepos/, job_pepos.pbs)
# walltimes: estimate + compilation, with margin (only the time really used is charged)
CTMRG_WALLTIME_PILOT=03:30:00                   # 2 cases × 80 min + compilation
VBPEPS_WALLTIME_PILOT=02:00:00                  # 1 case × 1 h + compilation
TIMING_WALLTIME=04:30:00                        # 30 min + compilation (PEPSKit)
CHECK_INIT_WALLTIME=30:00:00
CHECK_EXPONENT_WALLTIME=12:00:00
# ------------------------------------------

ls data/pepos/ising_3D_*_D${PEPO_D_CE}_*.jld2 > /dev/null 2>&1 || { echo "No PEPO file of D_CE = $PEPO_D_CE in data/pepos: run qsub development/hpc/job_pepos.pbs first (round 1a)"; exit 1; }
RUN=$(date +%Y%m%d_%H%M)
H=development/hpc
[ "$JOBS_PLANS" = yes ] && qsub "$H/job_plans.pbs"
if [ "$JOBS_CHECKS" = yes ]; then
    qsub -N check_init -l walltime=$CHECK_INIT_WALLTIME -v "CHECK=init" "$H/job_check_ctmrg3d.pbs"
    qsub -N check_exponent -l walltime=$CHECK_EXPONENT_WALLTIME -v "CHECK=exponent" "$H/job_check_ctmrg3d.pbs"
    qsub "$H/job_check_vbpeps_diff_env.pbs"
fi
[ "$JOBS_DECOMPOSITIONS" = yes ] && qsub "$H/job_decompositions.pbs"
if [ "$JOBS_TIMING" = yes ]; then
    for i in pepskit betterpepskit; do
        qsub -N "vbpeps_timing_$i" -l nodes=1:ppn=$TIMING_CORES -l walltime=$TIMING_WALLTIME -v "PROFILE=timing,IMPL=$i,CORES=$TIMING_CORES,RUN=${RUN}_timing_$i" "$H/job_vbpeps_compare.pbs"
    done
fi
if [ "$JOBS_PILOTS" = yes ]; then
    for d in $CTMRG_DIMS; do qsub -N "ctmrg_pilot_$d" -l walltime=$CTMRG_WALLTIME_PILOT -v "PROFILE=pilot,DIMS=$d,RUN=${RUN}_pilot_$d" "$H/job_ctmrg_compare.pbs"; done
    for d in $VBPEPS_DIMS; do
        qsub -N "vbpeps_pilot_$d" -l nodes=1:ppn=$VBPEPS_CORES -l walltime=$VBPEPS_WALLTIME_PILOT -v "PROFILE=pilot,IMPL=betterpepskit,CORES=$VBPEPS_CORES,DIMS=$d,RUN=${RUN}_pilot_$d" "$H/job_vbpeps_compare.pbs"
    done
fi
echo "Round 1b submitted (run $RUN). See the jobs with: qstat"
