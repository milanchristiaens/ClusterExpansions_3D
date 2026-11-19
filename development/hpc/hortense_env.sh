# hortense_env.sh -- the environment of all our Hortense jobs (VSC Tier-1, partition cpu_milan_rhel9). Sourced by setup_hortense.sh and the job scripts.
# Everything lives in the project scratch (2 TB): Julia, the package depot and the clone. $HOME on Hortense is only 3 GB.

export PROJECT=starting_2026_079
export PROJECT_DIR="$VSC_SCRATCH_PROJECTS_BASE/$PROJECT"
export JULIA_VERSION=1.12.7
export JULIA_DIR="$PROJECT_DIR/julia-$JULIA_VERSION"
export JULIA_DEPOT_PATH="$PROJECT_DIR/julia_depot"
export REPO="$PROJECT_DIR/ClusterExpansions"
export PATH="$JULIA_DIR/bin:$PATH"

# Plots without a display
export GKSwstype=nul
# BLAS threads before Julia sets them itself (OpenBLAS would otherwise start one thread per core of the WHOLE node, 128); the scripts set their own number
export OPENBLAS_NUM_THREADS=4
