# Environment of all Hortense jobs: Julia, the package depot and the clone live in the project scratch, because $HOME is only 3 GB.
# Sourced by setup_hortense.sh and the job scripts.

export PROJECT=starting_2026_079
export PROJECT_DIR="$VSC_SCRATCH_PROJECTS_BASE/$PROJECT"
export JULIA_VERSION=1.12.7
export JULIA_DIR="$PROJECT_DIR/julia-$JULIA_VERSION"
export JULIA_DEPOT_PATH="$PROJECT_DIR/julia_depot"
export REPO="$PROJECT_DIR/ClusterExpansions"
export PATH="$JULIA_DIR/bin:$PATH"

# Plots without a display
export GKSwstype=nul
# BLAS threads, set before Julia sets its own (OpenBLAS would otherwise start one per core of the whole node)
export OPENBLAS_NUM_THREADS=4
