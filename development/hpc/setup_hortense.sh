#!/bin/bash
# setup_hortense.sh -- ONE-TIME setup on a Hortense LOGIN node: Julia 1.12.7, the clone of milan-work, the BetterPEPSKit clone, the packages of both
# environments and Plots in the default environment (download only).
# The packages are precompiled afterwards in a short interactive job (see the guide), because the login nodes are not meant for long work.
#
# Run it on the login node (after `ssh vscXXXXX@tier1.hpc.ugent.be`):
#   curl -fsSLO https://raw.githubusercontent.com/milanchristiaens/ClusterExpansions_3D/milan-work/development/hpc/setup_hortense.sh
#   bash setup_hortense.sh
set -euo pipefail

PROJECT_DIR="$VSC_SCRATCH_PROJECTS_BASE/starting_2026_079"
cd "$PROJECT_DIR"

# 1. Julia 1.12.7 (the official binary; the same version as on the laptop)
if [ ! -x "$PROJECT_DIR/julia-1.12.7/bin/julia" ]; then
    curl -fsSLO https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.7-linux-x86_64.tar.gz
    tar -xzf julia-1.12.7-linux-x86_64.tar.gz
    rm julia-1.12.7-linux-x86_64.tar.gz
fi

# 2. The clone (branch milan-work)
if [ ! -d "$PROJECT_DIR/ClusterExpansions" ]; then
    git clone -b milan-work https://github.com/milanchristiaens/ClusterExpansions_3D.git ClusterExpansions
fi

# 3. BetterPEPSKit (private repository of Lander Burgelman, for the fast vbPEPS), next to the clone; git asks for the GitHub user name and a token.
#    Without access from here: copy the zip to $PROJECT_DIR with scp, unzip it and rename the folder to BetterPEPSKit.jl (see the guide).
if [ ! -d "$PROJECT_DIR/BetterPEPSKit.jl" ]; then
    git clone https://github.com/leburgel/BetterPEPSKit.jl.git BetterPEPSKit.jl || echo "BetterPEPSKit.jl was not cloned: copy it by hand (see the guide)"
fi

# 4. The environments (Julia, depot) and the packages of their Manifest.toml: download only, no precompilation here
source "$PROJECT_DIR/ClusterExpansions/development/hpc/hortense_env.sh"
mkdir -p "$JULIA_DEPOT_PATH"
cd "$REPO"
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=. -e 'using Pkg; Pkg.instantiate()'
# Plots is not a dependency of ClusterExpansions: the scripts load it from the default environment (as on the laptop), so it goes there (not into the clone)
JULIA_PKG_PRECOMPILE_AUTO=0 julia -e 'using Pkg; Pkg.add("Plots")'
if [ -d "$PROJECT_DIR/BetterPEPSKit.jl" ]; then
    JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=development/fast_vbpeps -e 'using Pkg; Pkg.instantiate()'
fi

julia --version
echo "Setup done: $REPO"
