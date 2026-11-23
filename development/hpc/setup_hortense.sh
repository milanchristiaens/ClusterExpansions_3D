#!/bin/bash
# One-time setup on a Hortense login node: Julia, the clone of milan-work, the BetterPEPSKit clone and the packages of both environments (download only).
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

# 3. BetterPEPSKit (private repository of Lander Burgelman), next to the clone; git asks for a user name and a token
if [ ! -d "$PROJECT_DIR/BetterPEPSKit.jl" ]; then
    git clone https://github.com/leburgel/BetterPEPSKit.jl.git BetterPEPSKit.jl || echo "BetterPEPSKit.jl was not cloned: copy it by hand (see the guide)"
fi

# 4. The environments (Julia, depot) and the packages of their Manifest.toml: download only, no precompilation here
source "$PROJECT_DIR/ClusterExpansions/development/hpc/hortense_env.sh"
mkdir -p "$JULIA_DEPOT_PATH"
cd "$REPO"
JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=. -e 'using Pkg; Pkg.instantiate()'
# Plots is not a dependency of ClusterExpansions: the scripts load it from the default environment
JULIA_PKG_PRECOMPILE_AUTO=0 julia -e 'using Pkg; Pkg.add("Plots")'
if [ -d "$PROJECT_DIR/BetterPEPSKit.jl" ]; then
    JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=development/fast_vbpeps -e 'using Pkg; Pkg.instantiate()'
fi

julia --version
echo "Setup done: $REPO"
