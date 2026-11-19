# development/fast_vbpeps: the BetterPEPSKit environment of the fast vbPEPS

`fast_bpeps.jl` is the vbPEPS of ClusterExpansions (`src/bpeps/`) on BetterPEPSKit (Lander Burgelman e.a., not registered; see the top of
`fast_bpeps.jl` for what is faster). BetterPEPSKit needs MPSKit 0.14 and OptimKit 0.5 and ClusterExpansions MPSKit 0.13 and OptimKit 0.4, so this
environment runs WITHOUT ClusterExpansions: the vbPEPS compare (`development/profiling/comparing/ising_model_3D_CE_vbPEPS_compare.jl`,
`VBPEPS_IMPL = :betterpepskit`) and the dense against block-Krylov benchmark (`ising_model_3D_CE_vbPEPS_decompositions.jl`) are run in it and read the
PEPOs from the PEPO files that ClusterExpansions made first (`ising_model_3D_CE_pepos.jl`, in the environment of the clone), for example

    julia +1.12 -t 4 --project=development/fast_vbpeps development/profiling/comparing/ising_model_3D_CE_vbPEPS_compare.jl

This environment has no Plots: the plots are made afterwards with REPLOT_DIR in the environment of the clone. Its packages: BetterPEPSKit and the
vendored packages, TimerOutputs, and JLD2 (the PEPO files).

The environment needs the BetterPEPSKit clone NEXT TO this clone (`Manifest.toml` refers to `../../../BetterPEPSKit.jl` and its `vendor/` folder):

    ...\Code\3D\ClusterExpansions      (this repository)
    ...\Code\3D\BetterPEPSKit.jl       (git clone of leburgel/BetterPEPSKit.jl, or its zip unpacked and the folder renamed to BetterPEPSKit.jl)

Install once (and again after a change of `Manifest.toml`), in PowerShell in the root folder of this clone:

    julia +1.12 --project=development/fast_vbpeps -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"

This only touches this environment, not the environment of ClusterExpansions. `Manifest.toml` fixes the versions of all packages (made with Julia 1.12.7).
