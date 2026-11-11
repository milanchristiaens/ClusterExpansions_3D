# Make the timing plots from the CSV files of ../ising_model_3D_CE_CTMRG_profiling.jl and
# ../ising_model_3D_CE_vbPEPS_profiling.jl, without rerunning the simulation.

include(joinpath(@__DIR__, "timing_plots.jl"))

csvdir = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "profiling"))  # <repo>/data/profiling, where the profiling scripts write

# The runs we know: (tag used for file names, label used for plot titles). Only the
# runs whose CSV file exists are plotted.
runs = [("CE", "CE + CTMRG3D"), ("SU", "SU + CTMRG3D"), ("CE_vbpeps", "CE + vbPEPS")]
filter!(r -> isfile(joinpath(csvdir, "timing_$(r[1]).csv")), runs)
isempty(runs) && error("no timing_*.csv in $csvdir: run ising_model_3D_CE_CTMRG_profiling.jl or
                        ising_model_3D_CE_vbPEPS_profiling.jl first")

# Read the CSV files first
all_rows = Dict(tag => read_timer_csv(joinpath(csvdir, "timing_$tag.csv")) for (tag, _)
                in runs)
# Loop over the runs: CE pair, SU pair, CE + vbPEPS pair
for (tag, label) in runs
    rows = all_rows[tag]

    plt_phases = plot_phases(rows, label)                                         # the main phases of the run
    plt_self = plot_self_time(rows, label; top = 10)                              # top 10 sections by self time
    # plt_ctm = plot_self_time(rows, label; under = "leading_boundary", top = 25) # only inside the CTMRG3D iterations
    plt_mean_alloc = plot_alloc_per_call(rows, label; top = 10)                   # top sections by mean self allocation per call
    # plt_mean_alloc_ctm = plot_alloc_per_call(rows, label; under = "leading_boundary")
    plt_max_alloc = plot_max_alloc_per_call(rows, label; top = 10)                # largest allocation of one call
    # plt_max_alloc_ctm = plot_max_alloc_per_call(rows, label; under = "leading_boundary")

    display(plt_phases)
    display(plt_self)
    # display(plt_ctm)
    display(plt_mean_alloc)
    # display(plt_mean_alloc_ctm)
    display(plt_max_alloc)
    # display(plt_max_alloc_ctm)

    savefig(plt_phases, joinpath(csvdir, "timing_$(tag)_phases.png"))
    savefig(plt_self, joinpath(csvdir, "timing_$(tag)_self.png"))
    # savefig(plt_ctm, joinpath(csvdir, "timing_$(tag)_ctmrg.png"))
    savefig(plt_mean_alloc, joinpath(csvdir, "timing_$(tag)_alloc_per_call.png"))
    # savefig(plt_mean_alloc_ctm, joinpath(csvdir, "timing_$(tag)_alloc_per_call_ctmrg.png"))
    savefig(plt_max_alloc, joinpath(csvdir, "timing_$(tag)_alloc_max_per_call.png"))
    # savefig(plt_max_alloc_ctm, joinpath(csvdir, "timing_$(tag)_alloc_max_per_call_ctmrg.png"))
end