# timing_io.jl -- the data side of the timing scripts: TimerOutput -> rows -> CSV file -> rows, and where the data of a script goes.
# Used by the profiling scripts (ising_model_3D_CE_CTMRG_profiling.jl, ising_model_3D_CE_vbPEPS_profiling.jl) and by comparing/ising_model_3D_CE_vbPEPS_compare.jl.
# Needs: using TimerOutputs, ClusterExpansions (and the three folder functions below need nothing else)
# The plots are in timing_plots.jl, the helpers of the compare scripts in compare_tools.jl.

## Where the data of a script goes: data/profiling/<same subfolders as development/profiling>/<script name>/<run>/
# Every profiling script has ONE folder in data/profiling, at the same place as the script has in development/profiling:
#   development/profiling/ising_model_3D_CE_CTMRG_profiling.jl          ->  data/profiling/ising_model_3D_CE_CTMRG_profiling/
#   development/profiling/comparing/ising_model_3D_CE_vbPEPS_compare.jl ->  data/profiling/comparing/ising_model_3D_CE_vbPEPS_compare/
# Every run of the script is a subfolder of that folder, named by the date and time of the run (for example 20261111_1248), with all files of that run in it
# (CSV files, the 5 plots, ...). REPLOT_DIR in a script is the name of such a run folder.
const PROFILING_SCRIPTS_DIR = normpath(joinpath(@__DIR__, ".."))                                  # <repo>/development/profiling
const PROFILING_DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "profiling"))    # <repo>/data/profiling

# The data folder of a script. script_file is @__FILE__ of the script (the script must be inside development/profiling).
function script_data_dir(script_file)
    rel = relpath(abspath(script_file), PROFILING_SCRIPTS_DIR)      # for example comparing/ising_model_3D_CE_vbPEPS_compare.jl
    startswith(rel, "..") && error("$script_file is not inside $PROFILING_SCRIPTS_DIR")
    parts = splitpath(rel)
    parts[end] = splitext(parts[end])[1]                            # without .jl
    return joinpath(PROFILING_DATA_DIR, parts...)
end

# A new run folder (date and time of now) of the script; it is created.
function new_run_dir(script_file)
    dir = joinpath(script_data_dir(script_file), Libc.strftime("%Y%m%d_%H%M", time()))
    mkpath(dir)
    return dir
end

# The run folder with this name (the REPLOT_DIR of the script); an error that lists the existing run folders if there is none.
function existing_run_dir(script_file, name)
    base = script_data_dir(script_file)
    dir = joinpath(base, name)
    isdir(dir) && return dir
    runs = isdir(base) ? sort(filter(d -> isdir(joinpath(base, d)), readdir(base))) : String[]
    error("REPLOT_DIR = \"$name\": there is no folder $dir. Run folders of this script: " * (isempty(runs) ? "none" : join(runs, ", ")))
end

## Reading the TimerOutput back (TimerOutputs only prints it, so we extract our own rows)
# TimerOutputs.todict(to) returns nested dictionaries, one per section:
# Dict("n_calls" => ..., "time_ns" => ..., "allocated_bytes" => ..., "inner_timers" => Dict("child name" => Dict(same keys...), ...))
# Now we want to turn this into rows

# Depth-first rows, siblings sorted by decreasing total time. Times in seconds, memory in MiB.
# `self` is the time not covered by any child section. Uses `TimerOutputs.todict`.
# e.g. 
# TOTAL (10 s)
# ├─ A (6 s)
# │   ├─ A1 (4 s)
# │   └─ A2 (1 s)
# └─ B (3 s)
#     └─ B1 (2 s)
# Then we get: TOTAL, A, A1, A2, B, B1
function timer_rows(to::TimerOutput)
    # Start empty list, each section will become one named tuple in it
    rows = NamedTuple[]
    
    # Small helper function: Calculates the total time (ns) of all direct children of section d, if no children = 0
    inner_ns(d) = sum((c["time_ns"] for c in values(d["inner_timers"])); init = 0)
    
    # Recursive helper that visits all children of d
    function walk(d, prefix, depth, parent_ns)
        # d = studied parent, prefix = path so far, depth is the level (0=top)
        # parent_ns = total parent's time

        # Now we loop over all children, in order of first the longest child time to lowest child time
        for (name, c) in sort(collect(d["inner_timers"]); by = kv -> kv[2]["time_ns"], rev = true)
            # collect() : turns children dictionary into a list of: (name child, child data) pairs
            # name = child name and c is the child's data

            # This creates a string corresponding to the full path: e.g. "TOTAL CE run > expectation_value > ..."
            path = isempty(prefix) ? name : prefix * " > " * name
            
            # We add to rows a named tuple: (path = path, name = name, depth = depth, ...)
            push!(rows, (;
                path, name, depth,
                ncalls = c["n_calls"],          # Number of calls
                total = c["time_ns"] / 1.0e9,                   # total time of child
                self = (c["time_ns"] - inner_ns(c)) / 1.0e9,    # total self time of child
                alloc_MiB = c["allocated_bytes"] / 2^20,        # Allocated memory in MiB (2^20 Bytes)
                pct_parent = parent_ns > 0 ? 100 * c["time_ns"] / parent_ns : NaN, # Share of parent's time in %
                max_alloc_MiB = get(ClusterExpansions.MAXALLOC, name, NaN) / 2^20, # Largest allocation of ONE call of a section with this name (MiB)
            ))

            # Make it resursive, we visit this section's own children, one level deeper, with this section as new parent
            # With the updated path/prefix
            walk(c, path, depth + 1, c["time_ns"])
        end
    end

    # Now we convert the whole time to dictionaries, start the walk at root with ampty prefix and depth 0
    top = TimerOutputs.todict(to)
    # We have to pass a parent_ns, but top doesn't really have a parent, so it shouldn't have self time
    # So we can simply use time of children inner_ns(top)
    walk(top, "", 0, inner_ns(top))
    return rows
end

# Compact overview of the first levels of the tree (print_timer itself has no depth limit)
function print_overview(rows; maxdepth = 2)
    # Print a header
    println(rpad("section", 70), lpad("calls", 8), lpad("total [s]", 11), lpad("self [s]", 10), lpad("% parent", 9), lpad("alloc [MiB]", 13))
    println("-"^121)

    # Loop over all rows that are not deeper than maxdepth
    for r in rows
        r.depth > maxdepth && continue

        # Print one row, indented via ("  "^r.depth)
        println(
            rpad("  "^r.depth * r.name, 70), lpad(r.ncalls, 8), lpad(round(r.total; digits = 3), 11),
            lpad(round(r.self; digits = 3), 10), lpad(round(r.pct_parent; digits = 1), 9),
            lpad(round(r.alloc_MiB; digits = 1), 13)
        )
    end
    return nothing
end


function write_timer_csv(path, rows)
    # Open file for writing at path
    open(path, "w") do io
        # Header line
        println(io, "path,depth,ncalls,total_s,self_s,alloc_MiB,pct_parent,max_alloc_MiB")        # Print one line per row, sometimes we use " ", because section names can contain commas or >
        # Comma's in between
        for r in rows
            println(io, '"', r.path, '"', ',', r.depth, ',', r.ncalls, ',', r.total, ',', r.self, ',', r.alloc_MiB, ',', r.pct_parent, ',', r.max_alloc_MiB)        
        end
    end
    return path
end

# Read a CSV written by write_timer_csv back into rows (same fields as timer_rows gives)
function read_timer_csv(path)
    # Create empty list that will hold one named tuple per CSV line
    rows = NamedTuple[]
    
    # Loop over lines in the CSV file
    for (i, line) in enumerate(eachline(path))
        (i == 1 || isempty(strip(line))) && continue   # skip the header and empty lines
        
        # e.g. One line of CSV: "TOTAL CE run > expectation_value (CTMRG3D and observable)",1,5,60.25,0.01,99.0,26.0
        # m then splits it as follows: (.*) is group 1 followed by (.*) group 2
        # m.captures[1]   # "TOTAL CE run > expectation_value (CTMRG3D and observable)"
        # m.captures[2]   # "1,5,60.25,0.01,99.0,26.0"
        m = match(r"^\"(.*)\",(.*)$", line)             # the path is in quotes, because it can contain commas
        m === nothing && error("Cannot parse line $i of $path: $line")
        
        # Now we split all numbers up between ','
        f = split(m.captures[2], ',')
        length(f) in (6, 7) || error("Expected 6 or 7 numbers after the path in line $i of $path, got $(length(f)): $line")        # Take the path text and turn into string
        p = String(m.captures[1])
        
        # Now put this data into a row
        push!(rows, (;
            path = p,
            name = String(last(split(p, " > "))),       # the last part of the path is the section name
            depth = parse(Int, f[1]),
            ncalls = parse(Int, f[2]),
            total = parse(Float64, f[3]),
            self = parse(Float64, f[4]),
            alloc_MiB = parse(Float64, f[5]),
            pct_parent = parse(Float64, f[6]),
            max_alloc_MiB = length(f) == 7 ? parse(Float64, f[7]) : NaN,   # old CSV files without this column give NaN
        ))
    end
    return rows
end
