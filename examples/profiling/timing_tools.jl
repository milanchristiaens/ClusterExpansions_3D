# Helper functions to read the TimerOutput `to` back into rows, print an overview and write a CSV file.
# Used by ising_model_3D_timing.jl (CE/SU + CTMRG3D) and ising_model_3D_timing_vbpeps.jl (CE + variational boundary PEPS).
# (the functions were moved here from ising_model_3D_timing.jl without changes)
# Needs: using TimerOutputs, ClusterExpansions

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
