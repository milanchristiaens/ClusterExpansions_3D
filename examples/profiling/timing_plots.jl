# Plot helpers for the timing rows of ising_model_3D_timing.jl.
#
# They work on "rows": named tuples with the fields path, name, depth, ncalls, total, self, alloc_MiB, pct_parent,
# either straight from timer_rows(to) or read back from a CSV file with read_timer_csv (no rerun needed).

using Plots
using Plots.PlotMeasures # for `mm`, used for plot margins

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

# Shorten a long section name so that it fits next to the bars
shorten(s, n = 55) = length(s) <= n ? s : first(s, n - 1) * "…"

# Sections with the same name are merged over the whole tree: (name, ncalls, self time), sorted by decreasing self time
function self_time_by_name(rows)
    acc = Dict{String, Tuple{Int, Float64}}()
    for r in rows
        # n = row's calls and s = self time
        n, s = get(acc, r.name, (0, 0.0))
        acc[r.name] = (n + r.ncalls, s + r.self)
    end
    # Turn Dict into list of named tuples (name, ncalls, self)
    out = [(; name = k, ncalls = v[1], self = v[2]) for (k, v) in acc]
    # Sort by largest self time first
    sort!(out; by = x -> x.self, rev = true)
    return out
end

# Horizontal bars with the first entry at the top. labels[i] is written to the right of bar i.
function hbars(names, values, labels; title = "", xlabel = "", color = :steelblue, figsize = (1400, 800))
    # Number of bars
    n = length(names)
    # Find largest value 
    xmax = max(maximum(values), 1e-9)
    # Find longest name 
    maxlen = maximum(length, names)

    # Create list of (x, y , text) tuples
    # Text for bar i starts at values[i] + 0.01 * xmax, so a bit right to where the bar ends
    # The text has font size 8, left aligned
    anns = [(values[i] + 0.01 * xmax, i, Plots.text(labels[i], 8, :left)) for i in 1:n]
    
    # Create an empty plot with following settigns
    plt = plot(;
        title, xlabel, legend = false, size = figsize,
        yticks = (1:n, names), ylims = (0.4, n + 0.6), yflip = true,   # yflip: entry 1 at the top
        xlims = (0, 1.4 * xmax),                                       # room on the right for the labels
        grid = :x,                                                     # Only keep vertical grid lines
        left_margin = (1.8 * maxlen + 4) * mm,                         # room on the left for the names
        bottom_margin = 8mm, ytickfontsize = 9,
        annotations = anns,
    )

    # Now add all bars, one after the other
    for i in 1:n
        plot!(plt, Plots.Shape([0, values[i], values[i], 0], [i - 0.4, i - 0.4, i + 0.4, i + 0.4]); fillcolor = color, linecolor = color)
    end
    return plt
end

# Time of the whole run = sum of the sections at depth 0
total_time(rows) = sum(r.total for r in rows if r.depth == 0)

# The `top` sections with the largest self time. Same-named sections (e.g. rot_ns_90 under many parents) are merged.
# under = "text": only sections whose path contains "text", e.g. under = "leading_boundary" for the CTMRG3D iterations.
# The percentage in brackets is relative to the whole run, followed by the number of calls.
function plot_self_time(rows, label; top = 20, under = "")
    # if under is empty, then sel is all rows, 
    # otherwise only keep rows r whose path contains that text occursin(under, r.path)
    sel = isempty(under) ? rows : filter(r -> occursin(under, r.path), rows)
    isempty(sel) && error("No sections with \"$under\" in their path")

    # Time of the whole run not only of the selection
    grand = total_time(rows)

    # Merge same named sections and only keep the first top entries
    flat = self_time_by_name(sel)
    flat = flat[1:min(top, length(flat))]

    # Shorten the nales
    names = [shorten(r.name) for r in flat]

    # Build label for each bar: e.g. 12.3 s (5.1%)  40x
    labels = ["$(round(r.self; sigdigits = 3)) s ($(round(100 * r.self / grand; digits = 1))%)  $(r.ncalls)x" for r in flat]
    
    # Build title
    ttl = "$label: top $(length(flat)) sections by self time" * (isempty(under) ? "" : ", inside \"$under\"")
    
    # Create the plot
    return hbars(names, [r.self for r in flat], labels; title = ttl,
        xlabel = "self time [s]   (percentage of the whole run in brackets, then number of calls)")
end

# The direct parts of the total run (depth 1), by total time. A last bar shows time that is in no depth-1 section.
function plot_phases(rows, label)
    # Take all rows at depth 1, sort them by total time (largest first)
    phases = sort(filter(r -> r.depth == 1, rows); by = r -> r.total, rev = true)
    
    # Define time of whole run
    grand = total_time(rows)
    names = [shorten(r.name) for r in phases]
    values = [r.total for r in phases]
    calls = [r.ncalls for r in phases]

    # Calculate the time that is none of the above sections
    rest = grand - sum(values)

    # If this time is bigger than 1%, add extra bar
    if rest > 0.01 * grand
        push!(names, "(rest: code outside the sections above)")
        push!(values, rest)
        push!(calls, 1)
    end

    # Build one label per bar 
    labels = ["$(round(values[i]; sigdigits = 3)) s ($(round(100 * values[i] / grand; digits = 1))%)  $(calls[i])x" for i in eachindex(values)]
    
    # create the plot
    return hbars(names, values, labels; title = "$label: main phases of the run",
        xlabel = "total time [s]   (percentage of the whole run in brackets, then number of calls)", figsize = (1400, 450))
end


# Memory plots: self allocation, categories of operations, time versus allocation

# alloc_MiB in the rows is INCLUSIVE (the section plus all its children).
# Self allocation = own allocation minus the allocation of the direct children (analogous to self time).
# The rows are in depth-first order (parent, then its children), so the parent of a row is the most recent row one level up.
function add_self_alloc(rows)
    kids = zeros(length(rows))              # kids[i] = sum of alloc_MiB of the direct children of row i
    last_at = Dict{Int, Int}()              # last_at[d] = index of the most recent row at depth d
    for (i, r) in enumerate(rows)
        if r.depth > 0 && haskey(last_at, r.depth - 1)
            kids[last_at[r.depth - 1]] += r.alloc_MiB
        end
        last_at[r.depth] = i
    end
    # merge(r, (; self_alloc = ...)) = row r with one extra field. max(..., 0.0) removes tiny negative rounding noise.
    return [merge(r, (; self_alloc = max(r.alloc_MiB - kids[i], 0.0))) for (i, r) in enumerate(rows)]
end

# Same-named sections merged over the tree: (name, ncalls, self time [s], self allocation [MiB]).
# under = "text": only sections whose path contains "text" (same meaning as in plot_self_time).
function stats_by_name(rows; under = "")
    full = add_self_alloc(rows)   # first on ALL rows (needs the whole tree), only then select
    sel = isempty(under) ? full : filter(r -> occursin(under, r.path), full)
    isempty(sel) && error("No sections with \"$under\" in their path")
    acc = Dict{String, Tuple{Float64, Float64, Float64}}()
    for r in sel
        n, s, a = get(acc, r.name, (0.0, 0.0, 0.0))
        acc[r.name] = (n + r.ncalls, s + r.self, a + r.self_alloc)
    end
    return [(; name = k, ncalls = round(Int, v[1]), self = v[2], self_alloc = v[3]) for (k, v) in acc]
end

# 123.4 MiB or 5.67 GiB
fmt_mib(x) = x >= 1024 ? "$(round(x / 1024; sigdigits = 3)) GiB" : "$(round(x; sigdigits = 3)) MiB"

# Mean self allocation of ONE call of each section (MiB), sections with the same name merged.
# = self allocation summed over all calls / number of calls. Needs only the CSV rows.
function plot_alloc_per_call(rows, label; top = 25, under = "")
    flat = [(; name = r.name, ncalls = r.ncalls, per_call = r.self_alloc / r.ncalls) for r in stats_by_name(rows; under) if r.ncalls > 0]
    sort!(flat; by = x -> x.per_call, rev = true)
    flat = flat[1:min(top, length(flat))]
    names = [shorten(r.name) for r in flat]
    labels = ["$(fmt_mib(r.per_call)) per call  ($(r.ncalls)x)" for r in flat]
    ttl = "$label: top $(length(flat)) sections by mean self allocation per call" * (isempty(under) ? "" : ", inside \"$under\"")
    return hbars(names, [r.per_call for r in flat], labels; title = ttl,
        xlabel = "mean self allocation of ONE call [MiB]   (number of calls in brackets)", color = :mediumpurple)
end


# Largest allocation of ONE call of each section (MiB), from the max_alloc_MiB column.
# That column is measured per section NAME (largest single call over all sections with that name) and includes sub-sections.
# leaves_only = true: only sections without timed sub-sections, so that wrappers (TOTAL run, leading_boundary, ...) do not fill the top of the plot.
# Shown per bar: largest call, mean call, number of calls.
function plot_max_alloc_per_call(rows, label; top = 25, under = "", leaves_only = true)
    # A row is a leaf if the next row is not deeper (the children of a row come directly after it)
    is_leaf = [i == length(rows) || rows[i + 1].depth <= r.depth for (i, r) in enumerate(rows)]
    keep = [(!leaves_only || is_leaf[i]) && (isempty(under) || occursin(under, r.path)) && !isnan(r.max_alloc_MiB) for (i, r) in enumerate(rows)]
    sel = rows[keep]
    isempty(sel) && error("No sections with max_alloc_MiB data (and \"$under\" in their path): rerun ising_model_3D_timing.jl with the new code to get the extra CSV column")

    # Merge same-named sections: (largest call, number of calls, total allocation)
    acc = Dict{String, Tuple{Float64, Int, Float64}}()
    for r in sel
        m, n, a = get(acc, r.name, (0.0, 0, 0.0))
        acc[r.name] = (max(m, r.max_alloc_MiB), n + r.ncalls, a + r.alloc_MiB)
    end
    flat = [(; name = k, max_alloc = v[1], ncalls = v[2], mean_alloc = v[3] / v[2]) for (k, v) in acc if v[2] > 0]
    sort!(flat; by = x -> x.max_alloc, rev = true)
    flat = flat[1:min(top, length(flat))]

    names = [shorten(r.name) for r in flat]
    labels = ["max $(fmt_mib(r.max_alloc)), mean $(fmt_mib(r.mean_alloc))  ($(r.ncalls)x)" for r in flat]
    ttl = "$label: top $(length(flat)) sections by largest allocation of ONE call" * (isempty(under) ? "" : ", inside \"$under\"")
    return hbars(names, [r.max_alloc for r in flat], labels; title = ttl,
        xlabel = "largest allocation of one call [MiB], GC-counted   (then mean per call and number of calls)", color = :tomato)
end