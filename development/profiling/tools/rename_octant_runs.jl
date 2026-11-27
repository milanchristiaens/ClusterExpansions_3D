# Renames the CTMRG3DOctant runs of the old face projector names in the data: octantedgeplain -> octantunweighted, octantedge -> octantsigmaweighted,
# octantedgelead -> octantleadweighted. It changes the run, case and setting columns of every results csv in data/profiling/comparing (after a backup copy),
# and the run folders and timing files whose names contain the old names. With DRY_RUN = true it only prints what it would change
# (CE_RENAME_DRY_RUN=false in the environment sets it to false, so the file itself does not have to change).

# ======================================================================================================================
# Settings
# ======================================================================================================================
const DRY_RUN = get(ENV, "CE_RENAME_DRY_RUN", "true") != "false"
const DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "data", "profiling", "comparing"))
# ======================================================================================================================

# octantedgelead and octantedgeplain first: octantedge is the start of both
const NEW_NAME = Dict("octantedgeplain" => "octantunweighted", "octantedgelead" => "octantleadweighted", "octantedge" => "octantsigmaweighted")
rename_text(s) = replace(s, r"octantedge(plain|lead)?" => m -> NEW_NAME[m])

# The run, case and setting columns (2, 3, 4) of a csv line; the other columns are kept as they are
function rename_line(line)
    p = split(line, ','; limit = 5)
    length(p) < 5 && return line
    return join([p[1], rename_text(p[2]), rename_text(p[3]), rename_text(p[4]), p[5]], ',')
end

function rename_csv(path)
    lines = readlines(path)
    new = rename_line.(lines)
    n = count(lines .!= new)
    println("$path: $n rows to rename")
    (n == 0 || DRY_RUN) && return nothing
    backup = path * ".bak_" * Libc.strftime("%Y%m%d_%H%M%S", time())
    cp(path, backup)
    open(path, "w") do io
        foreach(l -> println(io, l), new)
    end
    println("  done (backup: $backup)")
    return nothing
end

# Folders and files below DATA_DIR whose names contain an old name, deepest first (so that a folder is renamed after its files)
function rename_paths()
    paths = String[]
    for (root, dirs, files) in walkdir(DATA_DIR)
        for name in [dirs; files]
            occursin("octantedge", name) && push!(paths, joinpath(root, name))
        end
    end
    sort!(paths; by = p -> -count(==('/'), replace(p, '\\' => '/')))
    for p in paths
        q = joinpath(dirname(p), rename_text(basename(p)))
        println("$p -> $(basename(q))")
        ispath(q) && (println("  NOT renamed: $q already exists"); continue)
        DRY_RUN || mv(p, q)
    end
    return nothing
end

function main()
    isdir(DATA_DIR) || error("$DATA_DIR does not exist")
    DRY_RUN && println("DRY_RUN = true: nothing is changed, run again with CE_RENAME_DRY_RUN=false to rename")
    for f in readdir(DATA_DIR; join = true)
        endswith(f, ".csv") && rename_csv(f)
    end
    rename_paths()
    return nothing
end

main()
