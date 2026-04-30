#!/usr/bin/env julia
# solve_topo_json.jl
#
# Input:  a PowerModels-native topology JSON (possibly not solvable strict)
# Output: a PowerModels-native JSON that IS cold-strict solvable
#         (loads with parse_file, solves with strict AC-OPF from flat start,
#         no warm-start, no preprocessing).
#
# How: iterates relaxation levels L0, AC1, L1..L5 using run_opf_relaxation.jl.
# At each level: writes the mutated trial_data (decommit + impedance fix +
# DC-derived shunts + level-specific constraint relaxation) to a tmp file,
# then re-loads that file, zeros out warm-start fields, and solves strict
# AC-OPF to verify. First level that passes cold-strict is the winner.
#
# The output JSON has a `_relaxation` metadata field recording which level
# was applied and which constraints were changed. Electrical parameters
# (rate_a, br_x, vmin/vmax, pmin, shunts) are written into the output so
# downstream tools don't need the relaxation pipeline.
#
# Usage:
#   julia --project=<repo> solve_topo_json.jl <input.json> <output.solvable.json>
#
using PowerModels, Ipopt
using Printf

PowerModels.silence()

# Relaxation pipeline (PROGRAM_FILE guard prevents its main() from auto-running)
include(joinpath(@__DIR__, "run_opf_relaxation.jl"))

# L0, AC1, L1..L5 — escalation order matches run_opf_relaxation's progressive path
const LEVEL_ORDER = [0, 6, 1, 2, 3, 4, 5]


"Zero out vm/va/pg/qg so Ipopt does a flat start."
function strip_warm_start!(data)
    for (_, b) in get(data, "bus", Dict())
        b["vm"] = 1.0; b["va"] = 0.0
    end
    for (_, g) in get(data, "gen", Dict())
        g["pg"] = 0.0; g["qg"] = 0.0
    end
end


"Parse `path`, strip warm-start, solve strict AC-OPF. Return (solved?, status, obj)."
function cold_strict_solve(path::AbstractString)
    net = try
        # gridsfm_topo's _model.json files (and our own .solvable.json
        # outputs) may omit optional device dicts like "storage"/"switch".
        # _parse_with_default_devices injects empty dicts before validation
        # so PowerModels' _check_connectivity doesn't crash.
        _parse_with_default_devices(path; import_all=false, validate=true)
    catch e
        return (false, "PARSE_ERROR:" * first(sprint(showerror,e), 200), NaN)
    end
    strip_warm_start!(net)
    # Match run_opf_relaxation.jl's SOLVER_MAX_ITER (10000) so cold-strict
    # verification doesn't spuriously fail on large grids when the upstream
    # solver would have converged.
    solver = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "max_iter"    => 10000,
        "tol"         => 1e-6,
        "acceptable_tol" => 1e-4,
    )
    res = try
        PowerModels.solve_ac_opf(net, solver)
    catch e
        return (false, "SOLVE_ERROR:" * first(sprint(showerror,e), 200), NaN)
    end
    term = string(get(res, "termination_status", "UNKNOWN"))
    obj  = try Float64(get(res, "objective", NaN)) catch; NaN end
    solved = occursin("LOCALLY_SOLVED", uppercase(term)) ||
             occursin("OPTIMAL",        uppercase(term)) ||
             occursin("ALMOST_LOCALLY_SOLVED", uppercase(term))
    return (solved, term, obj)
end


"Call run_opf_relaxation.jl at exactly one level, save mutated data to tmp_path."
function solve_at_level(input_path, tmp_path, level)
    opts = Dict(
        "model_file"        => input_path,
        "formulation"       => "ac",
        "output_file"       => nothing,
        "dc_output_file"    => nothing,
        "warm_start_file"   => nothing,
        "interface_file"    => nothing,
        "save_relaxed_file" => tmp_path,
        "soc"               => false,
        "verbose"           => false,
        "progressive"       => false,  # single-level attempt, no escalation
        "start_level"       => level,
        "warmstart_only"    => false,
    )
    try
        _, solved_level = run_opf(input_path, opts)
        return solved_level
    catch e
        @warn "    pipeline crashed at L$level: $(first(sprint(showerror,e), 200))"
        return -1
    end
end


"Iterate levels until cold-strict passes. Write winner to `output_path`."
function make_solvable(input_path::AbstractString, output_path::AbstractString)
    @info "solve: $(basename(input_path)) → $(basename(output_path))"
    tmp_path = output_path * ".tmp.json"
    for L in LEVEL_ORDER
        lbl = L == 6 ? "AC1" : "L$L"
        @info "  Trying $lbl"
        isfile(tmp_path) && rm(tmp_path)
        solved_level = solve_at_level(input_path, tmp_path, L)
        if solved_level < 0 || !isfile(tmp_path)
            @info "    pipeline did not save at $lbl, moving on"
            continue
        end
        # Cold-strict verify (strip warm-start, re-solve)
        solved, term, obj = cold_strict_solve(tmp_path)
        if solved
            mv(tmp_path, output_path; force=true)
            @info "  ✓ cold-strict solved at $lbl  (obj=$(round(obj, digits=2)))"
            return (level=L, label=lbl, objective=obj, status=term)
        else
            @info "    saved but FAILS cold-strict at $lbl: $term"
        end
    end
    isfile(tmp_path) && rm(tmp_path)
    @warn "  ✗ no level produced a cold-strict solvable JSON"
    return nothing
end


function main()
    if length(ARGS) < 2
        println("""
Usage: julia solve_topo_json.jl <input.json> <output.solvable.json>

Produces a cold-strict solvable version of <input.json> at <output>
by iterating relaxation levels (L0, AC1, L1..L5) until one passes
cold-strict verification. Output JSON has electrical params written in
(rate_a, br_x, vmin/vmax, pmin, shunts) and can be loaded with
PowerModels.parse_file + solved with strict AC-OPF — no warm-start or
preprocessing needed.
""")
        exit(2)
    end

    input_path  = ARGS[1]
    output_path = ARGS[2]
    if !isfile(input_path)
        println("Input not found: $input_path"); exit(1)
    end
    mkpath(dirname(output_path))

    result = make_solvable(input_path, output_path)
    if result === nothing
        exit(1)
    end
    @printf("RESULT %s %s obj=%.2f\n", basename(input_path), result.label, result.objective)
    exit(0)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
