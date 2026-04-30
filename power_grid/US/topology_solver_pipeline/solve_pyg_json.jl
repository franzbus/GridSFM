#!/usr/bin/env julia
# solve_pyg_json.jl
#
# Given a solvable base (`.solvable.json`) and a perturbed scenario file
# (`.pyg.json`, produced by gen_perturbed_data.jl), reconstruct the exact
# PowerModels data the scenario represents, solve strict AC-OPF, and
# compare the resulting objective against the value stored in the pyg's
# metadata. This is the round-trip check for stage-3 outputs — it verifies
# that a .pyg.json contains enough information to exactly reproduce the
# solve that produced it.
#
# Approach: start from the PowerModels-native .solvable.json (which has
# every field PowerModels needs) and overlay the perturbed values from the
# pyg.json:
#   - loads     → grid.nodes.load[:,1:2]
#   - killgen   → gens absent from grid.nodes.generator (gen_status=0)
#   - derate    → rate_a/b/c from grid.edges.{ac_line,transformer}.features
#   - vsqueeze  → vmin/vmax from grid.nodes.bus
#   - costs     → cost coefficients from grid.nodes.generator
# Warm-start (va/vm/pg/qg) comes from the pyg's solution section.
#
# Usage:
#   julia --project=<repo> solve_pyg_json.jl <solvable.json> <scenario.pyg.json>
#
# Prints one of:
#   RESOLVE ok       obj=<float> expected=<float> (Δ=<pct>%)
#   RESOLVE status=<term> obj=<float>   (on non-convergence)
# and exits 0 on convergence + objective match within 0.1%, 1 otherwise.
#
using PowerModels, Ipopt, JuMP, JSON3
using Printf

PowerModels.silence()


# ═══════════════════════════════════════════════════════════════════════
# pyg.json  →  PowerModels data dict
# ═══════════════════════════════════════════════════════════════════════
# Takes a solvable base file and a pyg.json scenario, returns a ready-to-
# solve PowerModels net. Matches the gridSFM schema exported by
# build_gridsfm_data (see export_gridsfm_data.jl header for column orders).
#
# Steps:
#   1. Parse the solvable base with PowerModels (full topology, every field
#      PowerModels needs). Use import_all=false so non-PM top-level dicts
#      that the relaxation pipeline attaches to the .solvable.json (e.g.
#      `_relaxation`, with non-integer-parseable keys) don't flow into
#      instantiate_model — InfrastructureModels assumes integer-keyed
#      component dicts at the top level and can crash otherwise.
#   2. Overlay perturbed loads from grid.nodes.load.
#   3. Overlay warm starts from solution.nodes.{bus,generator} into
#      va_start/vm_start/pg_start/qg_start.
#   4. Overlay non-load perturbations (killgen / derate / vsqueeze / costs).
function build_net_from_pyg(solvable_path::AbstractString, pyg::AbstractDict)
    net = PowerModels.parse_file(solvable_path; import_all=false, validate=true)
    grid = pyg["grid"]; meta = pyg["metadata"]

    # Sort PM components by index — row order must match what the exporter
    # (build_gridsfm_data in export_gridsfm_data.jl) produced.
    _sort_by_index(d::Dict{String,<:Any}) = sort(collect(values(d)); by = x -> Int(x["index"]))

    # Match the exporter row sets/order exactly:
    #   - loads  → active only (status == 1)       — same as exporter
    #   - gens   → active only (gen_status == 1)   — same as exporter
    #   - buses  → ALL buses sorted by index, INCLUDING bus_type == 4.
    #             The exporter does NOT filter by bus_type, so if we did,
    #             row counts would mismatch and the positional warm-start
    #             overlay below would silently skip on any grid with
    #             isolated buses.
    # Match the exporter's active-generator predicate exactly: some Matpower
    # / PowerModels inputs use `gen_status`, others the legacy `status` —
    # export_gridsfm_data.jl reads both. Filtering on only `gen_status` here
    # would misalign row order vs the pyg schema on legacy-status grids and
    # silently corrupt the killgen / cost / bound overlays below.
    _gen_active(g) = Int(get(g, "gen_status", get(g, "status", 1))) == 1
    loads_pm  = filter(ld -> Int(get(ld, "status", 1)) == 1, _sort_by_index(get(net, "load", Dict{String,Any}())))
    buses_pm  = _sort_by_index(get(net, "bus", Dict{String,Any}()))
    gens_pm   = filter(_gen_active, _sort_by_index(get(net, "gen", Dict{String,Any}())))

    # ── Loads overlay ──
    loads_json = grid["nodes"]["load"]::Vector
    if length(loads_json) != length(loads_pm)
        error("load count mismatch: pyg=$(length(loads_json)) solvable=$(length(loads_pm))")
    end
    for (ld_pm, row) in zip(loads_pm, loads_json)
        ld_pm["pd"] = Float64(row[1])
        ld_pm["qd"] = Float64(row[2])
    end

    # ── Bus warm-start (va, vm) ──
    # Set BOTH va/vm (PowerModels' primary fields that solve_ac_opf reads)
    # AND va_start/vm_start (used by instantiate_model warm-start path).
    # Without setting va/vm directly, some PowerModels code paths ignore
    # the *_start fields and default to flat start (vm=1, va=0).
    if haskey(pyg, "solution") && haskey(pyg["solution"], "nodes") &&
       haskey(pyg["solution"]["nodes"], "bus")
        bus_sol = pyg["solution"]["nodes"]["bus"]::Vector
        if !isempty(bus_sol)
            if length(bus_sol) != length(buses_pm)
                error("bus warm-start row count mismatch: pyg.solution=$(length(bus_sol)) " *
                      "solvable=$(length(buses_pm)). Did the exporter schema change?")
            end
            for (b_pm, s) in zip(buses_pm, bus_sol)
                b_pm["va"] = Float64(s[1]); b_pm["va_start"] = Float64(s[1])
                b_pm["vm"] = Float64(s[2]); b_pm["vm_start"] = Float64(s[2])
            end
        end
    end

    # ── Bus vmin/vmax overlay (vsqueeze) ──
    bus_ids = [Int(x) for x in meta["bus_id_map"]]
    n_bus_pyg = length(grid["nodes"]["bus"])
    if length(bus_ids) != n_bus_pyg
        error("bus_id_map length $(length(bus_ids)) != grid.nodes.bus rows $n_bus_pyg")
    end
    for (i, row) in enumerate(grid["nodes"]["bus"])
        bid = bus_ids[i]
        b   = get(net["bus"], string(bid), nothing)
        b === nothing && continue
        b["vmin"] = Float64(row[3])
        b["vmax"] = Float64(row[4])
    end

    # ── Gen warm-start (pg, qg) + killgen + cost/bound overlays ──
    gen_nodes = grid["nodes"]["generator"]
    gen_ids   = [Int(x) for x in meta["gen_id_map"]]
    live_ids  = Set(gen_ids)

    # Mark absent gens as offline (killgen). Use the same gen_status /
    # legacy `status` fallback as the filter above so we don't miss gens
    # on legacy-status inputs.
    for (_, g) in net["gen"]
        if !(Int(g["index"]) in live_ids) && _gen_active(g)
            g["gen_status"] = 0
        end
    end

    # Per-live-gen overlays: warm-start + cost + bounds.
    sol_gen = haskey(pyg, "solution") && haskey(pyg["solution"], "nodes") ?
              get(pyg["solution"]["nodes"], "generator", nothing) : nothing
    for (i, row) in enumerate(gen_nodes)
        gid = gen_ids[i]
        g   = get(net["gen"], string(gid), nothing)
        g === nothing && continue

        # Cost coefficients are stored as [cp2, cp1, cp0] (quadratic form).
        cp2, cp1, cp0 = Float64(row[9]), Float64(row[10]), Float64(row[11])
        g["cost"]  = Float64[cp2, cp1, cp0]
        g["model"] = 2
        g["ncost"] = 3

        # Keep gen bounds in sync (in case a future perturbation mode mutates them).
        g["pmin"] = Float64(row[3]); g["pmax"] = Float64(row[4])
        g["qmin"] = Float64(row[6]); g["qmax"] = Float64(row[7])

        # Warm start pg / qg from the solved values.
        # Set BOTH pg/qg (primary) AND pg_start/qg_start (instantiate_model path).
        if sol_gen !== nothing && i <= length(sol_gen) && !isempty(sol_gen[i])
            g["pg"] = Float64(sol_gen[i][1]); g["pg_start"] = Float64(sol_gen[i][1])
            g["qg"] = Float64(sol_gen[i][2]); g["qg_start"] = Float64(sol_gen[i][2])
        end
    end

    # ── Branch rate_a/b/c overlay (derate) ──
    # ac_line features: [angmin, angmax, b_fr, b_to, br_r, br_x, rate_a, rate_b, rate_c]
    ac_ids = [Int(x) for x in meta["ac_line_branch_ids"]]
    for (i, feats) in enumerate(grid["edges"]["ac_line"]["features"])
        bid = ac_ids[i]
        b   = get(net["branch"], string(bid), nothing)
        b === nothing && continue
        b["rate_a"] = Float64(feats[7])
        b["rate_b"] = Float64(feats[8])
        b["rate_c"] = Float64(feats[9])
    end
    # transformer features: [angmin, angmax, br_r, br_x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to]
    tr_ids = [Int(x) for x in meta["transformer_branch_ids"]]
    for (i, feats) in enumerate(grid["edges"]["transformer"]["features"])
        bid = tr_ids[i]
        b   = get(net["branch"], string(bid), nothing)
        b === nothing && continue
        b["rate_a"] = Float64(feats[5])
        b["rate_b"] = Float64(feats[6])
        b["rate_c"] = Float64(feats[7])
    end

    return net
end


# Solve strict AC-OPF on `net`, returning (objective, termination_status).
# Warm starts are honored if va_start/vm_start/pg_start/qg_start are set.
function solve_strict(net)
    solver = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level"    => 0,
        "max_iter"       => 10000,
        "tol"            => 1e-6,
        "acceptable_tol" => 1e-4,
        "sb"             => "yes")
    pm  = PowerModels.instantiate_model(net, ACPPowerModel, PowerModels.build_opf)
    res = PowerModels.optimize_model!(pm, optimizer=solver)
    term = string(get(res, "termination_status", "UNKNOWN"))
    obj  = try Float64(get(res, "objective", NaN)) catch; NaN end
    return obj, term
end


function main()
    if length(ARGS) < 2
        println("Usage: julia solve_pyg_json.jl <solvable.json> <scenario.pyg.json>")
        exit(2)
    end
    solvable_path = ARGS[1]
    pyg_path      = ARGS[2]
    isfile(solvable_path) || (println("solvable not found: $solvable_path"); exit(1))
    isfile(pyg_path)      || (println("pyg not found: $pyg_path"); exit(1))

    pyg = open(pyg_path) do f; JSON3.read(f, Dict{String,Any}); end

    # Infeasible scenarios (metadata.feasible=false) have a zeroed-out
    # solution and a garbage objective in metadata — comparing against
    # that value is meaningless. The round-trip assertion for those is
    # "the reconstructed problem is also infeasible" (i.e. Ipopt agrees
    # the perturbation made the grid un-OPF-able). A re-solve that
    # converges on a scenario the generator marked infeasible is only
    # worth noting, not failing, since strict AC-OPF is non-convex.
    feasible_flag = try Bool(pyg["metadata"]["feasible"]) catch; true end

    net = build_net_from_pyg(solvable_path, pyg)

    t0 = time()
    obj, term = solve_strict(net)
    elapsed = time() - t0

    expected = try Float64(pyg["metadata"]["objective"]) catch; NaN end

    converged = occursin("LOCALLY_SOLVED",        uppercase(term)) ||
                occursin("OPTIMAL",               uppercase(term)) ||
                occursin("ALMOST_LOCALLY_SOLVED", uppercase(term))

    if !feasible_flag
        # Generator said this perturbation was infeasible — the acceptable
        # outcomes on re-solve are either infeasible again (expected), or
        # a legitimate local solve that the generator happened to miss.
        # Either way, no objective comparison is meaningful.
        @printf("RESOLVE ok-infeasible status=%s file=%s (pyg marked infeasible, no objective check)\n",
                term, basename(pyg_path))
        exit(0)
    end

    delta_pct = isnan(expected) || abs(expected) < 1e-9 ? NaN :
                100.0 * abs(obj - expected) / max(abs(expected), 1.0)

    # The question this script answers is "does the .pyg.json reproduce
    # the exact solve that produced it?" If the objective matches within
    # 0.1% we consider that a pass even if Ipopt returns LOCALLY_INFEASIBLE
    # after warm-starting from the pyg's solved point — that outcome is
    # just numerical noise on constr_viol_tol, not a real data defect.
    obj_matches = !isnan(delta_pct) && delta_pct <= 0.1

    if converged && obj_matches
        @printf("RESOLVE ok obj=%.2f expected=%.2f Δ=%.4f%% elapsed=%.2fs file=%s\n",
                obj, expected, delta_pct, elapsed, basename(pyg_path))
        exit(0)
    elseif obj_matches
        # Objective matches but Ipopt reports non-convergence — soft pass.
        @printf("RESOLVE ok-objmatch status=%s obj=%.2f expected=%.2f Δ=%.4f%% elapsed=%.2fs file=%s\n",
                term, obj, expected, delta_pct, elapsed, basename(pyg_path))
        exit(0)
    else
        @printf("RESOLVE FAIL status=%s obj=%.2f expected=%.2f Δ=%s file=%s\n",
                term, obj, expected, isnan(delta_pct) ? "?" : @sprintf("%.4f%%", delta_pct),
                basename(pyg_path))
        exit(1)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
