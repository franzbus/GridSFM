#!/usr/bin/env julia
#=
gen_perturbed_data.jl — Generate perturbed gridSFM-capable .pyg.json files
from one or more .solvable.json base topologies.

Global task queue: all grids' scenarios pooled together. Workers never idle.

Each scenario applies ONE pure perturbation mode (not a random combination)
so per-mode signal stays uncorrelated. Five modes:
  loads    — system load factor sf ∈ [0.8, 1.5], then per-load ±10% jitter
             on Pd / Qd (multiplicative: pd *= sf·(0.9 + 0.2·rand))
  costs    — cost-coefficient shuffle among ~40% of active gens, within
             same-ncost pools (preserves cost-function degree)
  killgen  — flip gen_status=0 on 1, 2, or 3 active gens (probabilities
             0.7 / 0.2 / 0.1). Preserves ≥2 active gens so the grid stays
             operable.
  derate   — on ~10% of active branches, scale rate_a/b/c by a factor
             drawn uniformly from [0.7, 0.95].
  vsqueeze — on ~10% of buses, shrink the voltage band by a random amount
             ∈ [0, 0.01] pu on EACH boundary independently (vmin += δ_lo,
             vmax -= δ_hi). Reverts if δ_lo + δ_hi would cross the bounds.

Plus one `base_unperturbed.pyg.json` per grid (the un-perturbed solvable file).

Grid-list format (one line per grid, # for comments):
  <solvable_json_path> <n_per_mode>
Total scenarios per grid = 1 (unperturbed) + 5 × n_per_mode.

Usage:
  julia --project=. gen_perturbed_data.jl <grid_list_file> [n_proc] [out_root]
=#
using Distributed, Printf, Random

positional = filter(a -> !startswith(a, "--"), ARGS)
if length(positional) < 1
    println("Usage: julia gen_perturbed_data.jl <grid_list_file> [n_proc] [out_root]")
    exit(1)
end

grid_file = positional[1]
n_proc    = length(positional) > 1 ? parse(Int, positional[2]) : min(Sys.CPU_THREADS, 120)
out_root  = length(positional) > 2 ? positional[3] : "./out"

# Parse grid list
grid_specs = []
for line in readlines(grid_file)
    line = strip(line)
    (isempty(line) || startswith(line, "#")) && continue
    parts = split(line)
    length(parts) >= 2 || continue
    push!(grid_specs, (path=String(parts[1]), n_per_mode=parse(Int, parts[2])))
end

println("gen_perturbed_data.jl")
println("  grids:    $(length(grid_specs))")
println("  workers:  $n_proc")
println("  out_root: $out_root")

addprocs(n_proc)

@everywhere begin
    using PowerModels, Ipopt, JuMP, JSON3, OrderedCollections, Random, Printf, Dates, Polynomials
    PowerModels.silence()

    # Reuse build_gridsfm_data — single source of truth for the pyg.json schema.
    include(joinpath(@__DIR__, "export_gridsfm_data.jl"))

    # ── 5 pure perturbation modes ──
    # Each returns extra metadata fields describing what was done.
    function _mode_loads!(d, rng)
        sf = 0.8 + rand(rng)*(1.5 - 0.8)
        for (_, ld) in get(d, "load", Dict())
            ld["pd"] *= sf * (0.9 + rand(rng)*0.2)
            ld["qd"] *= sf * (0.9 + rand(rng)*0.2)
        end
        Dict("system_load_factor" => round(sf, digits=3))
    end

    function _mode_costs!(d, rng)
        act = [(k,g) for (k,g) in get(d, "gen", Dict()) if Int(get(g,"gen_status",get(g,"status",1))) == 1]
        # Need at least 2 active gens to meaningfully shuffle costs.
        # With 1 gen: round(Int, 0.4) = 0, clamp(0, 2, 1) = 2 (lo > hi
        # edge case in clamp), then randperm(1)[1:2] is out-of-bounds.
        length(act) < 2 && return Dict("cost_shuffle_pct" => 0.0)
        ns = clamp(round(Int, length(act)*0.4), 2, length(act))
        sel = [act[i] for i in randperm(rng, length(act))[1:ns]]
        by = Dict{Int,Vector{Int}}()
        for (i,(_,g)) in enumerate(sel)
            nc = get(g, "ncost", length(get(g,"cost",[])))
            push!(get!(by, nc, Int[]), i)
        end
        n = 0
        for (_, ix) in by
            length(ix) < 2 && continue
            cs = [try Float64.(sel[i][2]["cost"]) catch; Float64[] end for i in ix]
            pm = randperm(rng, length(ix))
            for (j,i) in enumerate(ix)
                !isempty(cs[pm[j]]) && (sel[i][2]["cost"] = cs[pm[j]])
            end
            n += length(ix)
        end
        Dict("cost_shuffle_pct" => round(n/max(1,length(d["gen"]))*100, digits=1))
    end

    function _mode_killgen!(d, rng)
        r = rand(rng); nk = r < 0.7 ? 1 : (r < 0.9 ? 2 : 3)
        act = [(k,g) for (k,g) in get(d, "gen", Dict())
               if Int(get(g,"gen_status",get(g,"status",1))) == 1 && get(g,"pmax",0.0) > 0.01]
        length(act) <= 3 && return Dict("n_gens_killed" => 0)
        nk = min(nk, length(act)-2)
        for i in randperm(rng, length(act))[1:nk]
            act[i][2]["gen_status"] = 0
        end
        Dict("n_gens_killed" => nk)
    end

    function _mode_derate!(d, rng)
        br = [(k,b) for (k,b) in get(d, "branch", Dict())
              if Int(get(b,"br_status",1)) == 1 && get(b,"rate_a",0.0) > 0]
        isempty(br) && return Dict("n_lines_derated" => 0)
        nd = max(1, round(Int, length(br)*0.1))
        for i in randperm(rng, length(br))[1:nd]
            f = 0.7 + rand(rng)*0.25; _,b = br[i]
            b["rate_a"] *= f
            haskey(b,"rate_b") && (b["rate_b"] *= f)
            haskey(b,"rate_c") && (b["rate_c"] *= f)
        end
        Dict("n_lines_derated" => nd)
    end

    function _mode_vsqueeze!(d, rng)
        bs = collect(get(d, "bus", Dict())); isempty(bs) && return Dict("n_buses_vsqueezed" => 0)
        ns = max(1, round(Int, length(bs)*0.1))
        for i in randperm(rng, length(bs))[1:ns]
            _,b = bs[i]
            lo,hi = get(b,"vmin",0.9), get(b,"vmax",1.1)
            b["vmin"] = lo + 0.01*rand(rng); b["vmax"] = hi - 0.01*rand(rng)
            b["vmin"] >= b["vmax"] && (b["vmin"] = lo; b["vmax"] = hi)
        end
        Dict("n_buses_vsqueezed" => ns)
    end

    const MODE_FNS = Dict(
        "loads"    => _mode_loads!,
        "costs"    => _mode_costs!,
        "killgen"  => _mode_killgen!,
        "derate"   => _mode_derate!,
        "vsqueeze" => _mode_vsqueeze!,
    )
    const MODES = ("loads", "costs", "killgen", "derate", "vsqueeze")

    function _solve_and_build(data)
        pm = PowerModels.instantiate_model(data, ACPPowerModel, PowerModels.build_opf)
        solver = optimizer_with_attributes(Ipopt.Optimizer,
            "print_level" => 0, "sb" => "yes",
            "max_iter" => 3000, "tol" => 1e-6, "acceptable_tol" => 1e-4)
        result = PowerModels.optimize_model!(pm, optimizer=solver)
        return build_gridsfm_data(pm, result, data)
    end
end  # @everywhere

# Build global task list across ALL grids
tasks = []
for gs in grid_specs
    if !isfile(gs.path)
        @warn "missing solvable file: $(gs.path)"; continue
    end
    case = splitext(basename(gs.path))[1]
    # strip trailing .solvable if present
    endswith(case, ".solvable") && (case = case[1:end-length(".solvable")])
    case_out = joinpath(out_root, case)
    mkpath(case_out)

    # Unperturbed (scenario 0)
    unpert_path = joinpath(case_out, "base_unperturbed.pyg.json")
    if !isfile(unpert_path)
        push!(tasks, (gs.path, "base", 0, true, case_out))
    end
    # Perturbed — one pure mode per scenario
    for mode in MODES
        for s in 1:gs.n_per_mode
            fname = @sprintf("%s_%04d.pyg.json", mode, s)
            out_path = joinpath(case_out, fname)
            isfile(out_path) && continue
            push!(tasks, (gs.path, mode, s, false, case_out))
        end
    end
    total = 1 + 5*gs.n_per_mode
    queued = length(filter(t -> t[5] == case_out, tasks))
    println("  $case: $queued/$total tasks queued")
end

println("\nTotal: $(length(tasks)) tasks across $(length(grid_specs)) grids → pmap with $n_proc workers")

results = pmap(tasks; batch_size=1) do (file, mode, sidx, unpert, out_dir)
    try
        # Julia's hash(...) is salted per session (HASH_SEED is randomized),
        # so omitting the second argument would make scenarios non-reproducible
        # across runs. Pass an explicit UInt seed so the seed derivation is
        # stable (file path + mode name → same seed every run).
        rng = MersenneTwister(42 + sidx + hash((file, mode), UInt(0)))
        # import_all=false: don't carry stage-2 top-level metadata dicts
        # (e.g. `_relaxation`) into PowerModels data — their non-integer keys
        # can break InfrastructureModels inside instantiate_model below.
        data = PowerModels.parse_file(file; import_all=false, validate=true)
        extra = Dict{String,Any}()
        unpert || (extra = MODE_FNS[mode](data, rng))
        opf, feas = _solve_and_build(data)
        opf["metadata"]["scenario_id"]  = sidx
        opf["metadata"]["perturb_mode"] = mode
        opf["metadata"]["feasible"]     = feas
        for (k,v) in extra
            opf["metadata"][k] = v
        end
        fname = unpert ? "base_unperturbed.pyg.json" : @sprintf("%s_%04d.pyg.json", mode, sidx)
        open(joinpath(out_dir, fname), "w") do io
            JSON3.pretty(io, opf)
        end
        return (out_dir, mode, feas)
    catch e
        @warn "[$mode/$sidx $(basename(file))] FAILED: $(first(sprint(showerror,e), 200))"
        return (out_dir, mode, false)
    end
end

# Per-grid, per-mode summary
println("\n=== Summary ===")
for gs in grid_specs
    case = splitext(basename(gs.path))[1]
    endswith(case, ".solvable") && (case = case[1:end-length(".solvable")])
    case_out = joinpath(out_root, case)
    grid_results = filter(r -> r[1] == case_out, results)
    isempty(grid_results) && continue
    n_files = isdir(case_out) ? length(filter(f -> endswith(f, ".pyg.json"), readdir(case_out))) : 0
    println("  $case  ($n_files files on disk):")
    for mode in ("base", MODES...)
        mr = filter(r -> r[2] == mode, grid_results)
        isempty(mr) && continue
        n_feas = count(r -> r[3], mr); n_fail = count(r -> !r[3], mr)
        @printf("    %-9s  %4d feasible  %4d infeasible\n", mode, n_feas, n_fail)
    end
end

println("\nDone!")
