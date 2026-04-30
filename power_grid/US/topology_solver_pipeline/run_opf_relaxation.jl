#!/usr/bin/env julia
"""
run_opf_relaxation.jl - Run Optimal Power Flow with Progressive Relaxation

Tries increasingly relaxed formulations until the model solves.
Relaxation levels are defined in shared/relaxation_levels.json (single source
of truth).  Levels 0-5 target DC-relevant constraints (angles, thermal limits,
load).  AC1 relaxes only voltage bounds and reactive-power limits.

For AC-OPF the solver tries L0 → AC1 → L1 → … → L5; if AC1 alone doesn't
solve it, AC1's V/Q relaxation is kept as a base layer for L1-L5
(monotonically more relaxed).

Also used as an include() target by solve_topo_json.jl to access the
pipeline functions (run_opf, decommit_generators!, apply_relaxation!, etc.)
for the cold-strict solve workflow. A PROGRAM_FILE guard at the bottom
prevents main() from running on include.

Usage (script):
    julia run_opf_relaxation.jl <model.json> [options]

Options:
    --ac                Run AC-OPF (default)
    --dc                Run DC-OPF (faster, linear approximation)
    --soc               Run SOC relaxation (convex, between DC and AC)
    --output FILE       Save results to JSON file
    --dc-output FILE    Save DC warm-start results to JSON (AC mode only)
    --save-warm-start F Save warm-started model (with DC-derived shunts) to JSON
    --save-relaxed F    Save relaxed model (mutated trial_data with all level
                        changes applied) to JSON — used to produce a JSON that
                        solves cold-strict without this pipeline
    --interface-file F  Load interface limits from separate JSON file
    --relax-level N     Start at relaxation level N (0-5, or ac1). Requires
                        --no-progressive to run ONLY that level
    --no-progressive    Disable progressive relaxation, run single level at
                        --relax-level (or L0 if unspecified)
    --warmstart-only    Do DC warm-start and exit (for subprocess progressive mode)
    --verbose           Show detailed solver output
"""

using PowerModels
using Ipopt
using JuMP
using JSON

# Silence PowerModels warnings
PowerModels.silence()

# ═══════════════════════════════════════════════════════════════════════
# JSON loader that injects empty defaults for optional device dicts
# ═══════════════════════════════════════════════════════════════════════
# PowerModels' parse_file → correct_network_data! → _check_connectivity
# indexes data["storage"] (and a few other optional keys) unconditionally,
# so a JSON that omits those keys crashes during validation even though
# the omission is semantically valid (just no devices of that type).
#
# Workaround: pre-read the JSON, inject `{}` for any missing key in the
# allow-list, write it to a `.json` temp file, then call parse_file as
# normal. Keeps validate=true so real issues are still caught. When all
# expected keys are already present, defer to parse_file directly so
# the fast path matches unpatched behaviour exactly.

# Optional top-level device dicts that PowerModels references but that
# gridsfm_topo's _model.json output sometimes omits.
const _PM_OPTIONAL_DEVICE_KEYS = ("storage", "switch", "dcline", "shunt")

"""
    _parse_with_default_devices(path; import_all=false, validate=true)

Load a PowerModels-native JSON like `parse_file`, but inject empty dicts
for any missing entry in `_PM_OPTIONAL_DEVICE_KEYS` so PowerModels'
validation step doesn't crash on `KeyError("storage")` etc.
"""
function _parse_with_default_devices(path::AbstractString;
                                     import_all::Bool=false,
                                     validate::Bool=true)
    raw = JSON.parsefile(path)
    missing_keys = filter(k -> !haskey(raw, k), _PM_OPTIONAL_DEVICE_KEYS)
    if isempty(missing_keys)
        # Fast path: nothing to inject, behave exactly like parse_file.
        return PowerModels.parse_file(path; import_all=import_all, validate=validate)
    end
    for k in missing_keys
        raw[k] = Dict{String,Any}()
    end
    keys_str = join(missing_keys, ", ")
    @info "  Injected empty defaults for missing keys: $keys_str"
    # parse_file dispatches by extension — write to a .json temp file so
    # the JSON parser is selected (parse_json takes a path, not a string).
    tmp_path = tempname() * ".json"
    try
        open(tmp_path, "w") do io
            JSON.print(io, raw)
        end
        return PowerModels.parse_file(tmp_path; import_all=import_all, validate=validate)
    finally
        rm(tmp_path; force=true)
    end
end

# ═══════════════════════════════════════════════════════════════════════
# CLI Argument Parsing
# ═══════════════════════════════════════════════════════════════════════

function parse_args(args)
    options = Dict(
        "model_file" => nothing,
        "formulation" => "ac",
        "output_file" => nothing,
        "dc_output_file" => nothing,
        "warm_start_file" => nothing,
        "save_relaxed_file" => nothing,
        "interface_file" => nothing,
        "soc" => false,
        "verbose" => false,
        "progressive" => true,
        "start_level" => 0,
        "warmstart_only" => false,
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--ac"
            options["formulation"] = "ac"
        elseif arg == "--dc"
            options["formulation"] = "dc"
        elseif arg == "--soc"
            options["soc"] = true
        elseif arg == "--output" && i < length(args)
            i += 1
            options["output_file"] = args[i]
        elseif arg == "--dc-output" && i < length(args)
            i += 1
            options["dc_output_file"] = args[i]
        elseif arg == "--save-warm-start" && i < length(args)
            i += 1
            options["warm_start_file"] = args[i]
        elseif arg == "--save-relaxed" && i < length(args)
            i += 1
            options["save_relaxed_file"] = args[i]
        elseif arg == "--relax-level" && i < length(args)
            i += 1
            lvl_str = lowercase(args[i])
            if lvl_str == "ac1"
                options["start_level"] = 6  # internal index for AC1
            else
                # Audit fix (missing guard): tryparse + clear error message
                lvl_parsed = tryparse(Int, lvl_str)
                if lvl_parsed === nothing
                    println("ERROR: --relax-level expects an integer 0-5 or \"ac1\", got: $(args[i])")
                    exit(1)
                end
                if !(0 <= lvl_parsed <= 5)
                    println("ERROR: --relax-level must be 0-5 or \"ac1\", got: $lvl_parsed")
                    exit(1)
                end
                options["start_level"] = lvl_parsed
            end
        elseif arg == "--interface-file" && i < length(args)
            i += 1
            options["interface_file"] = args[i]
        elseif arg == "--no-progressive"
            options["progressive"] = false
        elseif arg == "--warmstart-only"
            options["warmstart_only"] = true
        elseif arg == "--verbose"
            options["verbose"] = true
        elseif !startswith(arg, "-") && options["model_file"] === nothing
            options["model_file"] = arg
        else
            @warn "Unknown argument: $arg"
        end
        i += 1
    end

    return options
end

function print_usage()
    println("""
    Usage: julia run_opf_relaxation.jl <model.json> [options]

    Options:
        --ac                Run AC-OPF (default)
        --dc                Run DC-OPF (faster, linear approximation)
        --soc               Run SOC relaxation (convex)
        --output FILE       Save results to JSON file
        --dc-output FILE    Save DC warm-start results to JSON (AC mode only)
        --save-warm-start F Save warm-started model (DC-derived shunts) to JSON
        --save-relaxed F    Save mutated relaxed model (solves cold-strict) to JSON
        --interface-file F  Load interface limits from separate JSON file
        --relax-level N     Start at relaxation level N (0-5, or ac1)
        --no-progressive    Run ONLY at --relax-level (no escalation);
                            default progressive mode tries L0→AC1→L1..L5
        --warmstart-only    Do DC warm-start and exit (subprocess mode)
        --verbose           Show detailed solver output
    """)
    # Auto-generated from shared/relaxation_levels.json
    println("    Relaxation Levels:")
    for (i, lv) in enumerate(RELAXATION_LEVELS)
        label = get(lv, "label", "L$(i-1)")
        pad_label = rpad(lowercase(label), 5)
        println("        $pad_label$(lv["name"]) ($(lv["description"]))")
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════
# Model Summary
# ═══════════════════════════════════════════════════════════════════════

function print_model_summary(data)
    n_buses = length(data["bus"])
    n_branches = length(data["branch"])
    n_gens = length(data["gen"])
    baseMVA = get(data, "baseMVA", 100.0)

    total_load_pu = 0.0
    if haskey(data, "load")
        for (_, load) in data["load"]
            total_load_pu += get(load, "pd", 0.0)
        end
    end

    total_pmax_pu = sum(
        gen["pmax"] for (_, gen) in data["gen"]
        if get(gen, "gen_status", 1) == 1
    )

    total_load = total_load_pu * baseMVA
    total_pmax = total_pmax_pu * baseMVA

    println("\n" * "="^60)
    println("MODEL SUMMARY")
    println("="^60)
    println("  Buses:       $n_buses")
    println("  Branches:    $n_branches")
    println("  Generators:  $n_gens")
    println()
    println("  Total load:     $(round(total_load, digits=1)) MW")
    println("  Total capacity: $(round(total_pmax, digits=1)) MW")

    if total_pmax > 0 && total_load > 0
        reserve = (total_pmax - total_load) / total_load * 100
        ratio = total_load / total_pmax * 100
        println("  Load/capacity:  $(round(ratio, digits=1))%")
        println("  Reserve margin: $(round(reserve, digits=1))%")
    elseif total_pmax > 0
        println("  Load/capacity:  0.0% (no load)")
    end
    println("="^60)
end

# ═══════════════════════════════════════════════════════════════════════
# Solver Tolerances (single source of truth)
# ═══════════════════════════════════════════════════════════════════════
const SOLVER_TOL          = 1e-4   # primary feasibility / optimality tolerance
const SOLVER_ACCEPTABLE   = 1e-2   # fallback "acceptable" tolerance (Ipopt)
const SOLVER_MAX_ITER     = 10000  # main solve iteration limit
const SOLVER_MAX_TIME     = 900.0 # max CPU time per solve attempt (seconds)
const WALL_TIME_LIMIT     = 900.0 # max wall-clock time per level (seconds) — Ipopt-side guard
const DC_WARMSTART_ITER   = 5000   # DC warm-start iteration limit

# ═══════════════════════════════════════════════════════════════════════
# Unit Commitment (Generator Decommitment)
# ═══════════════════════════════════════════════════════════════════════

"""
    decommit_generators!(data) -> Int

Heuristic unit commitment: when the sum of all online generators' minimum
power (∑pmin) exceeds total demand, selectively zero out pmin for the most
expensive dispatchable generators until the system is feasible.

This mimics real-world day-ahead unit commitment where grid operators shut
down expensive peakers and mid-merit plants during low-demand periods.
Without this, the OPF would need pmin relaxation (L3+) to be feasible —
which blankets ALL generators (physically unrealistic).

**Method:** Instead of toggling gen_status (which would remove the generator
entirely), we set pmin=0 for selected generators.  This allows the OPF to
dispatch them at zero real power (effectively off) while keeping them
grid-connected for reactive power / voltage support — crucial for AC-OPF
convergence, especially in small networks.

**Decommit priority** (first to relax):
1. Highest marginal cost (linear coefficient of cost polynomial)
2. Ties broken by smallest capacity (peakers before baseload)

**Protected generators** (never decommitted):
- Nuclear (baseload, physical constraints prevent cycling)
- Renewables: solar, wind, hydro, geothermal (zero marginal cost, must-take)
- Battery / energy storage

Returns the number of generators with pmin zeroed.
"""
function decommit_generators!(data)
    baseMVA = get(data, "baseMVA", 100.0)

    # ── Compute totals (per-unit) ──
    total_demand_pu = 0.0
    if haskey(data, "load")
        total_demand_pu = sum(get(load, "pd", 0.0) for (_, load) in data["load"])
    end

    total_pmin_pu = 0.0
    total_pmax_pu = 0.0
    for (_, gen) in data["gen"]
        if get(gen, "gen_status", 1) == 1
            total_pmin_pu += get(gen, "pmin", 0.0)
            total_pmax_pu += get(gen, "pmax", 0.0)
        end
    end

    # Nothing to do if pmin fits within demand
    if total_pmin_pu <= total_demand_pu
        return 0
    end

    println("\n── Unit commitment ──")
    println("  Total pmin $(round(total_pmin_pu * baseMVA, digits=0)) MW > " *
            "demand $(round(total_demand_pu * baseMVA, digits=0)) MW — decommitting expensive generators")

    # ── Protected fuel types (must-run / zero marginal cost) ──
    PROTECTED_KEYWORDS = ["nuclear", "hydro", "wind", "solar", "geothermal", "battery"]

    function _is_protected(fuel_type)
        if fuel_type === nothing || fuel_type == ""
            return false
        end
        # Audit fix (review #2): word-boundary match to avoid false positives
        # like "hydrogen" matching "hydro". Splits on non-alphanumeric so
        # multi-word fuel types like "natural_gas" / "hydro-electric" still
        # work. Protected if ANY token equals a protected keyword.
        ft_lower = lowercase(string(fuel_type))
        tokens = split(ft_lower, r"[^a-z0-9]+"; keepempty=false)
        return any(tok -> tok in PROTECTED_KEYWORDS, tokens)
    end

    # ── Build candidate list ──
    candidates = []
    for (gid, gen) in data["gen"]
        if get(gen, "gen_status", 1) != 1
            continue
        end

        fuel = get(gen, "fuel_type", nothing)
        if _is_protected(fuel)
            continue
        end

        pmin = get(gen, "pmin", 0.0)
        pmax = get(gen, "pmax", 0.0)

        # Skip generators that already have pmin ≈ 0
        if pmin < 1e-6
            continue
        end

        # Extract marginal cost from cost polynomial
        # PowerModels format: ncost=N, cost=[c_{N-1}, ..., c_1, c_0]
        # Use linear coefficient c_1 (dominant term) for ranking
        ncost = get(gen, "ncost", 0)
        cost_arr = get(gen, "cost", [])
        if ncost >= 2 && length(cost_arr) >= 2
            marginal_cost = cost_arr[end-1]   # c_1 linear coefficient
        else
            marginal_cost = 0.0
        end

        name = get(gen, "name", "?")
        push!(candidates, (gid=gid, marginal_cost=marginal_cost,
                           pmin=pmin, pmax=pmax,
                           fuel=fuel !== nothing ? string(fuel) : "unknown",
                           name=name))
    end

    if isempty(candidates)
        println("    No decommit candidates (all generators are nuclear/renewable)")
        return 0
    end

    # Sort: highest marginal cost first; tie-break by smallest capacity
    sort!(candidates, by=c -> (-c.marginal_cost, c.pmax))

    # ── Decommit loop: set pmin=0 for expensive generators ──
    TARGET_RATIO = 0.95  # target: total_pmin ≤ 95% of demand

    target_pmin = total_demand_pu * TARGET_RATIO

    n_decommitted = 0
    freed_pmin_mw = 0.0

    for c in candidates
        if total_pmin_pu <= target_pmin
            break
        end

        # Zero out pmin (generator stays online for Q support, can dispatch P≥0)
        data["gen"][c.gid]["pmin"] = 0.0
        total_pmin_pu -= c.pmin
        n_decommitted += 1
        freed_pmin_mw += c.pmin * baseMVA
    end

    if n_decommitted > 0
        println("    Set pmin=0 on $n_decommitted generator(s) (freed $(round(Int, freed_pmin_mw)) MW of pmin)")
        println("    Remaining pmin: $(round(Int, total_pmin_pu * baseMVA)) MW, " *
                "demand: $(round(Int, total_demand_pu * baseMVA)) MW")
    end
    # Audit fix (review #3): if we couldn't hit the target, say so explicitly.
    # Previously this silently proceeded to the OPF which would then fail
    # with a confusing LOCALLY_INFEASIBLE instead of a clear diagnostic.
    if total_pmin_pu > target_pmin
        excess_mw = round(Int, (total_pmin_pu - target_pmin) * baseMVA)
        println("    ⚠️ DECOMMIT TARGET NOT REACHED: pmin still exceeds target by ≈$(excess_mw) MW " *
                "(all dispatchable gens already at pmin=0, only protected fuel types remain). " *
                "OPF may require L3+ pmin relaxation to solve.")
    end

    return n_decommitted
end

# ═══════════════════════════════════════════════════════════════════════
# Relaxation Levels  (loaded from shared/relaxation_levels.json — single source of truth)
# ═══════════════════════════════════════════════════════════════════════

# Locate the JSON relative to this script. Tries two canonical layouts:
#   1. scripts + shared/ co-located    <dir>/shared/relaxation_levels.json  (self-contained OSS layout)
#   2. scripts at <repo>/topo_solver_pipe/   shared at <repo>/shared/        (one-up, legacy)
# so the SAME script works in either layout without editing.
function _find_shared_json()
    for cand in [
        joinpath(@__DIR__, "shared", "relaxation_levels.json"),            # self-contained
        joinpath(dirname(@__DIR__), "shared", "relaxation_levels.json"),   # legacy one-up
    ]
        isfile(cand) && return cand
    end
    error("relaxation_levels.json not found in ./shared or ../shared relative to $(@__DIR__)")
end
const _RELAX_JSON = _find_shared_json()

function _load_relaxation_levels()
    raw = JSON.parsefile(_RELAX_JSON)
    levels = Vector{Dict{String,Any}}()
    for lv in raw["levels"]
        d = Dict{String,Any}()
        for (k, v) in lv
            # JSON nulls become `nothing`
            d[k] = v === nothing ? nothing : v
        end
        push!(levels, d)
    end
    return levels
end

const RELAXATION_LEVELS = _load_relaxation_levels()

"""
    level_label(level::Int) -> String

Human-readable label for a relaxation level.
Uses the "label" field from relaxation_levels.json.
"""
function level_label(level::Int)
    idx = level + 1  # Julia is 1-indexed
    if 1 <= idx <= length(RELAXATION_LEVELS)
        return get(RELAXATION_LEVELS[idx], "label", "L$level")
    end
    return "L$level"
end

"""
    fix_impedance_consistency!(data)

Ensure `rate_a × x ≤ π/2` (90°) for every branch.  PowerModels caps
branch angle differences at ±90° (via correct_voltage_angle_differences!),
so if the thermal rating requires a larger angle the branch can never reach
its rated capacity and the DC-OPF becomes infeasible.

This typically happens when the pipeline aggregates N parallel circuits
into one high-capacity branch but keeps single-circuit impedance.  The fix
is equivalent to treating the branch as proper parallel circuits (combined x).

Returns the number of adjusted branches.
"""
function fix_impedance_consistency!(data)
    max_angle = π / 2  # 90° — PowerModels' hard cap
    n_fixed = 0
    for (_, br) in data["branch"]
        if get(br, "br_status", 1) != 1
            continue
        end
        rate_a = get(br, "rate_a", 0.0)
        x = get(br, "br_x", 0.0)
        # Audit fix (review #4): also handle x<0 (series capacitors / TCSC).
        # Previously skipped, which meant series-comp branches could have
        # |rate_a·x| > π/2 and silently break DC-OPF.
        if rate_a > 0 && abs(x) > 0 && rate_a * abs(x) > max_angle
            # Preserve sign (series capacitors have x<0 deliberately)
            br["br_x"] = sign(x) * max_angle / rate_a
            n_fixed += 1
        end
    end
    return n_fixed
end

"""
Apply a single relaxation level to the model data (in-place).
Returns a list of human-readable change descriptions.
"""
function apply_relaxation!(data, level::Int)
    if level < 0 || level >= length(RELAXATION_LEVELS)
        error("Invalid relaxation level: $level (valid: 0-$(length(RELAXATION_LEVELS)-1))")
    end

    relax = RELAXATION_LEVELS[level + 1]  # Julia is 1-indexed
    changes = String[]

    # ── Angle relaxation ──
    if relax["angle_deg"] !== nothing
        angle_rad = relax["angle_deg"] * π / 180.0
        count = 0
        for (_, br) in data["branch"]
            angmin = get(br, "angmin", -0.52)
            angmax = get(br, "angmax", 0.52)
            if angmin > -angle_rad || angmax < angle_rad
                br["angmin"] = min(angmin, -angle_rad)
                br["angmax"] = max(angmax, angle_rad)
                count += 1
            end
        end
        if count > 0
            push!(changes, "Widened angles to ±$(Int(relax["angle_deg"]))° on $count branches")
        end
    end

    # ── Thermal relaxation ──
    # Level 5 specifically uses thermal_factor=nothing to mean "remove thermal limits entirely".
    # AC1 (level 6) also has thermal_factor=nothing but means "don't touch thermal limits".
    if level == 5 && relax["thermal_factor"] === nothing
        for (_, br) in data["branch"]
            br["rate_a"] = 1e6
            br["rate_b"] = 1e6
            br["rate_c"] = 1e6
        end
        push!(changes, "Removed thermal limits (set to 1e6)")
    elseif relax["thermal_factor"] !== nothing
        factor = relax["thermal_factor"]
        count = 0
        for (_, br) in data["branch"]
            # Use original ratings as base to avoid compounding across levels.
            # _orig_rate_X is saved on first thermal relaxation and reused
            # for all subsequent levels so L3 gives ×1.5 (not ×1.2 × 1.5 = ×1.8).
            if !haskey(br, "_orig_rate_a")
                br["_orig_rate_a"] = get(br, "rate_a", 0.0)
                br["_orig_rate_b"] = get(br, "rate_b", br["_orig_rate_a"])
                br["_orig_rate_c"] = get(br, "rate_c", br["_orig_rate_a"])
            end
            orig = br["_orig_rate_a"]
            if orig > 0 && orig < 1e5
                br["rate_a"] = orig * factor
                br["rate_b"] = br["_orig_rate_b"] * factor
                br["rate_c"] = br["_orig_rate_c"] * factor
                count += 1
            end
        end
        if count > 0
            push!(changes, "Scaled ratings ×$(factor) on $count branches")
        end
    end

    # ── Load shedding ──
    if relax["load_cap_ratio"] !== nothing
        baseMVA = get(data, "baseMVA", 100.0)
        total_pmax_pu = sum(
            gen["pmax"] for (_, gen) in data["gen"]
            if get(gen, "gen_status", 1) == 1
        )
        total_load_pu = 0.0
        if haskey(data, "load")
            total_load_pu = sum(get(load, "pd", 0.0) for (_, load) in data["load"])
        end

        max_load_pu = total_pmax_pu * relax["load_cap_ratio"]
        if total_load_pu > max_load_pu && total_load_pu > 0
            scale = max_load_pu / total_load_pu
            for (_, load) in data["load"]
                load["pd"] = get(load, "pd", 0.0) * scale
                load["qd"] = get(load, "qd", 0.0) * scale
            end
            shed_mw = (total_load_pu - max_load_pu) * baseMVA
            push!(changes, "Shed $(round(Int, shed_mw)) MW load (capped at $(round(Int, relax["load_cap_ratio"]*100))% capacity)")
        end
    end

    # ── Voltage bound relaxation (AC-specific) ──
    if haskey(relax, "vmin") && relax["vmin"] !== nothing
        target_vmin = relax["vmin"]
        target_vmax = relax["vmax"]
        v_count = 0
        for (_, bus) in data["bus"]
            old_vmin = get(bus, "vmin", 0.94)
            old_vmax = get(bus, "vmax", 1.06)
            new_vmin = min(old_vmin, target_vmin)
            new_vmax = max(old_vmax, target_vmax)
            if new_vmin < old_vmin || new_vmax > old_vmax
                bus["vmin"] = new_vmin
                bus["vmax"] = new_vmax
                v_count += 1
            end
        end
        if v_count > 0
            push!(changes, "Widened voltage bounds to [$target_vmin, $target_vmax] on $v_count buses")
        end
    end

    # ── Generator Q limit relaxation (AC-specific) ──
    if haskey(relax, "q_factor") && relax["q_factor"] !== nothing
        qf = relax["q_factor"]
        q_count = 0
        for (_, gen) in data["gen"]
            if get(gen, "gen_status", 1) != 1
                continue
            end
            # Save original Q limits on first Q relaxation to avoid compounding
            # (same pattern as _orig_rate_a for thermal limits).
            if !haskey(gen, "_orig_qmax")
                gen["_orig_qmax"] = get(gen, "qmax", 0.0)
                gen["_orig_qmin"] = get(gen, "qmin", 0.0)
            end
            qmax = gen["_orig_qmax"]
            qmin = gen["_orig_qmin"]
            # Only scale if the generator has meaningful Q range
            if abs(qmax - qmin) > 1e-6
                gen["qmax"] = qmax * qf
                gen["qmin"] = qmin * qf  # qmin is usually negative, scaling preserves sign
                q_count += 1
            elseif abs(qmax) < 1e-6 && abs(qmin) < 1e-6
                # Generator has zero Q capability — give it some based on Pmax
                pmax = get(gen, "pmax", 0.0)
                if pmax > 1e-6
                    gen["qmax"] = pmax * 0.3 * qf   # typical power factor ~0.95 → Q ≈ 0.3P
                    gen["qmin"] = -pmax * 0.1 * qf
                    q_count += 1
                end
            end
        end
        if q_count > 0
            push!(changes, "Scaled Q limits ×$(qf) on $q_count generators")
        end
    end

    # ── Pmin relaxation (for off-peak where total pmin > demand) ──
    if haskey(relax, "pmin_factor") && relax["pmin_factor"] !== nothing
        pf = relax["pmin_factor"]
        pmin_count = 0
        for (_, gen) in data["gen"]
            if get(gen, "gen_status", 1) != 1
                continue
            end
            old_pmin = get(gen, "pmin", 0.0)
            if old_pmin > 1e-6
                gen["pmin"] = old_pmin * pf
                pmin_count += 1
            end
        end
        if pmin_count > 0
            if pf == 0.0
                push!(changes, "Set pmin = 0 on $pmin_count generators")
            else
                push!(changes, "Scaled pmin ×$(pf) on $pmin_count generators")
            end
        end
    end

    # ── Impedance/capacity consistency ──
    # Must come after thermal scaling since rate_a may have changed.
    # Skip at L5 where rate_a is set to 1e6 (removing thermal limits) —
    # otherwise we'd crush all br_x to ~1e-6, destroying the network.
    if !(level == 5 && relax["thermal_factor"] === nothing)
        n_x_fix = fix_impedance_consistency!(data)
        if n_x_fix > 0
            push!(changes, "Adjusted x on $n_x_fix branches for angle feasibility")
        end
    end

    return changes
end

# ═══════════════════════════════════════════════════════════════════════
# DC-Derived Reactive Compensation
# ═══════════════════════════════════════════════════════════════════════

"""
    inject_dc_derived_shunts!(data, dc_sol)

Use the DC-OPF solution to estimate per-bus reactive power needs and
inject/augment shunts in `data` before the AC-OPF solve.

**Physical rationale:**
DC-OPF gives us the real power flow pattern (bus angles → branch P flows).
From these we can estimate each branch's reactive losses and line charging,
giving a bus-level Q-balance that reveals where reactive compensation is
actually needed under the solved dispatch — far more accurate than the
heuristic "load Qd vs local gen Qmax" in the Python preprocessing.

**Method:**
1.  Extract bus voltage angles (θ) from DC solution.
2.  For each branch, compute P flow:  Pij = (θi − θj) / xij   [pu].
3.  Estimate branch reactive absorption:  Qloss ≈ Pij² · xij   [pu].
4.  Credit branch line charging:  Qcharge = b/2  at each end   [pu].
5.  At each bus, sum:
      Q_needed = Qd_load + ΣQ_loss_half  −  ΣQ_charge
      Q_available = gen_Qmax + existing_shunt_bs
    where Q_loss is split equally between from/to buses.
6.  Where Q_needed > Q_available, add a shunt sized to the deficit × margin.

The shunts are added in per-unit (same unit system as the parsed PowerModels
data).  They will participate in the AC-OPF's reactive power balance.
"""
function inject_dc_derived_shunts!(data, dc_sol)
    # Audit fix (review #10): idempotency guard. Previous calls leave a flag
    # in data; a second call skips to avoid double-injection.
    if get(data, "_dc_shunts_injected", false) == true
        println("  inject_dc_derived_shunts!: already injected, skipping (idempotent guard)")
        return 0
    end

    baseMVA = get(data, "baseMVA", 100.0)
    # Audit fix: guards on required top-level dicts (review: missing haskey).
    # Without these, an unusual JSON crashes with KeyError instead of a clear
    # error message.
    for k in ("branch", "bus", "gen")
        haskey(data, k) || error("inject_dc_derived_shunts!: data is missing required key \"$k\"")
    end
    branches = data["branch"]
    buses = data["bus"]
    gens = data["gen"]
    loads = get(data, "load", Dict())
    shunts = get(data, "shunt", Dict())

    # ── 1. Bus voltage angles from DC solution ──
    bus_va = Dict{String, Float64}()
    if haskey(dc_sol, "bus")
        for (bid, bsol) in dc_sol["bus"]
            bus_va[bid] = get(bsol, "va", 0.0)
        end
    end
    if isempty(bus_va)
        println("  ⚠️ No bus angles in DC solution — skipping DC-derived shunts")
        return 0  # consistent Int return; previously `return` leaked `nothing`
                  # into `n_dc_shunts` which callers stored into _warm_start dict
    end

    # ── 2. Per-bus Q demand from loads ──
    bus_qd = Dict{String, Float64}()
    for (_, load) in loads
        bid = string(load["load_bus"])
        qd = get(load, "qd", 0.0)
        bus_qd[bid] = get(bus_qd, bid, 0.0) + qd
    end

    # ── 3. Per-bus generator reactive capability ──
    bus_qmax = Dict{String, Float64}()
    for (_, gen) in gens
        if get(gen, "gen_status", 1) != 1
            continue
        end
        bid = string(gen["gen_bus"])
        qmax = max(get(gen, "qmax", 0.0), 0.0)
        bus_qmax[bid] = get(bus_qmax, bid, 0.0) + qmax
    end

    # ── 4. Per-bus existing shunt compensation ──
    bus_shunt_bs = Dict{String, Float64}()
    for (_, sh) in shunts
        bid = string(sh["shunt_bus"])
        bs = get(sh, "bs", 0.0)
        bus_shunt_bs[bid] = get(bus_shunt_bs, bid, 0.0) + bs
    end

    # ── 5. Branch reactive losses and line charging ──
    bus_qloss = Dict{String, Float64}()     # Q absorbed by branches (split 50/50)
    bus_qcharge = Dict{String, Float64}()   # Q injected by line charging

    # Signed per-bus shunt compensation (separate injection vs absorption)
    bus_qcharge_ind = Dict{String, Float64}()  # inductive shunts in branches (b<0), Q absorption
    for (_, br) in branches
        if get(br, "br_status", 1) != 1
            continue
        end
        fbid = string(br["f_bus"])
        tbid = string(br["t_bus"])
        x = get(br, "br_x", 0.0)
        tap = get(br, "tap", 1.0)
        # PowerModels uses split π-model: b_fr and b_to (half line charging at each end)
        b_fr = get(br, "b_fr", 0.0)
        b_to = get(br, "b_to", 0.0)

        # Get bus angles (default 0.0 if not in DC solution)
        va_f = get(bus_va, fbid, 0.0)
        va_t = get(bus_va, tbid, 0.0)

        # Active power flow through branch (DC approximation).
        # Audit fix (review #5): for transformers, x is the series reactance
        # in the T-equivalent; DC flow uses x directly with 1/tap scaling on
        # the from side. Approximation: use |x|/tap² as the effective series
        # reactance for the Q-loss estimate on transformers.
        x_eff = (tap != 0.0 && abs(tap - 1.0) > 1e-8) ? abs(x) / tap^2 : abs(x)
        if x_eff > 1e-10
            p_flow = (va_f - va_t) / x_eff   # pu
        else
            p_flow = 0.0  # HVDC or zero-impedance (skip)
        end

        # Reactive losses: I²X_eff ≈ P²·X_eff (at V≈1.0 pu).
        q_loss = p_flow^2 * x_eff  # pu, always positive

        # Split losses 50/50 between from and to buses
        bus_qloss[fbid] = get(bus_qloss, fbid, 0.0) + q_loss / 2
        bus_qloss[tbid] = get(bus_qloss, tbid, 0.0) + q_loss / 2

        # Audit fix (review #6): handle BOTH signs of b_fr / b_to.
        # Positive = line charging (Q injection); negative = inductive branch
        # shunt (Q absorption). Drop-sign-negative previously ignored inductive
        # contributions, understating Q-absorption capacity.
        if b_fr > 0
            bus_qcharge[fbid] = get(bus_qcharge, fbid, 0.0) + b_fr
        elseif b_fr < 0
            bus_qcharge_ind[fbid] = get(bus_qcharge_ind, fbid, 0.0) + abs(b_fr)
        end
        if b_to > 0
            bus_qcharge[tbid] = get(bus_qcharge, tbid, 0.0) + b_to
        elseif b_to < 0
            bus_qcharge_ind[tbid] = get(bus_qcharge_ind, tbid, 0.0) + abs(b_to)
        end
    end

    # ── 6a. Per-bus generator Q absorption capability (qmin) ──
    bus_qmin = Dict{String, Float64}()
    for (_, gen) in gens
        if get(gen, "gen_status", 1) != 1
            continue
        end
        bid = string(gen["gen_bus"])
        qmin = min(get(gen, "qmin", 0.0), 0.0)  # Only negative (absorption)
        bus_qmin[bid] = get(bus_qmin, bid, 0.0) + qmin
    end

    # ── 6b. Compute per-bus Q deficit/surplus and inject shunts ──
    # Q balance at each bus:
    #   Q_needed  = Qd (load) + Q_loss (half of connected branches)
    #   Q_supply  = gen Qmax + existing shunt bs + line charging
    #   deficit   = Q_needed - Q_supply  → add capacitive shunt
    #
    # Also check the SURPLUS case (common with long HV lines):
    #   Q_excess  = line_charging - Qd - Q_loss
    #   Q_absorption = |gen Qmin| + existing inductive shunts
    #   surplus   = Q_excess - Q_absorption  → add inductive shunt (reactor)
    #
    # Real grids install shunt reactors to absorb excess line charging;
    # OSM models lack these, so we add them here.
    MARGIN = 1.15  # 15% margin to cover approximation errors
    # Audit fix (review #9): cap injected bs at a physically plausible level.
    # Unclamped bs on pathological DC flows produced multi-GVAr single-bus
    # shunts that let AC-OPF "solve" at unphysical voltages. Real utility
    # shunt banks are typically 10-300 MVAr per install (0.1-3 pu at 100 MVA
    # base). A cap of 3 pu = 300 MVAr allows generous real compensation while
    # preventing runaway injections that mask data-quality issues.
    SHUNT_BS_MAX = 3.0  # pu

    # Find next shunt index
    max_shunt_idx = 0
    for (k, _) in shunts
        idx = tryparse(Int, k)
        if idx !== nothing && idx > max_shunt_idx
            max_shunt_idx = idx
        end
    end
    shunt_id = max_shunt_idx + 1

    n_cap_added = 0
    n_ind_added = 0
    total_cap_bs = 0.0
    total_ind_bs = 0.0
    n_clamped    = 0   # count of buses where SHUNT_BS_MAX cap kicked in

    for (bid, _) in buses
        qd = get(bus_qd, bid, 0.0)
        qloss = get(bus_qloss, bid, 0.0)
        qcharge = get(bus_qcharge, bid, 0.0)
        qmax_gen = get(bus_qmax, bid, 0.0)
        qmin_gen = get(bus_qmin, bid, 0.0)  # negative
        existing_bs = get(bus_shunt_bs, bid, 0.0)

        bus_id_int = tryparse(Int, bid)
        if bus_id_int === nothing
            continue
        end

        # --- Case 1: Q deficit (need more Q than available) ---
        # Audit fix (review #7): use SIGNED qd so capacitive loads (qd<0,
        # leading-PF customers) correctly reduce Q need.
        qcharge_ind = get(bus_qcharge_ind, bid, 0.0)  # inductive line shunts (Q absorbers)
        q_needed = qd * MARGIN + qloss + qcharge_ind
        q_available = qmax_gen + max(existing_bs, 0.0) + qcharge
        deficit = q_needed - q_available

        if deficit > 1e-4  # pu threshold
            # Audit fix (review #9): cap bs at SHUNT_BS_MAX to prevent
            # multi-GVAr single-bus shunts from masking unphysical data.
            bs_inj = min(deficit, SHUNT_BS_MAX)
            bs_inj < deficit && (n_clamped += 1)
            shunts[string(shunt_id)] = Dict(
                "index" => shunt_id,
                "shunt_bus" => bus_id_int,
                "gs" => 0.0,
                "bs" => bs_inj,  # Capacitive (positive), clamped
                "status" => 1,
            )
            shunt_id += 1
            n_cap_added += 1
            total_cap_bs += bs_inj
            continue  # Deficit and surplus are mutually exclusive per bus
        end

        # --- Case 2: Q surplus from line charging (gens can't absorb) ---
        if qcharge > 1e-4
            # Audit fix (review #7): signed qd — capacitive loads add to surplus.
            q_excess = (qcharge - qd - qloss) * MARGIN
            # Q absorption available: |qmin| of generators + inductive shunts
            # (both branch-side inductive shunts and existing inductive bus shunts)
            q_absorption = abs(qmin_gen) + abs(min(existing_bs, 0.0)) + qcharge_ind
            surplus = q_excess - q_absorption

            if surplus > 1e-4
                # Audit fix (review #9): cap magnitude at SHUNT_BS_MAX.
                bs_mag = min(surplus, SHUNT_BS_MAX)
                bs_mag < surplus && (n_clamped += 1)
                shunts[string(shunt_id)] = Dict(
                    "index" => shunt_id,
                    "shunt_bus" => bus_id_int,
                    "gs" => 0.0,
                    "bs" => -bs_mag,  # Inductive (negative) = reactor, clamped
                    "status" => 1,
                )
                shunt_id += 1
                n_ind_added += 1
                total_ind_bs += bs_mag
            end
        end
    end

    # Update the model
    data["shunt"] = shunts
    # Audit fix (review #10): set idempotency flag so a second call short-circuits.
    data["_dc_shunts_injected"] = true

    n_total = n_cap_added + n_ind_added
    if n_total > 0
        if n_cap_added > 0
            mvar_cap = total_cap_bs * baseMVA
            println("     DC-derived shunts: $n_cap_added capacitor(s) (+$(round(mvar_cap, digits=0)) MVAr)")
        end
        if n_ind_added > 0
            mvar_ind = total_ind_bs * baseMVA
            println("     DC-derived shunts: $n_ind_added reactor(s) (-$(round(mvar_ind, digits=0)) MVAr)")
        end
        if n_clamped > 0
            println("     ⚠️ SHUNT_BS_MAX ($(SHUNT_BS_MAX) pu) clamp kicked in on $n_clamped bus(es) — data may be pathological")
        end
        if length(bus_shunt_bs) > 0
            existing_total = sum(max(v, 0) for v in values(bus_shunt_bs)) * baseMVA
            println("     (preserving $(length(bus_shunt_bs)) existing shunts, +$(round(existing_total, digits=0)) MVAr)")
        end
    else
        println("  ✅ DC-derived shunt analysis: no additional compensation needed")
    end

    return n_total
end


# ═══════════════════════════════════════════════════════════════════════
# OPF Execution
# ═══════════════════════════════════════════════════════════════════════
# Interface (Flowgate) Constraints
# ═══════════════════════════════════════════════════════════════════════

"""
Add inter-area transfer limit constraints to an instantiated PowerModel.

For each interface, constrains the sum of active power flows (pf) on the
member branches to lie within [-limit, +limit]:

    -limit ≤ Σ pf[branch_id] ≤ +limit

where limit is in per-unit (same as branch rate_a).  The branch_ids and
limit come from the model's "interface" section, built by the Python
interface_limits module.
"""
function add_interface_constraints!(pm, data, interface_data; scale_factor::Float64=1.0)
    n_added = 0
    for (iid, iface) in interface_data
        branch_ids_str = iface["branch_ids"]
        limit = iface["limit"] * scale_factor
        name = get(iface, "name", "iface_$iid")

        # Collect the pf variables for each branch in this interface group
        p_var = var(pm, :p)
        p_vars = []
        for bid_str in branch_ids_str
            bid = parse(Int64, bid_str)
            branch = get(data["branch"], bid_str, nothing)
            if branch === nothing
                continue
            end
            f_bus = branch["f_bus"]
            t_bus = branch["t_bus"]
            f_idx = (bid, f_bus, t_bus)
            # PowerModels stores branch power flow as p[f_idx] in a DenseAxisArray
            try
                push!(p_vars, p_var[f_idx])
            catch
                # Branch arc not present in model variables — skip
            end
        end

        if isempty(p_vars)
            continue
        end

        # Add bidirectional transfer limit: -limit ≤ Σpf ≤ +limit
        @constraint(pm.model, sum(p_vars) <= limit)
        @constraint(pm.model, sum(p_vars) >= -limit)
        n_added += 1
    end

    if n_added > 0
        sf_str = scale_factor == 1.0 ? "" : " (×$(scale_factor))"
        println("  Added $n_added interface flow limit constraint(s)$sf_str")
    end
    return n_added
end

"""
Compute the interface limit scale factor for a given relaxation level.
Matches the progressive relaxation philosophy: tighter at L0, removed at L5.
  L0:  ×1.0  (strict)
  AC1: ×1.0  (voltage/Q only, interfaces unchanged)
  L1:  ×1.0  (angle relaxation only)
  L2:  ×1.2  (matches thermal_factor)
  L3:  ×1.5  (matches thermal_factor)
  L4:  ×2.0  (aggressive — load shedding helps too)
  L5:  removed (no interface constraints)
"""
function interface_scale_for_level(level::Int)
    if level <= 1 || level == 6  # L0, L1, AC1
        return 1.0
    elseif level == 2
        return 1.2
    elseif level == 3
        return 1.5
    elseif level == 4
        return 2.0
    else  # level >= 5
        return Inf  # signal to skip interface constraints entirely
    end
end

# ═══════════════════════════════════════════════════════════════════════

"""
Solve OPF at a given relaxation level.
Makes a deep copy of data so each attempt preserves warm-start values
from previous levels. Applies relaxations cumulatively from level 1 up
to the target level.

For AC-OPF, levels 1-5 include AC1 (voltage + Q relaxation) as a base
layer — this ensures the escalation is monotonically more relaxed.
DC-OPF skips AC1 entirely.

If interface_data is provided, the solver uses instantiate_model + optimize_model!
so that custom branch-group flow constraints can be injected between model
construction and solving.
"""
function solve_with_level(data, options, level; interface_data=nothing)
    trial_data = deepcopy(data)
    is_ac = (options["formulation"] == "ac" && !options["soc"])

    # Level 6 (AC1): standalone voltage + Q relaxation.
    # Levels 1-5: cumulative L1..level.  For AC-OPF, AC1 is prepended as a
    # base layer so that L1-L5 never regress below AC1's V/Q relaxation.
    all_changes = String[]
    if level == 6
        changes = apply_relaxation!(trial_data, 6)
        append!(all_changes, changes)
    else
        if is_ac && level >= 1
            # AC base layer: apply AC1 (voltage + Q) before cumulative chain
            changes = apply_relaxation!(trial_data, 6)
            append!(all_changes, changes)
        end
        for l in 1:level
            changes = apply_relaxation!(trial_data, l)
            append!(all_changes, changes)
        end
    end

    # Configure solver
    solver = optimizer_with_attributes(
        Ipopt.Optimizer,
        "tol" => SOLVER_TOL,
        "acceptable_tol" => SOLVER_ACCEPTABLE,
        "max_iter" => SOLVER_MAX_ITER,
        "max_cpu_time" => SOLVER_MAX_TIME,
        "max_wall_time" => WALL_TIME_LIMIT,
        "print_level" => options["verbose"] ? 5 : 0,
        "warm_start_init_point" => "yes",
    )

    # Fix impedance/capacity consistency at L0 (apply_relaxation! handles L1+)
    if level == 0
        fix_impedance_consistency!(trial_data)
    end

    # Run OPF — use two-step build+solve when interface constraints exist
    iface_scale = interface_scale_for_level(level)
    use_iface = interface_data !== nothing && !isempty(interface_data) && isfinite(iface_scale)
    if use_iface
        # Determine formulation type
        if options["soc"]
            pm = instantiate_model(trial_data, SOCWRPowerModel, PowerModels.build_opf)
        elseif options["formulation"] == "ac"
            pm = instantiate_model(trial_data, ACPPowerModel, PowerModels.build_opf)
        else
            pm = instantiate_model(trial_data, DCPPowerModel, PowerModels.build_opf)
        end

        # Add interface flow constraints (scaled for this relaxation level)
        add_interface_constraints!(pm, trial_data, interface_data; scale_factor=iface_scale)

        # Solve
        result = optimize_model!(pm, optimizer=solver)
    else
        # Standard one-step solve (no interface constraints)
        if options["soc"]
            result = solve_opf(trial_data, SOCWRPowerModel, solver)
        elseif options["formulation"] == "ac"
            result = solve_ac_opf(trial_data, solver)
        else
            result = solve_dc_opf(trial_data, solver)
        end
    end

    return result, trial_data, all_changes
end

function is_solved(result)
    status = result["termination_status"]
    return status == LOCALLY_SOLVED || status == OPTIMAL
end

function is_almost_solved(result)
    return result["termination_status"] == ALMOST_LOCALLY_SOLVED
end

function is_acceptable(result)
    return is_solved(result) || is_almost_solved(result)
end

# ═══════════════════════════════════════════════════════════════════════
# Results Formatting
# ═══════════════════════════════════════════════════════════════════════

function print_results(result, data, solve_time, solved_level, all_changes, max_level)
    baseMVA = get(data, "baseMVA", 100.0)
    obj = result["objective"]
    status = result["termination_status"]

    # Compute totals
    total_pg = 0.0
    total_load_mw = 0.0
    if haskey(result, "solution") && haskey(result["solution"], "gen")
        total_pg = sum(gen["pg"] for (_, gen) in result["solution"]["gen"]) * baseMVA
    end
    if haskey(data, "load")
        total_load_mw = sum(get(load, "pd", 0.0) for (_, load) in data["load"]) * baseMVA
    end

    # Format cost for readability
    if obj >= 1e6
        obj_str = "$(round(obj / 1e6, digits=3)) M\$/hr"
    elseif obj >= 1e3
        obj_str = "$(round(obj / 1e3, digits=1)) k\$/hr"
    else
        obj_str = "$(round(obj, digits=2)) \$/hr"
    end

    avg_cost = total_load_mw > 0 ? obj / total_load_mw : 0.0

    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)
    println("  Status:       $(status)")
    println("  Solve time:   $(round(solve_time, digits=2)) sec")
    println()
    println("  Total cost:   $(obj_str)")
    println("  Total gen:    $(round(total_pg, digits=1)) MW")
    println("  Total load:   $(round(total_load_mw, digits=1)) MW")
    println("  Avg cost:     $(round(avg_cost, digits=1)) \$/MWh")
    println()

    if is_solved(result) || is_almost_solved(result)
        level_info = RELAXATION_LEVELS[solved_level + 1]
        approx_tag = is_almost_solved(result) ? " (approximate)" : ""
        icon = is_almost_solved(result) ? "⚠️" : "✅"
        if solved_level == 0
            println("  $(icon) Solved at L0 (Strict)$(approx_tag) — no relaxations needed")
            println("    Model parameters from pipeline are sufficient.")
        else
            lbl = level_label(solved_level)
            println("  $(icon) Solved at $(lbl) ($(level_info["name"]))$(approx_tag)")
            println("    $(level_info["description"])")
            if !isempty(all_changes)
                println()
                println("    Relaxations applied:")
                for change in all_changes
                    println("      • $change")
                end
            end
        end

        if is_almost_solved(result)
            println()
            println("    Note: Converged within acceptable_tol=1e-2 but not strict tol=1e-4.")
            println("    Solution satisfies constraints to ~1%. Standard for large networks.")
        end

        # Cost sanity check
        if avg_cost > 0
            println()
            if avg_cost < 10
                println("  ⚠️ Avg cost \$$(round(avg_cost, digits=1))/MWh is very low — check cost coefficients")
            elseif avg_cost > 200
                println("  ⚠️ Avg cost \$$(round(avg_cost, digits=1))/MWh is very high — may indicate congestion")
            else
                println("  ✅ Avg wholesale cost \$$(round(avg_cost, digits=1))/MWh (typical US: \$20-80/MWh)")
            end
        end
    else
        println("  ❌ OPF did not converge at any relaxation level (L0-$(level_label(max_level)))")
        println("    This suggests a fundamental model issue:")
        println("    • Check that generators have valid pmin/pmax ranges")
        println("    • Check that bus connectivity forms a connected graph")
        if max_level <= 5
            println("    • Consider running with --ac to use AC-specific relaxation (AC1)")
        end
        println("    • Try --verbose for detailed solver output")
    end
    println("="^60)
end

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

function run_opf(model_file::String, options::Dict)
    println("\nLoading model: $(model_file)")
    # import_all=false avoids pulling non-PM top-level dicts (e.g. _relaxation
    # from stage-2 outputs) into the data struct. The stripping loop below is
    # a secondary defense for files loaded with import_all=true elsewhere.
    # gridsfm_topo's *_model.json output omits "storage"/"switch"/"dcline"
    # top-level keys when those device types aren't modelled. PowerModels'
    # _check_connectivity (run inside parse_file when validate=true) indexes
    # data["storage"] unconditionally and crashes. Inject empty dicts BEFORE
    # parse so validation passes — keeps validate=true catching real issues.
    data = _parse_with_default_devices(model_file; import_all=false, validate=true)

    # Strip any remaining custom metadata keys whose dict keys aren't
    # integer-parseable. PowerModels/InfrastructureModels iterates every
    # top-level Dict{String,Any} and calls parse(Int64, key) — non-integer
    # keys like "TVA" cause a crash. First, extract interface limits before
    # they get stripped.
    interface_data = get(data, "interface", nothing)
    if interface_data !== nothing
        delete!(data, "interface")
        n_iface = length(interface_data)
        println("  Extracted $n_iface interface flow limit(s)")
    end

    # Load interface data from external file (used by subprocess progressive mode)
    if options["interface_file"] !== nothing && isfile(options["interface_file"])
        interface_data = JSON.parsefile(options["interface_file"])
        println("  Loaded $(length(interface_data)) interface limit(s) from $(options["interface_file"])")
    end
    for key in collect(keys(data))
        val = data[key]
        if isa(val, Dict) && !isempty(val)
            if any(k -> tryparse(Int64, k) === nothing, keys(val))
                delete!(data, key)
            end
        end
    end

    print_model_summary(data)

    # ── Unit commitment: decommit expensive generators if pmin > demand ──
    n_decommitted = decommit_generators!(data)

    formulation_str = options["soc"] ? "SOC-OPF" :
                      options["formulation"] == "dc" ? "DC-OPF" : "AC-OPF"
    # DC-OPF only uses levels 0-5 (angles/thermal/load);
    # AC-OPF can also use AC1 (voltage/Q relaxation)
    is_ac = (options["formulation"] == "ac" && !options["soc"])
    max_level = is_ac ? length(RELAXATION_LEVELS) - 1 : min(5, length(RELAXATION_LEVELS) - 1)
    # AC1 (level 6) is only meaningful for AC-OPF (voltage/Q relaxation).
    # Reject it explicitly in DC/SOC mode instead of silently clamping to L5.
    if !is_ac && options["start_level"] == 6
        error("AC1 relaxation level (--relax-level ac1) is only valid for AC-OPF. " *
              "Use L0-L5 for DC/SOC formulations, or drop --dc / --soc.")
    end
    start_level = min(options["start_level"], max_level)

    # ── DC warm-start for AC-OPF ──
    # Solve a quick DC-OPF first and use its solution (bus angles, generator
    # dispatch) as starting values for the AC solver. This gives Ipopt a much
    # better initial point than flat start (vm=1, va=0), helping it find the
    # correct local optimum — especially for large networks.
    # Uses progressive relaxation (up to L5) to ensure DC converges.
    if is_ac
        println("\n── DC warm-start for AC-OPF ──")
        # Note: dc_warmstart_base (below) is the mutable base used by the
        # DC escalation loop. Previous code allocated an unused second
        # deepcopy here — removed to avoid a needless full-network clone.
        dc_solver = optimizer_with_attributes(
            Ipopt.Optimizer,
            "tol" => SOLVER_TOL,
            "acceptable_tol" => SOLVER_ACCEPTABLE,
            "max_iter" => DC_WARMSTART_ITER,
            "max_cpu_time" => SOLVER_MAX_TIME,
            "max_wall_time" => WALL_TIME_LIMIT,
            "print_level" => 0,
        )

        # Try progressive DC relaxation for warm-start
        # Include interface constraints so DC dispatch respects transfer limits,
        # giving the AC solver a feasible starting point.
        dc_result = nothing
        dc_solved_level = -1
        dc_max_level = min(5, length(RELAXATION_LEVELS) - 1)
        dc_warmstart_base = deepcopy(data)  # Mutable base for warm-start propagation
        for dc_level in 0:dc_max_level
            dc_trial = deepcopy(dc_warmstart_base)
            for l in 1:dc_level
                apply_relaxation!(dc_trial, l)
            end

            # Fix impedance/capacity consistency (also done inside
            # apply_relaxation!, but needed at L0 where no relaxation is applied)
            # Skip at L5 where rate_a is set to 1e6 (removing thermal limits)
            # — same guard as in apply_relaxation!
            if dc_level < 5
                dc_x_fixes = fix_impedance_consistency!(dc_trial)
                if dc_x_fixes > 0 && dc_level == 0
                    println("  Adjusted x on $dc_x_fixes branches for DC angle feasibility")
                end
            end

            # Use two-step solve when interface constraints exist
            dc_iface_scale = interface_scale_for_level(dc_level)
            if interface_data !== nothing && !isempty(interface_data) && isfinite(dc_iface_scale)
                dc_pm = instantiate_model(dc_trial, DCPPowerModel, PowerModels.build_opf)
                add_interface_constraints!(dc_pm, dc_trial, interface_data; scale_factor=dc_iface_scale)
                dc_result = optimize_model!(dc_pm, optimizer=dc_solver)
            else
                dc_result = solve_dc_opf(dc_trial, dc_solver)
            end
            if dc_result["termination_status"] == LOCALLY_SOLVED ||
               dc_result["termination_status"] == OPTIMAL
                dc_solved_level = dc_level
                break
            end
            # Propagate partial solution as warm-start for next level
            partial = get(dc_result, "solution", nothing)
            if partial !== nothing
                if haskey(partial, "bus")
                    for (bid, bsol) in partial["bus"]
                        if haskey(dc_warmstart_base["bus"], bid)
                            dc_warmstart_base["bus"][bid]["va"] = get(bsol, "va", 0.0)
                        end
                    end
                end
                if haskey(partial, "gen")
                    for (gid, gsol) in partial["gen"]
                        if haskey(dc_warmstart_base["gen"], gid)
                            dc_warmstart_base["gen"][gid]["pg"] = get(gsol, "pg", 0.0)
                        end
                    end
                end
            end
        end

        if dc_solved_level >= 0
            dc_sol = dc_result["solution"]

            # Transfer bus voltage angles from DC solution
            va_count = 0
            if haskey(dc_sol, "bus")
                for (bus_id, bus_sol) in dc_sol["bus"]
                    if haskey(data["bus"], bus_id)
                        data["bus"][bus_id]["va"] = get(bus_sol, "va", 0.0)
                        va_count += 1
                    end
                end
            end

            # Transfer generator active power dispatch from DC solution
            pg_count = 0
            if haskey(dc_sol, "gen")
                for (gen_id, gen_sol) in dc_sol["gen"]
                    if haskey(data["gen"], gen_id)
                        data["gen"][gen_id]["pg"] = get(gen_sol, "pg", 0.0)
                        pg_count += 1
                    end
                end
            end

            # Set voltage magnitudes to 1.0 pu (DC doesn't solve these)
            for (_, bus) in data["bus"]
                bus["vm"] = 1.0
            end

            println("  ✅ DC-OPF solved (L$dc_solved_level) — warm-starting AC with $va_count bus angles, $pg_count gen dispatches")

            # ── Inject reactive compensation based on DC power flows ──
            n_dc_shunts = inject_dc_derived_shunts!(data, dc_sol)
        else
            println("  ❌ DC-OPF did not converge (L0-$dc_max_level) — using flat start for AC")
        end

        # ── Save warm-started model (with DC-derived shunts, angles, dispatch) ──
        if options["warm_start_file"] !== nothing
            ws_output = deepcopy(data)

            # Tag the model with warm-start metadata
            ws_output["_warm_start"] = Dict(
                "dc_solved_level" => dc_solved_level,
                "dc_objective" => dc_solved_level >= 0 ? dc_result["objective"] : nothing,
                "n_dc_shunts" => @isdefined(n_dc_shunts) ? n_dc_shunts : 0,
                "total_shunts" => length(get(ws_output, "shunt", Dict())),
                "vm_init" => 1.0,
                "warm_start_applied" => dc_solved_level >= 0,
            )

            open(options["warm_start_file"], "w") do f
                JSON.print(f, ws_output, 2)
            end
            println("\n  Warm-started model saved: $(options["warm_start_file"])")
        end

        # ── Save DC warm-start results alongside AC results ──
        if options["dc_output_file"] !== nothing && dc_solved_level >= 0
            baseMVA_dc = get(data, "baseMVA", 100.0)
            dc_total_load = haskey(data, "load") ?
                sum(get(load, "pd", 0.0) for (_, load) in data["load"]) * baseMVA_dc : 0.0
            dc_total_gen = 0.0
            if haskey(dc_result, "solution") && haskey(dc_result["solution"], "gen")
                dc_total_gen = sum(gen["pg"] for (_, gen) in dc_result["solution"]["gen"]) * baseMVA_dc
            end

            dc_output = Dict(
                "termination_status" => string(dc_result["termination_status"]),
                "objective" => dc_result["objective"],
                "solve_time" => 0.0,  # warm-start DC time is negligible
                "formulation" => "dc",
                "relaxation_level" => dc_solved_level,
                "relaxation_label" => level_label(dc_solved_level),
                "relaxation_name" => RELAXATION_LEVELS[dc_solved_level + 1]["name"],
                "n_shunts" => 0,
                "n_decommitted" => @isdefined(n_decommitted) ? n_decommitted : 0,
                "total_load_mw" => round(dc_total_load, digits=1),
                "total_gen_mw" => round(dc_total_gen, digits=1),
                "n_buses" => length(get(data, "bus", Dict())),
                "n_branches" => length(get(data, "branch", Dict())),
                "n_gens" => length(get(data, "gen", Dict())),
                "n_loads" => length(get(data, "load", Dict())),
                "solution" => get(dc_result, "solution", nothing),
            )

            open(options["dc_output_file"], "w") do f
                JSON.print(f, dc_output, 2)
            end
            println("  DC results saved: $(options["dc_output_file"])")
        end

        # ── Save interface data for subprocess progressive mode ──
        if interface_data !== nothing && options["warm_start_file"] !== nothing
            iface_file = replace(options["warm_start_file"], ".json" => "_interfaces.json")
            open(iface_file, "w") do f
                JSON.print(f, interface_data, 2)
            end
            println("  Interface data saved: $iface_file")
        end

        # ── Early exit for --warmstart-only (subprocess progressive mode) ──
        if options["warmstart_only"]
            println("\n✅ Warm-start prep complete (--warmstart-only). Exiting.")
            return Dict("termination_status" => "WARM_START_ONLY", "objective" => 0.0), -1
        end
    end

    # ── Progressive relaxation loop ──
    local result, trial_data, all_changes
    solved_level = -1
    total_solve_time = 0.0

    if options["progressive"]
        println("\nRunning $formulation_str with progressive relaxation...")

        # For AC-OPF, try AC1 (voltage/Q) right after L0 — AC failures are
        # almost always reactive-power/voltage issues, not thermal/angle.
        # Documented escalation: L0 → AC1 → L1 → L2 → L3 → L4 → L5.
        # AC1 is standalone; L1-L5 include AC1 as a base layer (monotonic).
        #
        # NOTE: for start_level > 0 on AC we used to do
        #   level_sequence = collect(start_level:max_level)
        # which (since max_level includes AC1=6) put AC1 at the END of the
        # escalation for runs started mid-ladder — e.g. start_level=2 gave
        # L2→L3→L4→L5→AC1, which violates both the documented order and
        # the monotonicity claim (L1-L5 already include AC1 as a base layer,
        # so AC1-as-tail is meaningless). Build the sequence explicitly.
        if is_ac
            if start_level == 0
                ac_order = [0, 6, 1, 2, 3, 4, 5]
                level_sequence = [l for l in ac_order if l <= max_level]
            elseif start_level == 6
                level_sequence = [6]
            else
                # Start at L{start_level} and escalate L{start_level+1}..L5.
                # AC1 (=6) is not re-attempted here because L1..L5 already
                # layer it in via solve_with_level.
                level_sequence = collect(start_level:min(max_level, 5))
            end
        else
            level_sequence = collect(start_level:max_level)
        end

        for level in level_sequence
            level_info = RELAXATION_LEVELS[level + 1]

            if level == 0
                println("\n> L0: Strict (no relaxations)")
            else
                lbl = level_label(level)
                println("\n> $(lbl): $(level_info["name"]) — $(level_info["description"])")
            end

            t_start = time()
            result, trial_data, all_changes = solve_with_level(data, options, level; interface_data=interface_data)
            elapsed = time() - t_start
            total_solve_time += elapsed

            status = result["termination_status"]
            println("  → $(status) ($(round(elapsed, digits=1))s)")
            flush(stdout)

            if is_solved(result) || is_almost_solved(result)
                solved_level = level
                # Save the mutated trial_data (decommit + impedance fix +
                # DC-shunt injection + relaxation level mutations applied)
                # to a new JSON so strict solver can load-and-solve it
                # from cold start. Used to generate "_relaxed.json" variants.
                if options["save_relaxed_file"] !== nothing
                    relaxed_out = deepcopy(trial_data)
                    relaxed_out["_relaxation"] = Dict(
                        "solved_level" => level,
                        "level_label"  => level_label(level),
                        "changes_applied" => all_changes,
                        "source_input" => options["model_file"],
                    )
                    open(options["save_relaxed_file"], "w") do f
                        JSON.print(f, relaxed_out, 2)
                    end
                    println("  Saved relaxed model to $(options["save_relaxed_file"])")
                end
                break
            end

            # Propagate solution as warm-start for the next relaxation level
            if haskey(result, "solution")
                sol = result["solution"]
                if haskey(sol, "bus")
                    for (bid, bsol) in sol["bus"]
                        if haskey(data["bus"], bid)
                            haskey(bsol, "va") && (data["bus"][bid]["va"] = bsol["va"])
                            haskey(bsol, "vm") && (data["bus"][bid]["vm"] = bsol["vm"])
                        end
                    end
                end
                if haskey(sol, "gen")
                    for (gid, gsol) in sol["gen"]
                        if haskey(data["gen"], gid)
                            haskey(gsol, "pg") && (data["gen"][gid]["pg"] = gsol["pg"])
                            haskey(gsol, "qg") && (data["gen"][gid]["qg"] = gsol["qg"])
                        end
                    end
                end
            end
        end
    else
        # Single attempt at start_level
        level_info = RELAXATION_LEVELS[start_level + 1]
        lbl = level_label(start_level)
        println("\nRunning $formulation_str at $(lbl): $(level_info["name"]) (no progressive)")

        t_start = time()
        result, trial_data, all_changes = solve_with_level(data, options, start_level; interface_data=interface_data)
        total_solve_time = time() - t_start

        if is_acceptable(result)
            solved_level = start_level
            # Save the mutated trial_data to a JSON (same hook as the
            # progressive path). Used by solve_topo_json.jl's make_solvable()
            # for cold-strict verification of each level's output.
            if options["save_relaxed_file"] !== nothing
                relaxed_out = deepcopy(trial_data)
                relaxed_out["_relaxation"] = Dict(
                    "solved_level" => start_level,
                    "level_label"  => level_label(start_level),
                    "changes_applied" => all_changes,
                    "source_input" => options["model_file"],
                )
                open(options["save_relaxed_file"], "w") do f
                    JSON.print(f, relaxed_out, 2)
                end
                println("  Saved relaxed model to $(options["save_relaxed_file"])")
            end
        end
    end

    # ── Report results ──
    print_results(result, trial_data, total_solve_time, solved_level, all_changes, max_level)

    # ── Save results ──
    if options["output_file"] !== nothing
        formulation = options["soc"] ? "soc" : options["formulation"]

        # Count DC-derived shunts (those added by inject_dc_derived_shunts!)
        n_shunts = @isdefined(n_dc_shunts) ? n_dc_shunts : 0

        # Compute model totals from the data that was actually solved
        baseMVA = get(data, "baseMVA", 100.0)
        total_load_mw = haskey(data, "load") ?
            sum(get(load, "pd", 0.0) for (_, load) in data["load"]) * baseMVA : 0.0
        total_gen_mw = 0.0
        if haskey(result, "solution") && haskey(result["solution"], "gen")
            total_gen_mw = sum(gen["pg"] for (_, gen) in result["solution"]["gen"]) * baseMVA
        end

        # Count decommitted generators
        n_decommit = @isdefined(n_decommitted) ? n_decommitted : 0

        output = Dict(
            "termination_status" => string(result["termination_status"]),
            "objective" => result["objective"],
            "solve_time" => total_solve_time,
            "formulation" => formulation,
            "relaxation_level" => solved_level,
            "relaxation_label" => solved_level >= 0 ? level_label(solved_level) : "none",
            "relaxation_name" => solved_level >= 0 ? RELAXATION_LEVELS[solved_level + 1]["name"] : "none",
            "n_shunts" => n_shunts,
            "n_decommitted" => n_decommit,
            "total_load_mw" => round(total_load_mw, digits=1),
            "total_gen_mw" => round(total_gen_mw, digits=1),
            "n_buses" => length(get(data, "bus", Dict())),
            "n_branches" => length(get(data, "branch", Dict())),
            "n_gens" => length(get(data, "gen", Dict())),
            "n_loads" => length(get(data, "load", Dict())),
            "n_interfaces" => interface_data !== nothing ? length(interface_data) : 0,
            "solution" => get(result, "solution", nothing),
        )

        open(options["output_file"], "w") do f
            JSON.print(f, output, 2)
        end
        println("\n✅ Results saved to: $(options["output_file"])")
    end

    return result, solved_level
end

function main()
    if length(ARGS) < 1
        print_usage()
        exit(1)
    end

    options = parse_args(ARGS)

    if options["model_file"] === nothing
        println("ERROR: No model file specified")
        print_usage()
        exit(1)
    end

    if !isfile(options["model_file"])
        println("ERROR: File not found: $(options["model_file"])")
        exit(1)
    end

    result, solved_level = run_opf(options["model_file"], options)

    # Exit with appropriate code
    if options["warmstart_only"]
        exit(0)  # prep succeeded
    end
    exit(solved_level >= 0 ? 0 : 1)
end

# Run main only when invoked as a script (not when include()'d from batch driver)
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
