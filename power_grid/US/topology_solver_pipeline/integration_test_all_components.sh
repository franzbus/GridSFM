#!/bin/bash
# integration_test_all_components.sh — end-to-end verification of the
# topo_solver_pipe/ pipeline.
#
# Orchestrates the four Julia CLI binaries in this directory plus an
# optional Python sanity-import check; shell is the right tool for that
# (no Julia Test-stdlib subprocess wrapper needed).
#
# Steps (using a raw topology JSON that is NOT strict-solvable as-is):
#   1. solve:    input.json             →  input.solvable.json      (solve_topo_json.jl)
#                Iterate relaxation levels L0, AC1, L1..L5 until a
#                cold-strict solvable JSON is produced.
#   2. export:   input.solvable.json    →  gridsfm.pyg.json          (export_gridsfm_data.jl)
#                Solve the solvable JSON and emit a gridSFM pyg.json.
#   3. resolve:  input.solvable.json    →  (strict AC-OPF again)
#                Verify the solvable JSON still solves cold-strict and
#                reports the same objective as the exported one.
#   4. perturb:  input.solvable.json    →  {base, loads, costs, killgen, derate,
#                                           vsqueeze} × 1 pyg.json   (gen_perturbed_data.jl)
#                Generate one scenario per mode (5 modes + unperturbed),
#                then for each: reconstruct PowerModels data from the
#                pyg.json via solve_pyg_json.jl, re-solve, and confirm
#                the objective matches metadata.objective within 0.1%.
#
# Pass criteria:
#   - Step 1 succeeds (exits 0)
#   - Step 2 succeeds and objectives match step-1 objective
#   - Step 3 objective matches step-2 objective
#   - Python's build_hetero_data_from_json loads the pyg.json cleanly
#   - Step 4 successfully re-solves each scenario with Δ < 0.1% vs the
#     objective recorded in its pyg.json metadata
#
# Usage:
#   bash integration_test_all_components.sh <input.json> [output_dir]
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JULIA_PROJECT="$SCRIPT_DIR"  # self-contained — Project.toml is in this dir

if [ $# -lt 1 ]; then
    echo "Usage: bash integration_test_all_components.sh <input.json> [output_dir]"
    exit 2
fi

INPUT="$1"
OUT_DIR="${2:-/tmp/topo_solver_pipe_test}"
mkdir -p "$OUT_DIR"

[ ! -f "$INPUT" ] && { echo "Input not found: $INPUT"; exit 1; }

BASE=$(basename "$INPUT" .json)
SOLVABLE="$OUT_DIR/${BASE}.solvable.json"
PYG="$OUT_DIR/${BASE}.pyg.json"
RESOLVE_LOG="$OUT_DIR/${BASE}_resolve.log"

echo "=========================================================="
echo "  Round-trip test for: $INPUT"
echo "  Output dir:          $OUT_DIR"
echo "=========================================================="

# ─────────────────────────────────────────────────────────────
# Step 1 — solve to cold-strict solvable
# ─────────────────────────────────────────────────────────────
echo ""
echo "[STEP 1/4] SOLVE  solve_topo_json.jl <input> <solvable>"
SOLVE_OUT=$(julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/solve_topo_json.jl" \
    "$INPUT" "$SOLVABLE" 2>&1)
SOLVE_RC=$?
echo "$SOLVE_OUT" | grep -E "Trying|cold-strict|RESULT|✓|✗" | sed 's/^/  /'
if [ $SOLVE_RC -ne 0 ]; then
    echo "  ❌ Step 1 failed — could not produce a cold-strict solvable JSON"
    exit 1
fi
SOLVE_OBJ=$(echo "$SOLVE_OUT" | grep -oP 'RESULT \S+ \S+ obj=\K-?[0-9.]+(?:[eE][+-]?[0-9]+)?' | head -1)
SOLVE_LEVEL=$(echo "$SOLVE_OUT" | grep -oP 'RESULT \S+ \K\S+(?= obj=)' | head -1)
echo "  ✓ solvable at $SOLVE_LEVEL, obj=$SOLVE_OBJ"

# ─────────────────────────────────────────────────────────────
# Step 2 — export to gridSFM pyg.json
# ─────────────────────────────────────────────────────────────
echo ""
echo "[STEP 2/4] EXPORT  export_gridsfm_data.jl <solvable> <pyg>"
EXPORT_OUT=$(julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/export_gridsfm_data.jl" \
    "$SOLVABLE" "$PYG" 2>&1)
EXPORT_RC=$?
echo "$EXPORT_OUT" | grep -E "Solve|pyg.json|feas" | sed 's/^/  /'
if [ $EXPORT_RC -ne 0 ]; then
    echo "  ❌ Step 2 failed — export script exited non-zero"
    exit 1
fi
EXPORT_OBJ=$(echo "$EXPORT_OUT" | grep -oP 'obj=\K-?[0-9.]+(?:[eE][+-]?[0-9]+)?' | head -1)
echo "  ✓ exported, obj=$EXPORT_OBJ"

# ─────────────────────────────────────────────────────────────
# Step 3 — re-solve the solvable JSON to verify round-trip consistency
# ─────────────────────────────────────────────────────────────
echo ""
echo "[STEP 3/4] RESOLVE  re-solve solvable JSON, compare objective"
RESOLVE_OUT=$(julia --project="$JULIA_PROJECT" -e "
using PowerModels, Ipopt
PowerModels.silence()
# import_all=false: don't carry stage-2 top-level dicts (e.g. _relaxation,
# whose keys aren't integer-parseable) into the PowerModels data struct —
# InfrastructureModels assumes integer-keyed component dicts at the top level.
net = PowerModels.parse_file(\"$SOLVABLE\"; import_all=false, validate=true)
# Cold: zero warm-start fields
for (_, b) in get(net, \"bus\", Dict()); b[\"vm\"] = 1.0; b[\"va\"] = 0.0; end
for (_, g) in get(net, \"gen\", Dict()); g[\"pg\"] = 0.0; g[\"qg\"] = 0.0; end
solver = optimizer_with_attributes(Ipopt.Optimizer,
    \"print_level\"=>0, \"max_iter\"=>10000, \"tol\"=>1e-6, \"acceptable_tol\"=>1e-4)
res = PowerModels.solve_ac_opf(net, solver)
term = string(get(res, \"termination_status\", \"UNKNOWN\"))
obj  = try Float64(get(res, \"objective\", NaN)) catch; NaN end
println(\"RESOLVE status=\$(term) obj=\$(obj)\")
" 2>&1 | tee "$RESOLVE_LOG")
RESOLVE_RC=$?
echo "$RESOLVE_OUT" | grep -E "RESOLVE" | sed 's/^/  /'
if [ $RESOLVE_RC -ne 0 ]; then
    echo "  ❌ Step 3 failed — resolve script crashed"
    exit 1
fi
RESOLVE_OBJ=$(echo "$RESOLVE_OUT" | grep -oP 'RESOLVE status=\S+ obj=\K-?[0-9.]+(?:[eE][+-]?[0-9]+)?' | head -1)
RESOLVE_STATUS=$(echo "$RESOLVE_OUT" | grep -oP 'RESOLVE status=\K\S+(?= obj=)' | head -1)
if [ "$RESOLVE_STATUS" != "LOCALLY_SOLVED" ] && [ "$RESOLVE_STATUS" != "OPTIMAL" ]; then
    echo "  ❌ Step 3 resolve did not converge: $RESOLVE_STATUS"
    exit 1
fi
echo "  ✓ resolve, obj=$RESOLVE_OBJ"

# ─────────────────────────────────────────────────────────────
# Consistency check — objectives should match across all 3 steps
# ─────────────────────────────────────────────────────────────
echo ""
echo "[CHECK] objective consistency across steps"
python3 -c "
solve=$SOLVE_OBJ
export=$EXPORT_OBJ
resolve=$RESOLVE_OBJ
objs=[solve, export, resolve]
tag = ['solve','export','resolve']
for t,v in zip(tag,objs):
    print(f'  {t:8s} obj = {v:.2f}')
delta = max(objs) - min(objs)
rel = 100 * delta / max(abs(max(objs)), 1.0)
print(f'  max-min delta = {delta:.4f} ({rel:.4f}%)')
if rel > 0.01:
    print('  ⚠️  Objectives differ by more than 0.01% — possible round-trip drift')
    exit(1)
else:
    print('  ✓ Objectives consistent within 0.01%')
" || exit 1

# ─────────────────────────────────────────────────────────────
# Python-side load check — build_hetero_data_from_json happy?
# ─────────────────────────────────────────────────────────────
echo ""
echo "[CHECK] Python load sanity"
# Python must have `gridfm` importable. Override with GRIDFM_PYTHON if your
# interpreter lives somewhere other than `python3` on PATH.
PY_BIN="${GRIDFM_PYTHON:-python3}"
"$PY_BIN" -c "
import json, sys
try:
    from gridfm.data.opf_data_utils import build_hetero_data_from_json
except Exception as e:
    print(f'  (skipping Python check — build_hetero_data_from_json not importable: {e})')
    sys.exit(0)
d = json.load(open('$PYG'))
data = build_hetero_data_from_json(d)
print(f'  ✓ HeteroData built: {data.num_nodes} total nodes, {data.num_edges} total edges')
print(f'  ✓ bus.x={list(data[\"bus\"].x.shape)} gen.x={list(data[\"generator\"].x.shape)}')
print(f'  ✓ objective={float(data.objective):.2f} feasible={int(data.feasible)}')
" || exit 1

# ─────────────────────────────────────────────────────────────
# Step 4 — perturbed-scenario round-trip
# For each of the 5 perturbation modes + the unperturbed base, generate
# one scenario via gen_perturbed_data.jl, then reconstruct PowerModels
# data from its pyg.json and re-solve strict AC-OPF. The resulting
# objective must match metadata.objective within 0.1% — this is what
# makes a pyg.json "usable" as a gridSFM input.
# ─────────────────────────────────────────────────────────────
echo ""
echo "[STEP 4/4] PERTURB+RESOLVE  gen_perturbed_data.jl → solve_pyg_json.jl"
SCEN_OUT="$OUT_DIR/scenarios"
rm -rf "$SCEN_OUT" && mkdir -p "$SCEN_OUT"

# Feed gen_perturbed_data.jl a one-line grid list (one scenario per mode is enough).
GRID_LIST="$OUT_DIR/grids_solvable.txt"
echo "$SOLVABLE 1" > "$GRID_LIST"

GEN_OUT=$(julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/gen_perturbed_data.jl" \
    "$GRID_LIST" 6 "$SCEN_OUT" 2>&1)
GEN_RC=$?
echo "$GEN_OUT" | grep -E "feasible|Total|Summary|Done|FAILED" | sed 's/^/  /'
if [ $GEN_RC -ne 0 ]; then
    echo "  ❌ Step 4a failed — gen_perturbed_data.jl exited non-zero"
    exit 1
fi

# gen_perturbed_data writes under <out_root>/<case>/ where <case> is the
# input filename minus .solvable.json.
SCEN_DIR="$SCEN_OUT/${BASE}"
[ ! -d "$SCEN_DIR" ] && { echo "  ❌ Step 4a: scenario dir not found: $SCEN_DIR"; exit 1; }

N_OK=0
N_FAIL=0
for f in base_unperturbed loads_0001 costs_0001 killgen_0001 derate_0001 vsqueeze_0001; do
    PYG_PATH="$SCEN_DIR/$f.pyg.json"
    if [ ! -f "$PYG_PATH" ]; then
        echo "  (skip $f — not written, likely infeasible at generation time)"
        continue
    fi
    OUT=$(julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/solve_pyg_json.jl" \
          "$SOLVABLE" "$PYG_PATH" 2>&1 | grep -E "RESOLVE")
    RC=$?
    echo "  $OUT"
    if [ $RC -eq 0 ]; then N_OK=$((N_OK+1)); else N_FAIL=$((N_FAIL+1)); fi
done
if [ $N_FAIL -gt 0 ]; then
    echo "  ❌ Step 4 failed — $N_FAIL scenario(s) did not round-trip cleanly"
    exit 1
fi
if [ $N_OK -lt 2 ]; then
    echo "  ❌ Step 4 failed — only $N_OK scenario(s) passed (expected at least 2)"
    exit 1
fi
echo "  ✓ $N_OK scenario(s) round-tripped with Δ < 0.1%"

echo ""
echo "=========================================================="
echo "  ✓ ALL ROUND-TRIP CHECKS PASSED"
echo "  Artifacts:"
echo "    solvable:  $SOLVABLE"
echo "    pyg:       $PYG"
echo "    scenarios: $SCEN_DIR/"
echo "    resolve:   $RESOLVE_LOG"
echo "=========================================================="
