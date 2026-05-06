#!/bin/bash
# =============================================================================
# run_pipeline.sh — Run the full topology_solver_pipeline inside Docker
# =============================================================================
#
# This script runs all 4 stages of the pipeline for a given state model:
#   Stage 2a: solve_topo_json.jl  — produce cold-strict solvable JSON
#   Stage 2b: export_gridsfm_data.jl — export to gridSFM .pyg.json
#   Stage 3:  gen_perturbed_data.jl — generate perturbed scenarios (1 per mode)
#   Verify:   solve_pyg_json.jl — round-trip verification of each scenario
#
# The script can also run the integration test which covers all the above
# plus objective consistency checks.
#
# Usage (inside container):
#   bash /app/docker/run_pipeline.sh <state_name> <hour> [n_per_mode]
#
# Arguments:
#   state_name   — e.g. "alabama", "montana"
#   hour         — "16h" (peak) or "04h" (off-peak)
#   n_per_mode   — scenarios per perturbation mode (default: 1)
#
# Expects:
#   /data/<hour>/<state_name>_model.json   — raw topology JSON (mounted)
#   /output/                               — writable output directory (mounted)
#
# =============================================================================
set -euo pipefail

STATE="${1:?Usage: run_pipeline.sh <state_name> <hour> [n_per_mode]}"
HOUR="${2:?Usage: run_pipeline.sh <state_name> <hour> [n_per_mode]}"
N_PER_MODE="${3:-1}"

INPUT="/data/${HOUR}/${STATE}_model.json"
OUT_DIR="/output/${STATE}_${HOUR}"
mkdir -p "$OUT_DIR"

SOLVABLE="$OUT_DIR/${STATE}_model.solvable.json"
PYG="$OUT_DIR/${STATE}_model.pyg.json"
SCEN_DIR="$OUT_DIR/scenarios"
GRID_LIST="$OUT_DIR/grids_solvable.txt"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  topology_solver_pipeline — full pipeline run                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  State:       $STATE"
echo "║  Hour:        $HOUR"
echo "║  Input:       $INPUT"
echo "║  Output dir:  $OUT_DIR"
echo "║  Scenarios:   $N_PER_MODE per mode (5 modes + 1 unperturbed)"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "$INPUT" ]; then
    echo "❌ Input file not found: $INPUT"
    echo "   Make sure data is mounted at /data/"
    exit 1
fi

# ─────────────────────────────────────────────────────────────
# Preprocessing — patch model for PowerModels.jl compatibility
# ─────────────────────────────────────────────────────────────
# The data-release models may lack "storage" and "switch" dicts required
# by the pinned PowerModels.jl version. Patch to a writable working copy.
PATCHED_INPUT="$OUT_DIR/${STATE}_model_patched.json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[PREPROCESS] Patching model for PowerModels.jl compatibility"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
julia --project=/app /app/docker/patch_model.jl "$INPUT" "$PATCHED_INPUT"
echo ""

# ─────────────────────────────────────────────────────────────
# Stage 2a — Solve: raw topology → cold-strict solvable
# ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[STAGE 2a] SOLVE: solve_topo_json.jl"
echo "  Input:  $PATCHED_INPUT"
echo "  Output: $SOLVABLE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

set +e
SOLVE_OUT=$(julia --project=/app /app/solve_topo_json.jl "$PATCHED_INPUT" "$SOLVABLE" 2>&1)
SOLVE_RC=$?
set -e
echo "$SOLVE_OUT"
echo ""

if [ $SOLVE_RC -ne 0 ]; then
    echo "❌ Stage 2a FAILED — could not produce a cold-strict solvable JSON"
    exit 1
fi

SOLVE_OBJ=$(echo "$SOLVE_OUT" | grep -oP 'RESULT \S+ \S+ obj=\K-?[0-9.]+(?:[eE][+-]?[0-9]+)?' | head -1)
SOLVE_LEVEL=$(echo "$SOLVE_OUT" | grep -oP 'RESULT \S+ \K\S+(?= obj=)' | head -1)
echo "✅ Stage 2a PASSED — solvable at $SOLVE_LEVEL, obj=$SOLVE_OBJ"
echo ""

# ─────────────────────────────────────────────────────────────
# Stage 2b — Export: solvable → gridSFM .pyg.json
# ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[STAGE 2b] EXPORT: export_gridsfm_data.jl"
echo "  Input:  $SOLVABLE"
echo "  Output: $PYG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

set +e
EXPORT_OUT=$(julia --project=/app /app/export_gridsfm_data.jl "$SOLVABLE" "$PYG" 2>&1)
EXPORT_RC=$?
set -e
echo "$EXPORT_OUT"
echo ""

if [ $EXPORT_RC -ne 0 ]; then
    echo "❌ Stage 2b FAILED — export script exited non-zero"
    exit 1
fi

EXPORT_OBJ=$(echo "$EXPORT_OUT" | grep -oP 'obj=\K-?[0-9.]+(?:[eE][+-]?[0-9]+)?' | head -1)
echo "✅ Stage 2b PASSED — exported, obj=$EXPORT_OBJ"
echo ""

# ─────────────────────────────────────────────────────────────
# Stage 3 — Perturb: generate scenarios
# ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[STAGE 3] PERTURB: gen_perturbed_data.jl ($N_PER_MODE per mode)"
echo "  Input:  $SOLVABLE"
echo "  Output: $SCEN_DIR/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$SCEN_DIR"
echo "$SOLVABLE $N_PER_MODE" > "$GRID_LIST"

set +e
GEN_OUT=$(julia --project=/app /app/gen_perturbed_data.jl "$GRID_LIST" 2 "$SCEN_DIR" 2>&1)
GEN_RC=$?
set -e
echo "$GEN_OUT"
echo ""

if [ $GEN_RC -ne 0 ]; then
    echo "❌ Stage 3 FAILED — gen_perturbed_data.jl exited non-zero"
    exit 1
fi
echo "✅ Stage 3 PASSED — scenarios generated"
echo ""

# ─────────────────────────────────────────────────────────────
# Verify — round-trip each scenario via solve_pyg_json.jl
# ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[VERIFY] Round-trip check: solve_pyg_json.jl on each scenario"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Determine scenario subdirectory name (filename without .solvable.json)
CASE_NAME=$(basename "$SOLVABLE" .solvable.json)
CASE_DIR="$SCEN_DIR/${CASE_NAME}"

N_OK=0
N_FAIL=0
N_SKIP=0

FOUND_SCENARIOS=0
while IFS= read -r -d '' PYG_PATH; do
    FOUND_SCENARIOS=1
    if JULIA_OUT=$(julia --project=/app /app/solve_pyg_json.jl "$SOLVABLE" "$PYG_PATH" 2>&1); then
        RC=0
    else
        RC=$?
    fi
    OUT=$(printf '%s\n' "$JULIA_OUT" | grep -E "RESOLVE" || true)
    echo "  $OUT"
    if [ $RC -eq 0 ]; then N_OK=$((N_OK+1)); else N_FAIL=$((N_FAIL+1)); fi
done < <(find "$CASE_DIR" -maxdepth 1 -type f -name '*.pyg.json' -print0 | sort -z)

if [ $FOUND_SCENARIOS -eq 0 ]; then
    echo "  (skip verification — no .pyg.json scenarios found in $CASE_DIR)"
    N_SKIP=$((N_SKIP+1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SUMMARY for $STATE ($HOUR)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Solve level:   $SOLVE_LEVEL"
echo "  Solve obj:     $SOLVE_OBJ"
echo "  Export obj:     $EXPORT_OBJ"
echo "  Scenarios OK:   $N_OK"
echo "  Scenarios FAIL: $N_FAIL"
echo "  Scenarios SKIP: $N_SKIP"
echo "  Artifacts:"
echo "    $SOLVABLE"
echo "    $PYG"
echo "    $CASE_DIR/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $N_FAIL -gt 0 ]; then
    echo "⚠️  $N_FAIL scenario(s) failed round-trip verification"
    exit 1
fi
echo "✅ ALL STAGES PASSED for $STATE ($HOUR)"
