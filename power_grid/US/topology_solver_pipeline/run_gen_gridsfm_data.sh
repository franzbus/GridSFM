#!/bin/bash
# run_gen_gridsfm_data.sh — driver for gen_perturbed_data.jl.
#
# Usage:
#   bash run_gen_gridsfm_data.sh [n_proc] [out_root] [grid_list] [cpu_range]
#
# Defaults:
#   n_proc     51                                          (worker processes)
#   out_root   ./out                                       (per-grid folders land here)
#   grid_list  $SCRIPT_DIR/grids_solvable.txt              (one "<solvable.json> <n_per_mode>" per line)
#   cpu_range  77-127                                      (taskset mask; 51 cores)
#
# Any argument you pass positionally overrides its default; leave later args
# unset to keep their defaults.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JULIA_PROJECT="$SCRIPT_DIR"   # self-contained — Project.toml is in this dir

N_PROC="${1:-51}"
OUT_ROOT="${2:-./out}"
GRID_LIST="${3:-$SCRIPT_DIR/grids_solvable.txt}"
CPU_RANGE="${4:-77-127}"

if [ ! -f "$GRID_LIST" ]; then
    echo "Grid list not found: $GRID_LIST" >&2
    echo "Expected format (one line per grid):  <solvable.json_path>  <n_per_mode>" >&2
    exit 1
fi

# Pin BLAS / MUMPS / OpenMP to 1 thread per worker. Without this, each of
# the N_PROC pmap workers defaults to multi-threaded MUMPS = N_PROC × #cores
# threads competing for the same cores, which causes 5–7× wall-clock
# variance and 2–3× slowdown on contended runs (validated on the
# warm-start benchmark). With OMP=1, each worker uses exactly one thread,
# matching the 1-worker-per-core layout taskset's CPU_RANGE provides.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

taskset -c "$CPU_RANGE" julia --project="$JULIA_PROJECT" \
    "$SCRIPT_DIR/gen_perturbed_data.jl" \
    "$GRID_LIST" "$N_PROC" "$OUT_ROOT"
