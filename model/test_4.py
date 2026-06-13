#%%
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline: pandapower case30 → PowerModels JSON → GridSFM .pyg.json → model
# ─────────────────────────────────────────────────────────────────────────────
import json
import os
import shutil
import subprocess
import sys

from pandapower.networks.power_system_test_cases import case30, case57
from pandapower.converter.pandamodels.to_pm import convert_pp_to_pm

# Keys that PowerModels.jl accepts at the top level; everything else is pandapower-specific
_PM_STANDARD_KEYS = {
    "bus", "gen", "branch", "load", "shunt", "storage", "switch", "dcline",
    "baseMVA", "per_unit", "name", "source_version", "sourcetype",
}

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
JULIA_SCRIPT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../power_grid/US/topology_solver_pipeline/export_gridsfm_data.jl")
)
JULIA_PROJECT = os.path.dirname(JULIA_SCRIPT)
PM_JSON  = os.path.join(SCRIPT_DIR, "case57_pm.json")
PYG_JSON = os.path.join(SCRIPT_DIR, "case57_gridsfm.pyg.json")


# ── Step 1: load case57 ──────────────────────────────────────────────────────
net = case57()
print(net)


# ── Step 2: convert to PowerModels JSON ─────────────────────────────────────
print("\n[1/3] Converting case57\ → PowerModels JSON ...")
pm = convert_pp_to_pm(net)
pm_clean = {k: v for k, v in pm.items() if k in _PM_STANDARD_KEYS}
# inject mbase into each generator (pandapower omits it; Julia requires it)
base_mva = pm_clean.get("baseMVA", 100)
for g in pm_clean.get("gen", {}).values():
    g.setdefault("mbase", base_mva)
with open(PM_JSON, "w") as f:
    json.dump(pm_clean, f)
print(f"      Written: {PM_JSON}")


# ── Step 3: run Julia export script ─────────────────────────────────────────
def _find_julia():
    for candidate in [
        "julia",
        os.path.expanduser("~/.juliaup/bin/julia"),
        "/usr/local/bin/julia",
        "/opt/homebrew/bin/julia",
    ]:
        if shutil.which(candidate):
            return candidate
    return None

julia = _find_julia()
if julia is None:
    print("\nJulia not found. Install it with:")
    print("  curl -fsSL https://install.julialang.org | sh")
    print("Then re-run this script.")
    sys.exit(1)

print(f"\n[2/3] Running Julia export script ({julia}) ...")
proc = subprocess.run(
    [julia, f"--project={JULIA_PROJECT}", JULIA_SCRIPT, PM_JSON, PYG_JSON],
    text=True,
)
if proc.returncode != 0:
    print("Julia export failed — check output above.")
    sys.exit(1)
print(f"      Written: {PYG_JSON}")

#%%
# ── Step 4: run GridSFM model ────────────────────────────────────────────────
print("\n[3/3] Running GridSFM model ...")
import torch
from gridsfm import load_model, predict

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model  = load_model("checkpoints/gridsfm_open_v1.1.pt", device=device)
out    = predict(model, "case300_gridsfm.pyg.json")

print("\n── GridSFM results ──────────────────────────────────────────────────────")
print("V    :", out["V"])
print("theta:", out["theta"])
print("Pg   :", out["Pg"])
print("Qg   :", out["Qg"])
print("feas :", out["feas"])

# %%
