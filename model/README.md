# GridSFM Model — AC-OPF Inference

Minimal package for running the released **GridSFM-Open** AC-OPF foundation model.

> **Supported grid size: ≥500 buses.** GridSFM-Open is trained on grids with at least 500 buses. Smaller cases (`case14_ieee`, `case30_ieee`, `case57_ieee`, `case118_ieee`) are out of distribution and will produce meaningless results.

## Layout

```
model/
├── gridsfm/        # inference package
├── samples/        # 53 base scenarios (.pyg.json); see samples/README.md
├── examples/       # infer_samples, opfdata
├── tests/          # pytest suite
├── checkpoints/    # local cache for downloaded ckpts (gitignored)
├── pyproject.toml
└── README.md
```

License: top-level `LICENSE` covers this package.

## Install

Tested on Ubuntu 22.04 / 24.04 and macOS 14+. Python 3.10+, primary CI on 3.12.

```bash
cd model
python -m venv .venv
source .venv/bin/activate
pip install -e .            # package + runtime deps
pip install -e ".[test]"    # also pytest
```

Dependency pins live in `pyproject.toml` (`torch >=2.6,<2.9`, `torch_geometric >=2.5,<3`, `numpy <3`, `scipy <2`).

## Run the tests

```bash
cd model
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python -m pytest -v
```

Always invoke as `python -m pytest`, not bare `pytest` — a user-local `~/.local/bin/pytest` on `$PATH` will shadow the venv's. Tests that need the checkpoint `pytest.skip` if it's absent.

## Get the checkpoint

```python
from gridsfm import load_from_hf
model = load_from_hf("microsoft/GridSFM_Open")
```

Or download once and load locally:

```bash
hf download microsoft/GridSFM_Open gridsfm_open_v1.0.pt --local-dir checkpoints
```

```python
import torch
from gridsfm import load_model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model("checkpoints/gridsfm_open_v1.0.pt", device=device)
```

## Quickstart

Run the model on all 53 shipped samples in one batched forward and print per-case predictions:

```bash
cd model
python examples/infer_samples.py              # CPU, default ckpt
python examples/infer_samples.py --gpu 0      # GPU 0
python examples/infer_samples.py path/to/your_ckpt.pt
```

Run on the [OPFData](https://arxiv.org/abs/2406.07234) dataset via `torch_geometric.datasets.OPFDataset` (auto-downloads on first use, then cached):

```bash
python examples/opfdata.py --case pglib_opf_case500_goc --root ~/.cache/opfdata --gpu 0
```

Iterates the full test split in chunks of `--batch-size 128` and reports per-bus / per-generator MAE + cost MAPE vs the OPF-solved ground truth.

### In your own code

Single-graph:

```python
import torch
from gridsfm import load_model, predict

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model("checkpoints/gridsfm_open_v1.0.pt", device=device)
out = predict(model, "samples/case500_goc.pyg.json")
print(out["V"], out["theta"], out["Pg"], out["Qg"], out["feas"])
```

Batched (mixed topology):

```python
import torch
from pathlib import Path
from torch_geometric.utils import unbatch
from gridsfm import load_model, batch_data_list, load_pyg_json, prepare_for_inference

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model("checkpoints/gridsfm_open_v1.0.pt", device=device)

scenario_paths = sorted(Path("samples").glob("*.pyg.json"))
prepared = [prepare_for_inference(load_pyg_json(p)) for p in scenario_paths]
batch = batch_data_list(prepared).to(device)
with torch.no_grad():
    out = model(batch)

# Split per-graph predictions using torch_geometric.utils.unbatch:
V_per_graph  = unbatch(out["bus"].pred[:, 1],       out["bus"].batch)
Pg_per_graph = unbatch(out["generator"].pred[:, 0], out["generator"].batch)
for p, V_i, Pg_i in zip(scenario_paths, V_per_graph, Pg_per_graph):
    print(f"{p.name}: V=[{V_i.min():.3f},{V_i.max():.3f}]  Pg_sum={Pg_i.sum():.2f}")
```

## Output schema

`predict()` is single-graph only; returns a dict of CPU tensors / floats:

| key | shape | units |
|---|---|---|
| `V`          | `[n_bus]`     | voltage magnitude (per-unit) |
| `theta`      | `[n_bus]`     | voltage angle (radians) |
| `Pg`         | `[n_gen]`     | generator active power (per-unit) |
| `Qg`         | `[n_gen]`     | generator reactive power (per-unit) |
| `Pij,Qij,Pji,Qji` | `[n_flow_edges]` | branch flows concatenated across families (per-unit) |
| `flow_edge_types`  | `list[str]` | which families contributed flows, in order |
| `flow_edge_counts` | `list[int]` | row count per family |
| `feas`       | `float`       | scenario feasibility probability `[0,1]` |
| `feas_logit` | `float`       | raw feas-head logit |

Split the flat flows back per-family with `flow_edge_counts`:

```python
out = predict(model, "samples/case500_goc.pyg.json")
n_ac = out["flow_edge_counts"][0] if "ac_line" in out["flow_edge_types"] else 0
Pij_ac = out["Pij"][:n_ac]       # aligns with data["ac_line"].edge_index
Pij_tr = out["Pij"][n_ac:]       # aligns with data["transformer"].edge_index
```

For batched inference (`model(batch)` directly), `feas_logit` becomes `[n_graphs]` and `out["bus"].pred` / `out["generator"].pred` are concatenated across graphs.

## Inputs

Two formats:

- **`.pyg.json`** (native) — `{grid: {nodes: {bus, generator, load, shunt}, edges: {ac_line, transformer, generator_link, load_link, shunt_link}}}`. See `samples/case500_goc.pyg.json`.
- **OPFData JSON** (DeepMind release) — same layout; pass `fmt="opfdata"`.

Both loaders require a non-empty `bus` block and matching `senders`/`receivers` lengths per edge type. `load_opfdata` additionally requires EXACT column widths (4 / 11 / 9 / 11 for bus / generator / ac_line / transformer).

Column ordering is defined in `gridsfm/schema.py`:

| node/edge | columns |
|---|---|
| bus | `[base_kV, type, Vmin, Vmax]` |
| generator | `[mbase, _, Pmin, Pmax, _, Qmin, Qmax, Vg, cp2, cp1, cp0]` |
| load | `[Pd, Qd, ...]` |
| shunt | `[Bs, Gs, ...]` |
| ac_line edge_attr | `[angmin, angmax, b_fr, b_to, r, x, rate_a, ...]` |
| transformer edge_attr | `[angmin, angmax, r, x, rate_a, _, _, tap, shift, b_fr, b_to]` |

`prepare_for_inference(data)` mutates the input (attaches PE features, adds `branch_ac` / `branch_tr` / `cycle` node types, widens `bus.x` from 4 to 16 columns). `predict(model, scenario)` deep-copies a user-passed `HeteroData` before mutating.

## Caches

Three per-topology caches keyed by SHA-1 over topology + relevant `edge_attr` bytes, so impedance perturbations (line derating) and topology perturbations (line outages, N-1) both invalidate correctly.

- **`CycleBasisCache`** (`cycle_basis.py`): in-memory LRU **128**, optional disk persistence at `$XDG_CACHE_HOME/gridsfm/cycle_basis/`. Disk cache is schema-versioned.
- **`LaplacianFactorizationCache`** (`pe_features.py`): in-memory LRU **16**. SciPy SuperLU closures, not picklable.
- **`DCPriorCache`** (`dc_prior.py`): in-memory LRU **64**, same picklability constraint. Keyed on topology + `b = 1/x` + slack index.

Defaults are module-level singletons (`DEFAULT_CYCLE_CACHE`, `DEFAULT_LAPLACIAN_CACHE`, plus `model._dc_cache`). To customize sizes or disk paths for large N-1 / N-k sweeps, replace them before first use:

```python
import gridsfm.cycle_basis as cb, gridsfm.pe_features as pe
cb.DEFAULT_CYCLE_CACHE = cb.CycleBasisCache(max_cache=2048, cache_dir="/data/grid_cache")
pe.DEFAULT_LAPLACIAN_CACHE = pe.LaplacianFactorizationCache(max_cache=128)

from gridsfm import load_model
from gridsfm.dc_prior import DCPriorCache
model = load_model("checkpoints/gridsfm_open_v1.0.pt")
model._dc_cache = DCPriorCache(max_cache=512)
```

## Citation

A reference will be added once the accompanying paper is public. Until then, please cite `microsoft/GridSFM`.
