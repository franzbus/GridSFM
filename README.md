# GridSFM — with Congestion Mitigation Dashboard

> **This fork adds an interactive web dashboard** for grid congestion analysis on top of the original [microsoft/GridSFM](https://github.com/microsoft/GridSFM) repo.  
> Jump to [Running the Dashboard](#running-the-dashboard) to get started quickly.

---

## Running the Dashboard

The dashboard runs the full congestion mitigation pipeline — baseline prediction, battery shutdown scenario, and analysis report — interactively in the browser.

### 1. Clone and set up the environment

```bash
git clone https://github.com/franzbus/GridSFM.git
cd GridSFM/model

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # installs the gridsfm package
pip install -r requirements.txt    # installs Flask and all other deps
```

### 2. Download the model checkpoint

The checkpoint is hosted on Hugging Face. Download it once:

```python
from gridsfm import load_from_hf
load_from_hf("microsoft/GridSFM_Open")   # saves to model/checkpoints/
```

Or manually download `gridsfm_open_v1.1.pt` from  
https://huggingface.co/microsoft/GridSFM_Open and place it in `model/checkpoints/`.

### 3. Run the dashboard

```bash
cd model
python dashboard.py
```

Then open **http://localhost:5050** in your browser.

### 4. How to use it

1. **Select a sample** from the left panel (53 grids available — US state grids and standard case studies).
2. Click **Run Analysis**. A popup opens and results appear sequentially:
   - **Step 1 — Baseline prediction**: full network thermal loading map + zoom on the worst congested line.
   - **Step 2 — Battery shutdown**: charging batteries near the congested line are curtailed, the model re-predicts, and updated maps are shown.
   - **Step 3 — Mitigation report**: key metrics (loading before/after, cost comparison, savings) and a Case A/B/C verdict.
3. Scroll down inside the popup to view all maps at full resolution.

### Dashboard file

| File | Description |
|---|---|
| `model/dashboard.py` | Flask app — all analysis logic and the web UI |
| `model/test_5.py` | Standalone script version of the same pipeline |
| `model/samples/` | 53 `.pyg.json` grid scenarios ready to run |
| `model/checkpoints/` | Model checkpoint (download separately, see above) |

---

## Original GridSFM

GridSFM is an open-source framework for AC Optimal Power Flow (AC-OPF),
the optimization that determines the cost-minimizing generator dispatch
satisfying all of a power grid's physical and operational constraints.
The framework has two parts:

- **`power_grid/`** is the data pipeline. It turns grid topologies
  into solved AC-OPF scenarios in `.pyg.json` format, and ships a Hugging Face loader for fetching pre-built
  scenarios.
- **`model/`** loads the released GridSFM neural surrogate and runs fast
  AC-OPF inference on those `.pyg.json` scenarios. 

Model checkpoints and pre-built power-grid datasets are hosted on
Hugging Face: [microsoft/gridsfm](https://huggingface.co/collections/microsoft/gridsfm).


## Repository structure

```
GridSFM/
├── model/                  # Neural surrogate model loading & inference
│   ├── gridsfm/            # inference package + gridsfm.hf_util HuggingFace loader
│   ├── samples/            # 53 base scenarios (.pyg.json); see samples/README.md
│   ├── examples/           # infer_samples, opfdata
│   └── tests/              # pytest suite
└── power_grid/
    └── US/
        ├── topology_solver_pipeline/   # Raw topology → solved scenarios
        └── viewer/                     # Browser-based grid data viewer
```

## `model/` — Neural surrogate models

> **Tested OS:** Ubuntu 22.04 / 24.04.

### Typical workflow

1. **[Install](model/README.md#install)** — `cd model && python -m venv .venv && source .venv/bin/activate && pip install -e .`
2. **[Get the checkpoint](model/README.md#get-the-checkpoint)** — `load_from_hf("microsoft/GridSFM_Open")` or download once with `hf download`.
3. **[Run inference](model/README.md#quickstart)** — single-graph via `predict(model, scenario)`, or batched via `model(batch)`. Examples for shipped samples (`examples/infer_samples.py`) and the [OPFData](https://arxiv.org/abs/2406.07234) dataset (`examples/opfdata.py`).

See [model/README.md](model/README.md) for install, checkpoint download, output schema, the column conventions in `gridsfm/schema.py`, and cache customization for large N-1 sweeps.

## `power_grid/` — Grid data and processing pipeline

> **Tested OS:** Ubuntu 24.04 and macOS 26.4.1.

### Typical workflow

1. **[Download the dataset](#gridsfmhf_util--huggingface-dataset-loader)** — use the HuggingFace loader bundled with the `gridsfm` package to fetch the power grid models and OPF results to a local directory.
2. **[Inspect the data](#power_gridusviewer--data-viewer)** — launch the browser-based viewer to explore network topology and OPF results.
3. **[Run the topology solver pipeline](#power_gridustopology_solver_pipeline--raw-topology--solved-scenarios)** — process raw topologies into solved AC-OPF scenarios for model training.

### `gridsfm.hf_util` — HuggingFace dataset loader

A Python utility (`gridsfm_pg_loader.py`) for downloading and loading
GridSFM US power grid models and OPF results from HuggingFace Hub. Shipped
as part of the main `gridsfm` package — no separate `hf_util` install
needed; install `gridsfm` itself per [model/README.md#install](model/README.md#install).

```python
from gridsfm.hf_util import GridSFM_PG_Loader

loader = GridSFM_PG_Loader("microsoft/GridSFM_US_power_grid",
                            export_dir="./gridsfm_data")
model  = loader.load_model("texas", hour="16h")
```

See [model/gridsfm/hf_util/](model/gridsfm/hf_util/) for the loader source.

### `power_grid/US/viewer/` — Data viewer

A lightweight browser-based viewer for inspecting grid data. Requires a
data directory with `16h/` and `04h/` subfolders (e.g. the output of
`GridSFM_PG_Loader.download_all()`).

```bash
cd power_grid/US/viewer
python serve.py --data-dir /path/to/gridsfm_data
```

See the [viewer README](power_grid/US/viewer/README.md) for details.

### `power_grid/US/topology_solver_pipeline/` — Raw topology → solved scenarios

A self-contained Julia pipeline that turns raw grid topologies into
AC-OPF-solved `.pyg.json` scenario files ready for model training and
evaluation. The pipeline has two main stages:

1. **Topology solver** — takes a raw topology JSON and iteratively relaxes
   parameters until strict AC-OPF converges, producing a `.solvable.json`.
2. **Scenario generator** — applies controlled perturbations (load scaling,
   cost shuffling, generator outages, line derating, voltage squeezing) to
   the solvable base grid and solves each variant, emitting one `.pyg.json`
   per scenario.

See the [topology_solver_pipeline README](power_grid/US/topology_solver_pipeline/README.md) for setup and high-level usage, and [PIPELINE_DETAILS.md](power_grid/US/topology_solver_pipeline/PIPELINE_DETAILS.md) for in-depth file and stage documentation.

### Citation

If you use the power grid data or pipeline, please cite:

```bibtex
@article{britto2026powergrid,
  title   = {Building Power Grid Models from Open Data: A Complete Pipeline from OpenStreetMap to Optimal Power Flow},
  author  = {Britto, Andrea and Spina, Thiago and Yang, Weiwei and Fowers, Spencer and Zhang, Baosen and White, Chris},
  year    = {2026},
  note    = {Microsoft Research}
}
```

If you use the GridSFM neural surrogate model (`model/`), please cite:

```bibtex
@unpublished{yang2026gridsfm,
  author   = {Yang, Weiwei and Britto Mattos Lima, Andrea and Spina, Thiago V. and Fowers, Spencer and Zhang, Baosen and White, Chris},
  title    = {GridSFM: A Foundation Model for AC Optimal Power Flow},
  year     = {2026},
  month    = {May},
  url      = {https://www.microsoft.com/en-us/research/publication/gridsfm-a-foundation-model-for-ac-optimal-power-flow/}
}
```
