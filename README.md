# GridSFM

GridSFM is an open-source framework for neural surrogate modeling of AC
Optimal Power Flow (AC-OPF) on realistic approximations of the US power grids, derived exclusively from open data. It provides
tooling to obtain grid data, process raw topologies into solved AC-OPF
scenarios, and (coming soon) load and run pre-trained surrogate models for
fast AC-OPF estimation.

## Repository structure

```
GridSFM/
├── model/              # Neural surrogate model loading & inference [TODO]
└── power_grid/
    ├── hf_util/        # HuggingFace dataset loader
    └── US/
        ├── topology_solver_pipeline/   # Raw topology → solved scenarios
        └── viewer/                     # Browser-based grid data viewer
```

### `model/` — Neural surrogate models

**[TODO]** — This directory will contain code and documentation for loading
and running GridSFM neural surrogate models for AC-OPF estimation on power
grids. Details will be added once the model artifacts and inference code are
available.

### `power_grid/` — Grid data and processing pipeline

#### `power_grid/hf_util/` — HuggingFace dataset loader

A Python utility (`gridsfm_pg_loader.py`) for downloading and loading
GridSFM US power grid models and OPF results from HuggingFace Hub.

```bash
pip install huggingface_hub
```

```python
from gridsfm_pg_loader import GridSFM_PG_Loader

loader = GridSFM_PG_Loader("microsoft/GridSFM_US_power_grid_[model_date]")
model  = loader.load_model("texas", hour="16h")
```

See the module docstring in [power_grid/hf_util/gridsfm_pg_loader.py](power_grid/hf_util/gridsfm_pg_loader.py) for full usage.

#### `power_grid/US/topology_solver_pipeline/` — Raw topology → solved scenarios

A self-contained Julia pipeline that turns raw grid topologies into
AC-OPF-solved `.pyg.json` scenario files ready for model training and
evaluation. The pipeline has two main stages:

1. **Topology solver** — takes a raw topology JSON and iteratively relaxes
   parameters until strict AC-OPF converges, producing a `.solvable.json`.
2. **Scenario generator** — applies controlled perturbations (load scaling,
   cost shuffling, generator outages, line derating, voltage squeezing) to
   the solvable base grid and solves each variant, emitting one `.pyg.json`
   per scenario.

See [power_grid/US/topology_solver_pipeline/README.md](power_grid/US/topology_solver_pipeline/README.md) for full documentation, file descriptions, and usage examples.

#### `power_grid/US/viewer/` — Data viewer

A lightweight browser-based viewer for inspecting grid data. Serve locally
with `python serve.py` inside the `viewer/` directory.

## License

This project is licensed under the [MIT License](LICENSE).
