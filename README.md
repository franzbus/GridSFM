# GridSFM

GridSFM is an open-source framework for neural surrogate modeling of AC
Optimal Power Flow (AC-OPF) on realistic approximations of the US power grids, derived exclusively from open data. It provides
tooling to obtain grid data, process raw topologies into solved AC-OPF
scenarios, and (coming soon) load and run pre-trained surrogate models for
fast AC-OPF estimation.

Model checkpoints and power grid datasets are available on HuggingFace:
[microsoft/gridsfm](https://huggingface.co/collections/microsoft/gridsfm).
Use the instructions below to install our loaders and facilitate data loading.

## Repository structure

```
GridSFM/
├── model/              # Neural surrogate model loading & inference [Coming soon]
└── power_grid/
    ├── hf_util/        # HuggingFace dataset loader
    └── US/
        ├── topology_solver_pipeline/   # Raw topology → solved scenarios
        └── viewer/                     # Browser-based grid data viewer
```

## `model/` — Neural surrogate models

**[Coming soon]** — This directory will contain code and documentation for loading
and running GridSFM neural surrogate models for AC-OPF estimation on power
grids. Details will be added once the model artifacts and inference code are
available.

## `power_grid/` — Grid data and processing pipeline

### Typical workflow

1. **[Download the dataset](#power_gridhf_util--huggingface-dataset-loader)** — install the HuggingFace loader and fetch the power grid models to a local directory.
2. **[Inspect the data](#power_gridusviewer--data-viewer)** — launch the browser-based viewer to explore network topology and OPF results.
3. **[Run the topology solver pipeline](#power_gridustopology_solver_pipeline--raw-topology--solved-scenarios)** — process raw topologies into solved AC-OPF scenarios for model training.

### `power_grid/hf_util/` — HuggingFace dataset loader

A Python utility (`gridsfm_pg_loader.py`) for downloading and loading
GridSFM US power grid models and OPF results from HuggingFace Hub.

```bash
pip install ./power_grid/hf_util
```

```python
from gridsfm.hf_util import GridSFM_PG_Loader

loader = GridSFM_PG_Loader("microsoft/GridSFM_US_power_grid_[model_date]",
                            export_dir="./gridsfm_data")
model  = loader.load_model("texas", hour="16h")
```

See the [hf_util README](power_grid/hf_util/README.md) for full usage.

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

See [power_grid/US/topology_solver_pipeline/README.md](power_grid/US/topology_solver_pipeline/README.md) for full documentation, file descriptions, and usage examples.

## License

This project is licensed under the [MIT License](LICENSE).
