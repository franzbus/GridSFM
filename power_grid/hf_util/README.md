# GridSFM US Power Grid — HuggingFace Loader

Download and load GridSFM US power grid models and OPF results from
HuggingFace Hub using `gridsfm_pg_loader.py`.

## Install

```bash
pip install huggingface_hub
```

## Usage

```python
from gridsfm_pg_loader import GridSFM_PG_Loader

# Basic: files stay in HuggingFace cache
loader = GridSFM_PG_Loader("microsoft/GridSFM_US_power_grid_[model_date]")

# With export_dir: the entire dataset is pre-fetched to a local directory
loader = GridSFM_PG_Loader(
    "microsoft/GridSFM_US_power_grid_[model_date]",
    export_dir="./gridsfm_data",
)

# To skip the automatic download, set pre_fetch_all=False
loader = GridSFM_PG_Loader(
    "microsoft/GridSFM_US_power_grid_[model_date]",
    export_dir="./gridsfm_data",
    pre_fetch_all=False,
)

# Load a single model (case-insensitive; abbreviations work too)
model = loader.load_model("TX", hour="16h")
model = loader.load_model("texas", hour="16h")

# Load OPF results
ac = loader.load_ac_results("texas", hour="16h")
dc = loader.load_dc_results("texas", hour="16h")

# Load everything for a region
bundle = loader.load_bundle("texas", hour="16h")
# bundle["model"], bundle["ac_results"], bundle["dc_results"]

# Export a single file to a specific path
loader.export_file("TX", "model", hour="16h", dest="./my_models/texas.json")

# Save a (possibly modified) model dict back to JSON
loader.save_json(model, "./my_models/texas_modified.json")

# List available regions/states
print(loader.list_regions())

# Download all files to a local directory
loader.download_all("./data")
```
| AC1 | AC1 — Voltage + Q | Voltage [0.90, 1.10], Q limits ×1.5 (AC-OPF only) |

The `relaxation_level` and `relaxation_label` fields in results files indicate which level was needed.

## License

This data is released under the [MIT License](LICENSE).
