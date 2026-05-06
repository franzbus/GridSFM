# GridSFM US Power Grid — HuggingFace Loader

Download and load GridSFM US power grid models and OPF results from
HuggingFace Hub using `gridsfm_pg_loader.py`.

## Install

From the repository root:

```bash
pip install ./power_grid/hf_util
```

## Usage

```python
from gridsfm.hf_util import GridSFM_PG_Loader

# Basic: files stay in HuggingFace cache
loader = GridSFM_PG_Loader("microsoft/GridSFM_US_power_grid")

# With export_dir: the entire dataset is pre-fetched to a local directory
loader = GridSFM_PG_Loader(
    "microsoft/GridSFM_US_power_grid",
    export_dir="./gridsfm_data",
)

# To skip the automatic download, set pre_fetch_all=False
loader = GridSFM_PG_Loader(
    "microsoft/GridSFM_US_power_grid",
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

## License

This project is released under the [MIT License](../../LICENSE).
