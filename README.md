# GridSFM — Congestion Mitigation Dashboard

Interactive web dashboard for power grid congestion analysis built on top of the [GridSFM](https://github.com/microsoft/GridSFM) neural surrogate model for AC Optimal Power Flow (AC-OPF).

Given a grid scenario, the dashboard runs a full mitigation pipeline — predicts line thermal loading, identifies congested lines, shuts down nearby charging batteries, re-runs the model, and reports the economic outcome — all visualised step by step in the browser.

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/franzbus/GridSFM.git
cd GridSFM/model
```

### 2. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # installs the gridsfm package
pip install -r requirements.txt  # installs Flask and remaining deps
```

### 3. Download the model checkpoint

```python
from gridsfm import load_from_hf
load_from_hf("microsoft/GridSFM_Open")  # saves to model/checkpoints/
```

Or download `gridsfm_open_v1.1.pt` manually from  
https://huggingface.co/microsoft/GridSFM_Open and place it in `model/checkpoints/`.

### 4. Launch the dashboard

```bash
python dashboard.py
```

Open **http://localhost:5050** in your browser.

---

## How it works

1. **Select a grid** from the left panel — 53 scenarios are included (US state grids and standard case studies).
2. Click **Run Analysis**. A results popup opens and fills in three sequential steps:

| Step | What happens |
|---|---|
| **1 — Baseline prediction** | GridSFM predicts line flows. Full network map + zoom on the worst congested line are shown. |
| **2 — Battery shutdown** | Charging batteries within 2 hops of the congested line are curtailed. GridSFM re-predicts and updated maps are shown. |
| **3 — Mitigation report** | Loading before/after, overload cleared %, cost of battery compensation vs emergency redispatch, and a Case A/B/C verdict. |

3. Scroll inside the popup to review all maps at full resolution.

---

## Repository structure

```
GridSFM/
└── model/
    ├── dashboard.py        # Flask dashboard (this project)
    ├── test_5.py           # Standalone script version of the pipeline
    ├── gridsfm/            # GridSFM inference package
    ├── samples/            # 53 grid scenarios (.pyg.json)
    ├── checkpoints/        # Model checkpoint (downloaded separately)
    └── requirements.txt    # Python dependencies
```

---

## Requirements

- Python 3.10+
- See `model/requirements.txt` for the full dependency list
- Model checkpoint from Hugging Face (see step 3 above)
