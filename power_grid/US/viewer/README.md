# GridSFM Data Viewer

An interactive browser-based viewer for inspecting GridSFM US power grid
data. Visualizes network topology on a map and shows OPF results for any
state or region.

## Requirements

Python 3.10+ (stdlib only — no pip packages needed). The front-end uses
[Leaflet.js](https://leafletjs.com/) loaded from CDN.

## Usage

```bash
cd viewer
python serve.py --data-dir ../ --port 8050
```

Then open `http://localhost:8050` in a browser.

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `../` (parent of `viewer/`) | Root directory containing `16h/` and `04h/` folders |
| `--port` | `8050` | HTTP server port |

## Views

1. **Network Model** — bus-branch topology on a Leaflet map, color-coded by voltage level
2. **OPF Summary** — generation mix, cost breakdown, solver status
3. **Economic Dispatch** — generator dispatch stack by fuel type
4. **Line Congestion** — branch loading and thermal limit utilization
