"""GridSFM AC-OPF inference package."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Union

import torch
from torch_geometric.data import HeteroData

from . import schema
from .checkpoint import load_model, load_from_hf
from .data import (
    batch_data_list,
    load_pyg_json,
    load_opfdata,
    prepare_for_inference,
)
from .model import GridTransformerBackbone
from .schema import AC_LINE_KEY, TRANSFORMER_KEY

__version__ = "1.0.0"


@torch.no_grad()
def predict(
    model: GridTransformerBackbone,
    scenario: Union[str, Path, HeteroData],
    fmt: str = "auto",
) -> Dict[str, torch.Tensor]:
    if isinstance(scenario, HeteroData):
        data = deepcopy(scenario)
    else:
        path = Path(scenario)
        if fmt == "auto":
            if path.suffixes[-2:] == [".pyg", ".json"]:
                fmt = "pyg"
            elif path.suffix == ".json":
                fmt = "opfdata"
            else:
                raise ValueError(
                    f"Cannot auto-detect format for {path}. "
                    f"Pass fmt='pyg' or fmt='opfdata' explicitly."
                )
        if fmt == "pyg":
            data = load_pyg_json(path)
        elif fmt == "opfdata":
            data = load_opfdata(path)
        else:
            raise ValueError(f"Unknown fmt={fmt!r}. Expected 'auto', 'pyg', or 'opfdata'.")
    data = prepare_for_inference(data)

    device = next(model.parameters()).device
    data = data.to(device)
    if not hasattr(data["bus"], "batch"):
        data["bus"].batch = torch.zeros(data["bus"].x.size(0), dtype=torch.long, device=device)
    for nt in GridTransformerBackbone.NODE_TYPES:
        if nt == "bus":
            continue
        if nt in data.node_types and not hasattr(data[nt], "batch"):
            data[nt].batch = torch.zeros(data[nt].x.size(0), dtype=torch.long, device=device)
    n_graphs = int(getattr(data, "num_graphs", 1) or 1)
    if n_graphs > 1:
        raise ValueError(
            "predict() handles a single scenario; for multi-graph batches use "
            "batch_data_list(prepared) and call model(batch) directly."
        )
    data.num_graphs = 1

    out = model(data)

    bus_pred = out["bus"].pred
    gen_pred = out["generator"].pred
    flows = []
    flow_edge_types: list[str] = []
    flow_edge_counts: list[int] = []
    for k in (AC_LINE_KEY, TRANSFORMER_KEY):
        if k in out.edge_types and hasattr(out[k], "edge_flow_pred"):
            e = out[k].edge_flow_pred
            flows.append(e)
            flow_edge_types.append(k[1])
            flow_edge_counts.append(int(e.size(0)))
    if flows:
        flows = torch.cat(flows, dim=0)
    else:
        flows = torch.zeros(0, 4, device=device, dtype=bus_pred.dtype)

    feas_logit = float(out.feas_logit.item())
    return {
        "theta":      bus_pred[:, 0].cpu(),
        "V":          bus_pred[:, 1].cpu(),
        "Pg":         gen_pred[:, 0].cpu(),
        "Qg":         gen_pred[:, 1].cpu(),
        "Pij":        flows[:, 0].cpu(),
        "Qij":        flows[:, 1].cpu(),
        "Pji":        flows[:, 2].cpu(),
        "Qji":        flows[:, 3].cpu(),
        "flow_edge_types":  flow_edge_types,
        "flow_edge_counts": flow_edge_counts,
        "feas":       float(torch.sigmoid(out.feas_logit).item()),
        "feas_logit": feas_logit,
    }


__all__ = [
    "GridTransformerBackbone",
    "batch_data_list",
    "load_model",
    "load_from_hf",
    "load_pyg_json",
    "load_opfdata",
    "prepare_for_inference",
    "predict",
    "schema",
    "__version__",
]
