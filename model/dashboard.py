#!/usr/bin/env python3
"""
Grid Congestion Mitigation Dashboard
Flask app — mirrors the test_5.py analysis flow (baseline predict →
battery shutdown + re-predict → mitigation report) without run_optimization_scenarios.
Run:  python dashboard.py   then open http://localhost:5050
"""
import copy
import io
import json
import base64
import os
import sys
import uuid
import threading

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx

from flask import Flask, Response, render_template_string, request, stream_with_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gridsfm import load_model, predict

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
MODEL_PATH  = os.path.join(BASE_DIR, "checkpoints", "gridsfm_open_v1.1.pt")
TMP_DIR     = os.path.join(BASE_DIR, "_tmp_dashboard")
os.makedirs(TMP_DIR, exist_ok=True)

CONFIG = {
    "emergency_peaker_cost_per_mwh": 150.0,
    "battery_curtailment_cost_per_mwh": 20.0,
    "base_mva": 100.0,
    "dispatch_duration_hours": 1.0,
}

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                _model = load_model(MODEL_PATH, device=device)
    return _model


# ── Analysis helpers (logic from test_5.py, no imports to avoid side-effects) ─

def compute_line_loading(out):
    Pij = torch.as_tensor(out["Pij"], dtype=torch.float64)
    Qij = torch.as_tensor(out["Qij"], dtype=torch.float64)
    Pji = torch.as_tensor(out["Pji"], dtype=torch.float64)
    Qji = torch.as_tensor(out["Qji"], dtype=torch.float64)
    Sij = torch.sqrt(Pij ** 2 + Qij ** 2)
    Sji = torch.sqrt(Pji ** 2 + Qji ** 2)
    return {
        "S_max": torch.maximum(Sij, Sji),
        "flow_edge_types": out.get("flow_edge_types", []),
        "flow_edge_counts": out.get("flow_edge_counts", []),
    }


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def make_loading_figures(loading, graph_filename):
    """Return (b64_full, b64_zoom, worst, congested_sorted)."""
    with open(graph_filename) as f:
        g = json.load(f)

    S_max  = loading["S_max"]
    types  = loading["flow_edge_types"]
    counts = loading["flow_edge_counts"]
    nodes  = g["grid"]["nodes"]
    edges  = g["grid"]["edges"]
    schema = g["metadata"].get("feature_schema", {})

    def _rate_col(fam):
        feats = schema.get(fam, [])
        return feats.index("rate_a_mva") if "rate_a_mva" in feats else (6 if fam == "ac_line" else 4)

    gen_buses  = set(edges["generator_link"].get("receivers", []))
    load_buses = set(edges["load_link"].get("receivers", []))

    def _bus_color(i):
        if i in gen_buses and i in load_buses: return "#f4a261"
        if i in gen_buses:                     return "#31a354"
        if i in load_buses:                    return "#e6550d"
        return "#9ecae1"

    G = nx.Graph()
    n_buses = len(nodes["bus"])
    for i in range(n_buses):
        G.add_node(i)

    branch_edges, edge_colors = [], []
    congested = []
    offset = 0
    global_idx = 0
    for fam, cnt in zip(types, counts):
        rate_col = _rate_col(fam)
        snd  = edges[fam]["senders"]
        rcv  = edges[fam]["receivers"]
        feat = edges[fam]["features"]
        for k in range(cnt):
            rate_a = float(feat[k][rate_col])
            s      = float(S_max[offset + k])
            color  = "red" if (rate_a > 0 and s > rate_a) else "green"
            G.add_edge(snd[k], rcv[k])
            branch_edges.append((snd[k], rcv[k]))
            edge_colors.append(color)
            if color == "red":
                ratio = s / rate_a if rate_a > 0 else float("inf")
                congested.append((global_idx, fam, k, snd[k], rcv[k], s, rate_a, ratio))
            global_idx += 1
        offset += cnt

    worst = None
    congested_sorted = []
    if congested:
        congested_sorted = sorted(congested, key=lambda x: x[7], reverse=True)
        worst = congested_sorted[0]

    pos = nx.spring_layout(G, seed=0)
    node_colors = [_bus_color(n) for n in G.nodes]

    legend_handles = [
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#9ecae1", markersize=8, label="Bus (plain)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#31a354", markersize=8, label="Bus w/ generator"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#e6550d", markersize=8, label="Bus w/ load"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#f4a261", markersize=8, label="Bus w/ gen+load"),
        plt.Line2D([0],[0], color="green", lw=2, label="S_max ≤ rate_a"),
        plt.Line2D([0],[0], color="red",   lw=2, label="S_max > rate_a"),
    ]
    if worst is not None:
        legend_handles.append(plt.Line2D([0],[0], color="gold", lw=4,
            label=f"Worst: idx {worst[0]} ({worst[1]}, bus {worst[3]}→{worst[4]}, ratio {worst[7]:.3f})"))

    def _draw(ax, highlight=True):
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=20, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=branch_edges, edge_color=edge_colors, width=1.2, ax=ax)
        if highlight and worst is not None:
            nx.draw_networkx_edges(G, pos, edgelist=[branch_edges[worst[0]]],
                                   edge_color="gold", width=6.0, ax=ax)
        ax.axis("off")

    fig1, ax1 = plt.subplots(figsize=(16, 12))
    fig1.patch.set_facecolor("#1a1a2e")
    ax1.set_facecolor("#1a1a2e")
    _draw(ax1)
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=9,
               facecolor="#16213e", edgecolor="#0f3460", labelcolor="white")
    ax1.set_title("Full network — line thermal loading", fontsize=13, color="white", pad=12)
    fig1.tight_layout()
    b64_full = _fig_to_b64(fig1)

    b64_zoom = None
    if worst is not None:
        w_fr, w_to = worst[3], worst[4]
        x0, y0 = pos[w_fr]
        x1, y1 = pos[w_to]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pad = 0.15

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        fig2.patch.set_facecolor("#1a1a2e")
        ax2.set_facecolor("#1a1a2e")
        _draw(ax2, highlight=True)
        ax2.set_xlim(cx - pad, cx + pad)
        ax2.set_ylim(cy - pad, cy + pad)
        ax2.legend(handles=legend_handles, loc="upper right", fontsize=9,
                   facecolor="#16213e", edgecolor="#0f3460", labelcolor="white")
        ax2.set_title(
            f"Zoom — worst congested edge  (idx {worst[0]}, {worst[1]}, "
            f"bus {w_fr}→{w_to}, S/rate_a = {worst[7]:.3f})",
            fontsize=11, color="white", pad=10)
        fig2.tight_layout()
        b64_zoom = _fig_to_b64(fig2)

    return b64_full, b64_zoom, worst, congested_sorted


def shutdown_batteries(graph_filename_or_data, worst, n_hops=2, max_batteries=3):
    """Return (modified_dict, shutdown_details)."""
    if isinstance(graph_filename_or_data, str):
        with open(graph_filename_or_data) as f:
            d = json.load(f)
    else:
        d = graph_filename_or_data
    d = copy.deepcopy(d)

    nodes = d["grid"]["nodes"]
    edges = d["grid"]["edges"]
    meta  = d["metadata"]

    _, _, _, from_bus, to_bus, *_ = worst
    n_buses = len(nodes["bus"])
    adj = {i: set() for i in range(n_buses)}
    for fam in ("ac_line", "transformer"):
        for s, r in zip(edges[fam]["senders"], edges[fam]["receivers"]):
            adj[s].add(r); adj[r].add(s)

    def _bfs(start):
        visited, frontier = {start}, {start}
        for _ in range(n_hops):
            frontier = {nb for nd in frontier for nb in adj[nd]} - visited
            visited |= frontier
        return visited

    neighbourhood = _bfs(from_bus) | _bfs(to_bus)
    ll_s = edges["load_link"]["senders"]
    ll_r = edges["load_link"]["receivers"]
    load_to_bus = {ll_s[i]: ll_r[i] for i in range(len(ll_s))}

    schema     = meta.get("feature_schema", {})
    load_feats = schema.get("load", ["pd_pu", "qd_pu"])
    idx_pd = load_feats.index("pd_pu") if "pd_pu" in load_feats else 0
    idx_qd = load_feats.index("qd_pu") if "qd_pu" in load_feats else 1

    candidates = []
    for load_idx, bus_row in sorted(load_to_bus.items()):
        if bus_row not in neighbourhood:
            continue
        feat    = nodes["load"][load_idx]
        orig_pd = feat[idx_pd]
        orig_qd = feat[idx_qd]
        if orig_pd <= 0:
            continue
        candidates.append((load_idx, orig_pd, orig_qd))

    candidates.sort(key=lambda x: x[1], reverse=True)
    if max_batteries is not None:
        candidates = candidates[:max_batteries]

    shutdown_details = []
    for load_idx, orig_pd, orig_qd in candidates:
        nodes["load"][load_idx][idx_pd] = 0.0
        nodes["load"][load_idx][idx_qd] = 0.0
        shutdown_details.append((load_idx, orig_pd, orig_qd))

    return d, shutdown_details


def analyze_mitigation(loading_before, loading_after, worst, shutdown_details):
    cfg = CONFIG
    global_idx = worst[0]
    s_before   = float(worst[5])
    rate_a     = float(worst[6])
    s_after    = float(loading_after["S_max"][global_idx])

    overload      = max(0.0, s_before - rate_a)
    delta_s       = s_before - s_after
    total_delta_p = sum(pd for _, pd, _ in shutdown_details)
    impact_factor = delta_s / total_delta_p if total_delta_p > 1e-9 else 0.0

    if s_after < rate_a:
        case = "A"
    elif delta_s > 1e-4:
        case = "B"
    else:
        case = "C"

    if case == "A" and impact_factor > 1e-9:
        min_delta_p_needed = overload / impact_factor
        sorted_batt = sorted(shutdown_details, key=lambda x: x[1], reverse=True)
        min_subset, running = [], 0.0
        for item in sorted_batt:
            min_subset.append(item)
            running += item[1]
            if running >= min_delta_p_needed:
                break
    else:
        min_subset = shutdown_details

    base_mva = cfg["base_mva"]
    duration = cfg["dispatch_duration_hours"]
    min_curtailment_mw = sum(pd for _, pd, _ in min_subset) * base_mva

    redispatch_cost = overload * base_mva * cfg["emergency_peaker_cost_per_mwh"] * duration
    battery_cost    = min_curtailment_mw * cfg["battery_curtailment_cost_per_mwh"] * duration
    savings         = redispatch_cost - battery_cost

    return {
        "case":                 case,
        "s_before_pu":          round(s_before, 6),
        "s_after_pu":           round(s_after, 6),
        "rate_a_pu":            round(rate_a, 6),
        "delta_s_pu":           round(delta_s, 6),
        "overload_cleared_pct": round(delta_s / overload * 100, 1) if overload > 1e-9 else 100.0,
        "impact_factor":        round(impact_factor, 6),
        "batteries_curtailed":  len(shutdown_details),
        "total_curtailment_mw": round(total_delta_p * base_mva, 2),
        "min_curtailment_mw":   round(min_curtailment_mw, 2),
        "min_battery_subset":   [idx for idx, _, _ in min_subset],
        "redispatch_cost_usd":  round(redispatch_cost, 2),
        "battery_comp_usd":     round(battery_cost, 2),
        "savings_usd":          round(savings, 2),
        "action":               "BATTERY_CURTAILMENT" if case in ("A", "B") else "PEAKER_DISPATCH",
        "economically_viable":  savings > 0 and case in ("A", "B"),
    }


def make_report_figure(worst, analysis, graph_filename):
    """Return base64-encoded mitigation dashboard figure."""
    with open(graph_filename) as f:
        g = json.load(f)
    nodes = g["grid"]["nodes"]
    edges = g["grid"]["edges"]

    a          = analysis
    rate_a     = a["rate_a_pu"]
    pct_before = a["s_before_pu"] / rate_a * 100 if rate_a > 0 else 0
    pct_after  = a["s_after_pu"]  / rate_a * 100 if rate_a > 0 else 0

    case_color = {"A": "#2a9d8f", "B": "#f4a261", "C": "#e63946"}[a["case"]]
    case_label = {"A": "Overload\ncleared ✓", "B": "Partially\ncleared ~", "C": "No\neffect ✗"}[a["case"]]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#f0f2f5")

    gs_outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.4],
                                 hspace=0.38, top=0.91, bottom=0.05, left=0.04, right=0.97)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0], wspace=0.28)
    ax_s   = fig.add_subplot(gs_top[0])
    ax_l   = fig.add_subplot(gs_top[1])
    ax_e   = fig.add_subplot(gs_top[2])
    ax_net = fig.add_subplot(gs_outer[1])

    grid_name = os.path.basename(graph_filename).replace(".pyg.json", "")
    fig.suptitle(
        f"Congestion Mitigation Report  ·  Grid: {grid_name}  ·  "
        f"{worst[1].replace('_', ' ').title()} #{worst[0]}  "
        f"(bus {worst[3]} → bus {worst[4]})",
        fontsize=13, fontweight="bold", color="#1a1a2e", y=0.97,
    )

    for ax in (ax_s, ax_l, ax_e):
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Panel 1: Status
    ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.set_xticks([]); ax_s.set_yticks([])
    ax_s.add_patch(plt.Circle((0.22, 0.70), 0.14, color="#e63946", zorder=3, transform=ax_s.transData))
    ax_s.text(0.22, 0.70, "!", ha="center", va="center", fontsize=22, fontweight="bold", color="white", zorder=4)
    ax_s.text(0.22, 0.50, "CONGESTED", ha="center", fontsize=8, color="#e63946", fontweight="bold")
    ax_s.annotate("", xy=(0.60, 0.70), xytext=(0.40, 0.70),
                  arrowprops=dict(arrowstyle="-|>", color="#888", lw=2.0, mutation_scale=18))
    ax_s.add_patch(plt.Circle((0.78, 0.70), 0.14, color=case_color, zorder=3, transform=ax_s.transData))
    ax_s.text(0.78, 0.70, case_label.split("\n")[1][0],
              ha="center", va="center", fontsize=22, fontweight="bold", color="white", zorder=4)
    ax_s.text(0.78, 0.50, case_label, ha="center", fontsize=8,
              color=case_color, fontweight="bold", multialignment="center")
    action = "BATTERY CURTAILMENT" if a["economically_viable"] else "EMERGENCY PEAKER"
    bg = "#2a9d8f22" if a["economically_viable"] else "#e6394622"
    ec = "#2a9d8f" if a["economically_viable"] else "#e63946"
    ax_s.text(0.50, 0.24, f"Action: {action}", ha="center", va="center",
              fontsize=8.5, fontweight="bold", color=ec,
              bbox=dict(boxstyle="round,pad=0.4", facecolor=bg, edgecolor=ec, lw=1.2))
    ax_s.set_title("Grid Status", fontweight="bold", fontsize=11, pad=8)

    # Panel 2: Loading gauge
    ax_l.set_xlim(-10, 240); ax_l.set_ylim(0, 1); ax_l.set_xticks([]); ax_l.set_yticks([])

    def _gauge(ax, y, pct, label, color):
        ax.barh(y, 200, height=0.14, color="#e0e0e0", left=0, zorder=1)
        ax.barh(y, min(pct, 200), height=0.14, color=color, left=0, zorder=2, alpha=0.88)
        ax.text(-8, y, label, va="center", ha="right", fontsize=9, color="#444")
        ax.text(min(pct, 200) + 4, y, f"{pct:.0f}%", va="center", fontsize=10, fontweight="bold", color=color)

    _gauge(ax_l, 0.72, pct_before, "Before", "#e63946" if pct_before > 100 else "#f4a261")
    _gauge(ax_l, 0.38, pct_after,  "After",  "#e63946" if pct_after  > 100 else "#2a9d8f")
    ax_l.axvline(100, ymin=0.18, ymax=0.95, color="#333", lw=1.8, linestyle="--", zorder=5)
    ax_l.text(100, 0.96, "Thermal\nlimit", ha="center", va="top", fontsize=7.5, color="#333", multialignment="center")
    ax_l.set_title("Line Loading  (% of thermal limit)", fontweight="bold", fontsize=11, pad=8)

    # Panel 3: Economics
    labels = ["Redispatch\n(peaker)", "Battery\ncomp."]
    vals   = [a["redispatch_cost_usd"], a["battery_comp_usd"]]
    colors = ["#e63946", "#2a9d8f"]
    bars   = ax_e.bar(labels, vals, color=colors, alpha=0.85, width=0.45, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax_e.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + max(vals) * 0.03,
                  f"${v:,.0f}", ha="center", fontsize=10, fontweight="bold")
    if a["savings_usd"] > 0:
        ax_e.text(0.50, 0.97, f"Savings: ${a['savings_usd']:,.0f}",
                  ha="center", va="top", transform=ax_e.transAxes,
                  fontsize=11, fontweight="bold", color="#2a9d8f",
                  bbox=dict(boxstyle="round,pad=0.35", facecolor="#2a9d8f18", edgecolor="#2a9d8f", lw=1.2))
    ax_e.set_ylabel("Cost per dispatch hour (USD)", fontsize=8.5)
    ax_e.tick_params(labelsize=9); ax_e.spines[["top", "right"]].set_visible(False)
    ax_e.set_title("Cost Comparison", fontweight="bold", fontsize=11, pad=8)

    # Panel 4: Neighbourhood network
    ax_net.set_facecolor("white")
    for sp in ax_net.spines.values(): sp.set_visible(False)

    _, _, _, from_bus, to_bus, *_ = worst
    n_buses = len(nodes["bus"])
    adj = {i: set() for i in range(n_buses)}
    for fam in ("ac_line", "transformer"):
        for s, r in zip(edges[fam]["senders"], edges[fam]["receivers"]):
            adj[s].add(r); adj[r].add(s)

    def _bfs(start, hops=3):
        visited, frontier = {start}, {start}
        for _ in range(hops):
            frontier = {nb for nd in frontier for nb in adj[nd]} - visited
            visited |= frontier
        return visited

    viz_buses = _bfs(from_bus) | _bfs(to_bus)
    ll_s = edges["load_link"]["senders"]
    ll_r = edges["load_link"]["receivers"]
    load_to_bus = {ll_s[i]: ll_r[i] for i in range(len(ll_s))}
    shut_buses  = {load_to_bus[li] for li in a.get("min_battery_subset", []) if li in load_to_bus}

    G_sub = nx.Graph()
    G_sub.add_nodes_from(viz_buses)
    for fam in ("ac_line", "transformer"):
        for s, r in zip(edges[fam]["senders"], edges[fam]["receivers"]):
            if s in viz_buses and r in viz_buses:
                G_sub.add_edge(s, r)

    pos_sub  = nx.spring_layout(G_sub, seed=42)
    edgelist = list(G_sub.edges())
    worst_set = {from_bus, to_bus}

    nc  = ["#e63946" if n in worst_set else "#f4a261" if n in shut_buses else "#aec6cf" for n in G_sub.nodes()]
    ec_sub = ["#e63946" if set(e) == worst_set else "#bbbbbb" for e in edgelist]
    ew_sub = [4.5       if set(e) == worst_set else 1.0       for e in edgelist]

    nx.draw_networkx_nodes(G_sub, pos_sub, node_color=nc, node_size=160, ax=ax_net)
    nx.draw_networkx_edges(G_sub, pos_sub, edgelist=edgelist, edge_color=ec_sub, width=ew_sub, ax=ax_net)
    nx.draw_networkx_labels(G_sub, pos_sub, font_size=6, font_color="white", ax=ax_net)
    ax_net.axis("off")

    legend_items = [
        mpatches.Patch(color="#e63946", label=f"Congested line endpoints (bus {from_bus} & {to_bus})"),
        mpatches.Patch(color="#f4a261",
                       label=f"Batteries stopped  ({len(a.get('min_battery_subset', []))} units · {a['min_curtailment_mw']:.1f} MW)"),
        mpatches.Patch(color="#aec6cf", label="Other buses in affected area"),
    ]
    ax_net.legend(handles=legend_items, loc="lower right", fontsize=9, framealpha=0.92, edgecolor="#ccc")
    ax_net.set_title(
        f"Affected area  ·  3-hop neighbourhood  ·  {len(viz_buses)} buses shown",
        fontweight="bold", fontsize=10, pad=8,
    )

    return _fig_to_b64(fig)


# ── Flask Application ─────────────────────────────────────────────────────────

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Grid Congestion Mitigation Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0d1117;
      --sidebar:   #161b22;
      --card:      #1c2128;
      --border:    #30363d;
      --accent:    #58a6ff;
      --green:     #3fb950;
      --red:       #f85149;
      --orange:    #f4a261;
      --teal:      #2a9d8f;
      --text:      #e6edf3;
      --muted:     #8b949e;
      --radius:    10px;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }

    /* ── Sidebar ── */
    .sidebar {
      width: 280px;
      min-width: 280px;
      background: var(--sidebar);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .sidebar-header {
      padding: 20px 16px 14px;
      border-bottom: 1px solid var(--border);
    }

    .sidebar-header h1 {
      font-size: 15px;
      font-weight: 600;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .sidebar-header h1 .icon {
      width: 28px; height: 28px;
      background: linear-gradient(135deg, #1f6feb, #388bfd);
      border-radius: 6px;
      display: flex; align-items: center; justify-content: center;
      font-size: 14px;
    }

    .sidebar-header p {
      font-size: 11px;
      color: var(--muted);
      margin-top: 4px;
    }

    .search-box {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
    }

    .search-box input {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 10px;
      color: var(--text);
      font-size: 12px;
      outline: none;
    }

    .search-box input:focus { border-color: var(--accent); }
    .search-box input::placeholder { color: var(--muted); }

    .sample-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px 0;
    }

    .sample-group { margin-bottom: 4px; }

    .sample-group-header {
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      padding: 8px 14px 4px;
    }

    .sample-btn {
      display: block;
      width: 100%;
      text-align: left;
      background: none;
      border: none;
      color: var(--text);
      font-size: 12px;
      padding: 6px 14px;
      cursor: pointer;
      border-radius: 4px;
      transition: background 0.15s;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .sample-btn:hover { background: rgba(88,166,255,0.08); }

    .sample-btn.active {
      background: rgba(88,166,255,0.14);
      color: var(--accent);
      font-weight: 500;
    }

    /* ── Main content ── */
    .main {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .topbar {
      background: var(--sidebar);
      border-bottom: 1px solid var(--border);
      padding: 12px 20px;
      display: flex;
      align-items: center;
      gap: 14px;
      flex-shrink: 0;
    }

    .topbar h2 {
      font-size: 14px;
      font-weight: 600;
      flex: 1;
    }

    /* Progress steps */
    .steps {
      display: flex;
      align-items: center;
      gap: 0;
    }

    .step {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      color: var(--muted);
    }

    .step .dot {
      width: 22px; height: 22px;
      border-radius: 50%;
      border: 2px solid var(--border);
      display: flex; align-items: center; justify-content: center;
      font-size: 10px;
      font-weight: 700;
      transition: all 0.3s;
    }

    .step.active .dot   { border-color: var(--accent); color: var(--accent); background: rgba(88,166,255,0.12); }
    .step.done .dot     { border-color: var(--green); color: white; background: var(--green); }
    .step.active        { color: var(--text); }
    .step.done          { color: var(--green); }

    .step-line { width: 32px; height: 2px; background: var(--border); margin: 0 4px; }
    .step-line.done { background: var(--green); }

    .run-btn {
      background: linear-gradient(135deg, #1f6feb, #388bfd);
      color: white;
      border: none;
      border-radius: 6px;
      padding: 7px 16px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .run-btn:hover:not(:disabled) { opacity: 0.88; }
    .run-btn:disabled { opacity: 0.45; cursor: not-allowed; }

    /* Content scroll area */
    .content {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Empty state */
    .empty-state {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      color: var(--muted);
      text-align: center;
    }

    .empty-state .big-icon { font-size: 48px; opacity: 0.4; }
    .empty-state h3 { font-size: 16px; color: var(--text); }
    .empty-state p  { font-size: 13px; max-width: 340px; }

    /* Status bar */
    .status-bar {
      background: rgba(88,166,255,0.08);
      border: 1px solid rgba(88,166,255,0.2);
      border-radius: 8px;
      padding: 10px 16px;
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
    }

    .spinner {
      width: 16px; height: 16px;
      border: 2px solid rgba(88,166,255,0.3);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      flex-shrink: 0;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    /* Section card */
    .section-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      animation: fadeIn 0.4s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .section-header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .section-badge {
      width: 26px; height: 26px;
      border-radius: 6px;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px;
      font-weight: 700;
    }

    .badge-blue   { background: rgba(88,166,255,0.15); color: var(--accent); }
    .badge-orange { background: rgba(244,162,97,0.15); color: var(--orange); }
    .badge-green  { background: rgba(63,185,80,0.15);  color: var(--green); }

    .section-header h3 { font-size: 14px; font-weight: 600; flex: 1; }

    .section-meta { font-size: 11px; color: var(--muted); }

    .section-body { padding: 16px 18px; }

    /* Image stack — each map occupies its own full-width row */
    .img-stack {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .img-card {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }

    .img-card-title {
      padding: 8px 14px;
      font-size: 11px;
      font-weight: 600;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    /* Images render at 100% container width, height is proportional — never clipped */
    .img-card img {
      width: 100%;
      height: auto;
      display: block;
    }

    /* Metrics grid */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }

    .metric {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
    }

    .metric-label {
      font-size: 10px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }

    .metric-value {
      font-size: 20px;
      font-weight: 700;
    }

    .metric-sub {
      font-size: 10px;
      color: var(--muted);
      margin-top: 2px;
    }

    .metric.red    .metric-value { color: var(--red); }
    .metric.green  .metric-value { color: var(--green); }
    .metric.orange .metric-value { color: var(--orange); }
    .metric.blue   .metric-value { color: var(--accent); }
    .metric.teal   .metric-value { color: var(--teal); }

    /* Battery table */
    .batt-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 10px;
    }

    .batt-table th {
      text-align: left;
      padding: 7px 10px;
      background: rgba(255,255,255,0.04);
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .batt-table td {
      padding: 6px 10px;
      border-top: 1px solid var(--border);
    }

    .batt-table tr:hover td { background: rgba(255,255,255,0.02); }

    .tag {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 10px;
      font-weight: 600;
    }

    .tag-green  { background: rgba(63,185,80,0.15);  color: var(--green); }
    .tag-red    { background: rgba(248,81,73,0.15);  color: var(--red); }
    .tag-orange { background: rgba(244,162,97,0.15); color: var(--orange); }
    .tag-blue   { background: rgba(88,166,255,0.15); color: var(--accent); }

    /* Case badge */
    .case-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 700;
    }

    .case-A { background: rgba(42,157,143,0.15); color: #2a9d8f; border: 1px solid rgba(42,157,143,0.3); }
    .case-B { background: rgba(244,162,97,0.15); color: var(--orange); border: 1px solid rgba(244,162,97,0.3); }
    .case-C { background: rgba(248,81,73,0.15);  color: var(--red); border: 1px solid rgba(248,81,73,0.3); }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* ── Modal overlay ── */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.72);
      z-index: 100;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      backdrop-filter: blur(3px);
      animation: fadeBg 0.2s ease;
    }

    @keyframes fadeBg { from { opacity: 0; } to { opacity: 1; } }

    .modal-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      width: 96vw;
      max-width: 1600px;
      height: 95vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 24px 64px rgba(0,0,0,0.6);
    }

    .modal-hdr {
      background: var(--sidebar);
      border-bottom: 1px solid var(--border);
      padding: 12px 18px;
      display: flex;
      align-items: center;
      gap: 14px;
      flex-shrink: 0;
    }

    .modal-hdr h2 {
      font-size: 14px;
      font-weight: 600;
      flex: 1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .modal-close {
      background: rgba(255,255,255,0.07);
      border: 1px solid var(--border);
      color: var(--muted);
      border-radius: 6px;
      width: 28px; height: 28px;
      cursor: pointer;
      font-size: 13px;
      display: flex; align-items: center; justify-content: center;
      transition: background 0.15s, color 0.15s;
      flex-shrink: 0;
    }
    .modal-close:hover { background: rgba(248,81,73,0.15); color: var(--red); border-color: var(--red); }

    .modal-body {
      flex: 1;
      overflow-y: auto;
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    /* empty / done banner inside modal */
    .modal-empty {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 10px;
      color: var(--muted);
      text-align: center;
    }
    .modal-empty .big-icon { font-size: 36px; opacity: 0.35; }
    .modal-empty p { font-size: 13px; }

    .done-banner {
      background: rgba(63,185,80,0.08);
      border: 1px solid rgba(63,185,80,0.25);
      border-radius: 8px;
      padding: 12px 18px;
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      color: var(--green);
      font-weight: 500;
    }
  </style>
</head>
<body>

<!-- ── Sidebar ── -->
<aside class="sidebar">
  <div class="sidebar-header">
    <h1><span class="icon">⚡</span> GridSFM</h1>
    <p>Congestion mitigation analysis</p>
  </div>
  <div class="search-box">
    <input type="text" id="search" placeholder="Filter samples…" oninput="filterSamples(this.value)">
  </div>
  <div class="sample-list" id="sampleList">
    {% for grp, items in groups %}
    <div class="sample-group" data-group="{{ grp }}">
      <div class="sample-group-header">{{ grp }}</div>
      {% for s in items %}
      <button class="sample-btn" data-name="{{ s }}" onclick="selectSample('{{ s }}')">{{ s.replace('.pyg.json', '') }}</button>
      {% endfor %}
    </div>
    {% endfor %}
  </div>
</aside>

<!-- ── Main (selector + run button) ── -->
<div class="main">
  <div class="topbar">
    <h2 id="topTitle">Select a sample to begin</h2>
    <button class="run-btn" id="runBtn" disabled onclick="runAnalysis()">
      <span id="runIcon">▶</span> Run Analysis
    </button>
  </div>
  <div class="content">
    <div class="empty-state">
      <div class="big-icon">🔌</div>
      <h3>No analysis running</h3>
      <p>Select a grid sample from the left panel and click <strong>Run Analysis</strong>. Results will open in a popup where you can scroll through all maps.</p>
    </div>
  </div>
</div>

<!-- ── Results modal ── -->
<div class="modal-overlay" id="modal" style="display:none" onclick="onOverlayClick(event)">
  <div class="modal-box">
    <div class="modal-hdr">
      <h2 id="modalTitle">Analysis</h2>
      <!-- Progress steps inside modal header -->
      <div class="steps">
        <div class="step" id="step1"><div class="dot">1</div>Baseline</div>
        <div class="step-line" id="line1"></div>
        <div class="step" id="step2"><div class="dot">2</div>Mitigation</div>
        <div class="step-line" id="line2"></div>
        <div class="step" id="step3"><div class="dot">3</div>Report</div>
      </div>
      <button class="modal-close" onclick="closeModal()" title="Close">✕</button>
    </div>
    <div class="modal-body" id="modalBody">
      <div class="modal-empty"><div class="big-icon">⏳</div><p>Starting analysis…</p></div>
    </div>
  </div>
</div>

<script>
  let selectedSample = null;
  let eventSource    = null;

  function filterSamples(q) {
    q = q.toLowerCase();
    document.querySelectorAll('.sample-btn').forEach(btn => {
      btn.style.display = btn.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
    document.querySelectorAll('.sample-group').forEach(grp => {
      grp.style.display = [...grp.querySelectorAll('.sample-btn')].some(b => b.style.display !== 'none') ? '' : 'none';
    });
  }

  function selectSample(name) {
    selectedSample = name;
    document.querySelectorAll('.sample-btn').forEach(b => b.classList.toggle('active', b.dataset.name === name));
    document.getElementById('topTitle').textContent = name.replace('.pyg.json', '');
    document.getElementById('runBtn').disabled = false;
  }

  /* ── Modal helpers ── */
  function openModal(title) {
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalBody').innerHTML =
      '<div class="modal-empty"><div class="big-icon">⏳</div><p>Starting analysis…</p></div>';
    resetSteps();
    document.getElementById('modal').style.display = 'flex';
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    if (eventSource) { eventSource.close(); eventSource = null; }
    document.getElementById('modal').style.display = 'none';
    document.body.style.overflow = '';
    document.getElementById('runBtn').disabled = false;
    document.getElementById('runIcon').textContent = '▶';
  }

  function onOverlayClick(e) {
    if (e.target === document.getElementById('modal')) closeModal();
  }

  function resetSteps() {
    ['step1','step2','step3'].forEach(id => document.getElementById(id).className = 'step');
    ['line1','line2'].forEach(id => document.getElementById(id).className = 'step-line');
  }

  function setStep(n, state) {
    document.getElementById('step' + n).className = 'step ' + state;
    if (n > 1) document.getElementById('line' + (n-1)).className = 'step-line ' + state;
  }

  /* ── Content helpers — all target #modalBody ── */
  function getBody() { return document.getElementById('modalBody'); }

  function scrollToEnd() {
    const mb = getBody();
    if (mb) mb.scrollTo({ top: mb.scrollHeight, behavior: 'smooth' });
  }

  function showStatus(msg) {
    const mb = getBody();
    // remove the initial empty placeholder
    mb.querySelector('.modal-empty')?.remove();
    let bar = document.getElementById('statusBar');
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'statusBar';
      bar.className = 'status-bar';
      mb.appendChild(bar);
    }
    bar.innerHTML = `<div class="spinner"></div><span>${msg}</span>`;
    scrollToEnd();
  }

  function removeStatus() {
    document.getElementById('statusBar')?.remove();
  }

  function appendCard(html) {
    const mb = getBody();
    mb.querySelector('.modal-empty')?.remove();
    const wrapper = document.createElement('div');
    wrapper.innerHTML = html.trim();
    const card = wrapper.firstElementChild;
    const bar = document.getElementById('statusBar');
    if (bar) {
      mb.replaceChild(card, bar);   // card appears exactly where the spinner was
    } else {
      mb.appendChild(card);
    }
    // give images a frame to render, then scroll bottom into view
    requestAnimationFrame(() => scrollToEnd());
  }

  /* ── Main run logic ── */
  function runAnalysis() {
    if (!selectedSample || eventSource) return;

    document.getElementById('runBtn').disabled = true;
    document.getElementById('runIcon').textContent = '⏳';
    openModal(selectedSample.replace('.pyg.json', ''));
    setStep(1, 'active');

    eventSource = new EventSource('/analyze?sample=' + encodeURIComponent(selectedSample));

    eventSource.onmessage = function(e) {
      const data = JSON.parse(e.data);

      if (data.type === 'status') { showStatus(data.msg); return; }

      if (data.type === 'scenario1') {
        removeStatus();
        setStep(1, 'done');
        setStep(2, 'active');

        const congLabel = data.congested_count > 0
          ? `<span class="tag tag-red">${data.congested_count} congested line${data.congested_count>1?'s':''}</span>`
          : `<span class="tag tag-green">No congestion</span>`;

        const zoomHtml = data.zoom_img
          ? `<div class="img-card"><div class="img-card-title">🔍 Zoom — worst congested edge</div>
             <img src="data:image/png;base64,${data.zoom_img}" alt="zoom"></div>`
          : '';

        appendCard(`
          <div class="section-card">
            <div class="section-header">
              <div class="section-badge badge-blue">1</div>
              <h3>Scenario 1 — Baseline Prediction</h3>${congLabel}
            </div>
            <div class="section-body">
              <div class="img-stack">
                <div class="img-card"><div class="img-card-title">🗺 Full network — thermal loading</div>
                  <img src="data:image/png;base64,${data.full_img}" alt="full network"></div>
                ${zoomHtml}
              </div>
            </div>
          </div>`);
        return;
      }

      if (data.type === 'scenario2') {
        removeStatus();
        setStep(2, 'done');
        setStep(3, 'active');

        const battRows = data.shutdown_info.map((b, i) =>
          `<tr><td>#${i+1}</td><td>${b.idx}</td><td>${b.pd_mw.toFixed(2)} MW</td>
           <td><span class="tag tag-orange">Stopped charging</span></td></tr>`
        ).join('');

        const zoomHtml = data.zoom_img
          ? `<div class="img-card"><div class="img-card-title">🔍 Post-mitigation zoom</div>
             <img src="data:image/png;base64,${data.zoom_img}" alt="zoom after"></div>`
          : '';

        appendCard(`
          <div class="section-card">
            <div class="section-header">
              <div class="section-badge badge-orange">2</div>
              <h3>Scenario 2 — Battery Shutdown &amp; Re-prediction</h3>
              <span class="section-meta">${data.batteries_shut} batter${data.batteries_shut!==1?'ies':'y'} curtailed</span>
            </div>
            <div class="section-body">
              <div class="img-stack">
                <div class="img-card"><div class="img-card-title">🗺 Post-mitigation network</div>
                  <img src="data:image/png;base64,${data.full_img}" alt="full after"></div>
                ${zoomHtml}
              </div>
              ${battRows ? `<table class="batt-table" style="margin-top:14px">
                <thead><tr><th>#</th><th>Load index</th><th>Charging power</th><th>Status</th></tr></thead>
                <tbody>${battRows}</tbody></table>` : ''}
            </div>
          </div>`);
        return;
      }

      if (data.type === 'report') {
        removeStatus();
        setStep(3, 'done');

        const a = data.analysis;
        const caseColors = { A: 'case-A', B: 'case-B', C: 'case-C' };
        const caseDesc   = {
          A: 'Overload fully cleared by battery action',
          B: 'Partial improvement — peaker still needed for remainder',
          C: 'Battery had no measurable effect'
        };
        const viable = a.economically_viable
          ? `<span class="tag tag-green">✓ Economically viable</span>`
          : `<span class="tag tag-red">✗ Not viable</span>`;

        appendCard(`
          <div class="section-card">
            <div class="section-header">
              <div class="section-badge badge-green">3</div>
              <h3>Mitigation Analysis and Report</h3>${viable}
            </div>
            <div class="section-body">
              <div class="metrics-grid">
                <div class="metric ${a.s_before_pu > a.rate_a_pu ? 'red' : 'green'}">
                  <div class="metric-label">Loading before</div>
                  <div class="metric-value">${a.s_before_pu.toFixed(4)}</div>
                  <div class="metric-sub">p.u. (limit ${a.rate_a_pu.toFixed(4)})</div>
                </div>
                <div class="metric ${a.s_after_pu > a.rate_a_pu ? 'red' : 'green'}">
                  <div class="metric-label">Loading after</div>
                  <div class="metric-value">${a.s_after_pu.toFixed(4)}</div>
                  <div class="metric-sub">p.u. (${a.overload_cleared_pct.toFixed(1)}% cleared)</div>
                </div>
                <div class="metric blue">
                  <div class="metric-label">Δ S (improvement)</div>
                  <div class="metric-value">${a.delta_s_pu.toFixed(4)}</div>
                  <div class="metric-sub">p.u.</div>
                </div>
                <div class="metric orange">
                  <div class="metric-label">Batteries curtailed</div>
                  <div class="metric-value">${a.batteries_curtailed}</div>
                  <div class="metric-sub">${a.min_curtailment_mw.toFixed(1)} MW min subset</div>
                </div>
                <div class="metric red">
                  <div class="metric-label">Redispatch cost</div>
                  <div class="metric-value">$${a.redispatch_cost_usd.toLocaleString()}</div>
                  <div class="metric-sub">per dispatch hour</div>
                </div>
                <div class="metric teal">
                  <div class="metric-label">Battery compensation</div>
                  <div class="metric-value">$${a.battery_comp_usd.toLocaleString()}</div>
                  <div class="metric-sub">Savings: $${a.savings_usd.toLocaleString()}</div>
                </div>
              </div>
              <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                <span class="case-badge ${caseColors[a.case]}">Case ${a.case}</span>
                <span style="font-size:13px;color:var(--muted)">${caseDesc[a.case]}</span>
                <span style="margin-left:auto" class="tag ${a.economically_viable ? 'tag-green' : 'tag-red'}">${a.action}</span>
              </div>
            </div>
          </div>`);
        return;
      }

      if (data.type === 'no_congestion') {
        removeStatus();
        setStep(1, 'done');
        appendCard(`
          <div class="section-card">
            <div class="section-header">
              <div class="section-badge badge-green">✓</div>
              <h3>No congestion detected</h3>
            </div>
            <div class="section-body">
              <p style="color:var(--muted);font-size:13px">All lines operate within thermal limits. No mitigation needed.</p>
            </div>
          </div>`);
        return;
      }

      if (data.type === 'done') {
        eventSource.close();
        eventSource = null;
        document.getElementById('runBtn').disabled = false;
        document.getElementById('runIcon').textContent = '▶';
        removeStatus();
        // append a "done" banner at the bottom so the user knows it's finished
        const mb = getBody();
        const banner = document.createElement('div');
        banner.className = 'done-banner';
        banner.innerHTML = '✓ &nbsp;Analysis complete — scroll up to review all results';
        mb.appendChild(banner);
        scrollToEnd();
      }
    };

    eventSource.onerror = function() {
      removeStatus();
      eventSource?.close();
      eventSource = null;
      document.getElementById('runBtn').disabled = false;
      document.getElementById('runIcon').textContent = '▶';
      appendCard(`
        <div class="section-card" style="border-color:var(--red)">
          <div class="section-header">
            <div class="section-badge" style="background:rgba(248,81,73,0.15);color:var(--red)">!</div>
            <h3 style="color:var(--red)">Analysis error</h3>
          </div>
          <div class="section-body">
            <p style="font-size:13px;color:var(--muted)">An error occurred. Check the server console for details.</p>
          </div>
        </div>`);
    };
  }

  // close modal on Escape key
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
</script>
</body>
</html>
"""


@app.route("/")
def index():
    samples = sorted(f for f in os.listdir(SAMPLES_DIR) if f.endswith(".pyg.json"))
    msr   = [s for s in samples if s.startswith("msr_")]
    cases = [s for s in samples if not s.startswith("msr_")]
    groups = [("MSR (state grids)", msr), ("Case studies", cases)]
    return render_template_string(HTML, groups=groups)


@app.route("/analyze")
def analyze():
    sample = request.args.get("sample", "").strip()
    if not sample:
        return Response("Missing sample", status=400)
    graph_file = os.path.join(SAMPLES_DIR, sample)
    if not os.path.exists(graph_file):
        return Response("Sample not found", status=404)

    def generate():
        try:
            mdl = get_model()

            # ── Step 1: baseline prediction ────────────────────────────────────
            yield f"data: {json.dumps({'type': 'status', 'msg': 'Loading model and running baseline prediction…'})}\n\n"
            out_before     = predict(mdl, graph_file)
            loading_before = compute_line_loading(out_before)

            yield f"data: {json.dumps({'type': 'status', 'msg': 'Rendering network maps…'})}\n\n"
            b64_full, b64_zoom, worst, congested_sorted = make_loading_figures(loading_before, graph_file)

            yield f"data: {json.dumps({'type': 'scenario1', 'full_img': b64_full, 'zoom_img': b64_zoom, 'congested_count': len(congested_sorted), 'worst': list(worst) if worst else None})}\n\n"

            if worst is None:
                yield f"data: {json.dumps({'type': 'no_congestion'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # ── Step 2: battery shutdown + re-predict ──────────────────────────
            yield f"data: {json.dumps({'type': 'status', 'msg': 'Identifying and curtailing charging batteries…'})}\n\n"
            d_modified, shutdown_details = shutdown_batteries(graph_file, worst, n_hops=2, max_batteries=3)

            tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.pyg.json")
            with open(tmp_path, "w") as f:
                json.dump(d_modified, f)

            try:
                yield f"data: {json.dumps({'type': 'status', 'msg': 'Running post-mitigation prediction…'})}\n\n"
                out_after     = predict(mdl, tmp_path)
                loading_after = compute_line_loading(out_after)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            yield f"data: {json.dumps({'type': 'status', 'msg': 'Rendering post-mitigation maps…'})}\n\n"
            b64_full_a, b64_zoom_a, _, _ = make_loading_figures(loading_after, graph_file)

            shutdown_info = [{"idx": idx, "pd_mw": round(pd * CONFIG["base_mva"], 2)}
                             for idx, pd, _ in shutdown_details]
            yield f"data: {json.dumps({'type': 'scenario2', 'full_img': b64_full_a, 'zoom_img': b64_zoom_a, 'batteries_shut': len(shutdown_details), 'shutdown_info': shutdown_info})}\n\n"

            # ── Step 3: analysis & report ──────────────────────────────────────
            yield f"data: {json.dumps({'type': 'status', 'msg': 'Computing mitigation analysis…'})}\n\n"
            analysis = analyze_mitigation(loading_before, loading_after, worst, shutdown_details)

            yield f"data: {json.dumps({'type': 'report', 'analysis': analysis})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'msg': str(exc)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    print("Starting Grid Congestion Mitigation Dashboard on http://localhost:5050")
    app.run(debug=False, threaded=True, port=5050)
