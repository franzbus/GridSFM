#%%
import json
import math
import datetime
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── Congestion mitigation configuration ───────────────────────────────────────
CONFIG = {
    "emergency_peaker_cost_per_mwh": 150.0,   # $/MWh — cost of redispatch via peaker
    "battery_curtailment_cost_per_mwh": 20.0, # $/MWh — opportunity cost for battery operator
    "base_mva": 100.0,                         # system base MVA for per-unit conversion
    "dispatch_duration_hours": 1.0,            # assumed dispatch window length
}


def compute_line_loading(out):
    """
    Takes predict() output `out`, computes apparent power S in both directions
    per flow edge, and returns the per-edge maximum.

    Returns a dict with:
      S_max          : [n_flow_edges] tensor, max(Sij, Sji) per edge (per-unit)
      Sij, Sji       : [n_flow_edges] tensors
      flow_edge_types, flow_edge_counts : passed through for downstream mapping
    """
    Pij = torch.as_tensor(out["Pij"], dtype=torch.float64)
    Qij = torch.as_tensor(out["Qij"], dtype=torch.float64)
    Pji = torch.as_tensor(out["Pji"], dtype=torch.float64)
    Qji = torch.as_tensor(out["Qji"], dtype=torch.float64)

    Sij = torch.sqrt(Pij**2 + Qij**2)
    Sji = torch.sqrt(Pji**2 + Qji**2)
    S_max = torch.maximum(Sij, Sji)

    return {
        "S_max": S_max,
        "Sij": Sij,
        "Sji": Sji,
        "flow_edge_types": out.get("flow_edge_types", []),
        "flow_edge_counts": out.get("flow_edge_counts", []),
    }


def plot_line_loading(loading, graph_filename, save_path=None):
    """
    Takes the output of compute_line_loading() and the original PyG-JSON graph
    file, draws the network coloring nodes by type and lines green/red by
    whether S_max <= rate_a (green) or > rate_a (red).
    """
    with open(graph_filename) as f:
        g = json.load(f)

    S_max  = loading["S_max"]
    types  = loading["flow_edge_types"]
    counts = loading["flow_edge_counts"]

    nodes  = g["grid"]["nodes"]
    edges  = g["grid"]["edges"]
    schema = g["metadata"].get("feature_schema", {})

    # rate_a column index from feature schema, with known defaults
    def _rate_col(fam):
        feats = schema.get(fam, [])
        return feats.index("rate_a_mva") if "rate_a_mva" in feats else (6 if fam == "ac_line" else 4)

    # classify each bus by what's attached to it (generator > load > plain)
    gen_buses  = set(edges["generator_link"].get("receivers", []))
    load_buses = set(edges["load_link"].get("receivers", []))
    def _bus_color(i):
        if i in gen_buses and i in load_buses: return "#f4a261"   # both  → orange
        if i in gen_buses:                     return "#31a354"   # gen   → green
        if i in load_buses:                    return "#e6550d"   # load  → red-orange
        return "#9ecae1"                                          # plain → light blue

    # build graph with BUS nodes only; gen/load encoded as node color
    G = nx.Graph()
    n_buses = len(nodes["bus"])
    for i in range(n_buses):
        G.add_node(i, kind="bus")

    # branch edges colored by thermal loading; track congested ones
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

    # ── print congested lines ─────────────────────────────────────────────────
    if congested:
        congested_sorted = sorted(congested, key=lambda x: x[7], reverse=True)
        worst = congested_sorted[0]
        print(f"\nCongested lines ({len(congested)} total, sorted by S/rate_a):")
        print(f"  {'idx':>6}  {'type':<12}  {'from':>5}→{'to':<5}  {'S_max':>8}  {'rate_a':>8}  {'ratio':>7}")
        for idx, fam, k, fr, to, s, ra, ratio in congested_sorted:
            marker = "  ◄ WORST" if idx == worst[0] else ""
            print(f"  {idx:>6}  {fam:<12}  {fr:>5}→{to:<5}  {s:>8.4f}  {ra:>8.4f}  {ratio:>7.3f}{marker}")
    else:
        print("\nNo congested lines (all S_max ≤ rate_a).")
        congested_sorted, worst = [], None

    # ── layout (computed once, reused for both plots) ─────────────────────────
    pos = nx.spring_layout(G, seed=0)
    node_colors = [_bus_color(n) for n in G.nodes]

    def _draw_graph(ax, highlight_worst=True):
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=20, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=branch_edges,
                               edge_color=edge_colors, width=1.2, ax=ax)
        if highlight_worst and worst is not None:
            wi = worst[0]
            nx.draw_networkx_edges(G, pos, edgelist=[branch_edges[wi]],
                                   edge_color="gold", width=6.0, ax=ax)
        ax.axis("off")

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

    # ── plot 1: full network ───────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(22, 16))
    _draw_graph(ax1)
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax1.set_title("Full network — line thermal loading", fontsize=13)
    fig1.tight_layout()
    if save_path:
        fig1.savefig(save_path, dpi=150, bbox_inches="tight")

    # ── plot 2: zoom on worst congested edge ──────────────────────────────────
    if worst is not None:
        w_fr, w_to = worst[3], worst[4]
        x0, y0 = pos[w_fr]
        x1, y1 = pos[w_to]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pad = 0.15   # neighbourhood radius in layout coordinates

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        _draw_graph(ax2, highlight_worst=True)
        ax2.set_xlim(cx - pad, cx + pad)
        ax2.set_ylim(cy - pad, cy + pad)
        ax2.legend(handles=legend_handles, loc="upper right", fontsize=9)
        ax2.set_title(
            f"Zoom — worst congested edge  (idx {worst[0]}, {worst[1]}, "
            f"bus {w_fr}→{w_to}, S/rate_a = {worst[7]:.3f})", fontsize=11)
        fig2.tight_layout()
        if save_path:
            zoom_path = save_path.replace(".png", "_zoom.png")
            fig2.savefig(zoom_path, dpi=150, bbox_inches="tight")

    plt.show()
    return G, worst, congested_sorted

# %%
import copy

def shutdown_neighborhood_batteries(graph_filename, worst, output_path=None, n_hops=10, max_batteries=None):
    """
    Curtail CHARGING batteries (pd_pu > 0) within n_hops of the worst
    congested line endpoints.  Discharging batteries (pd_pu <= 0) are left
    untouched — stopping them would increase local demand and worsen congestion.

    If max_batteries is set, only that many batteries are curtailed, chosen by
    largest pd_pu first (greatest impact per unit).

    Returns
    -------
    (dict, list)
        modified graph (deep copy) and shutdown_details:
        [(load_idx, orig_pd_pu, orig_qd_pu), ...] for every curtailed load.
    """
    with open(graph_filename) as f:
        d = json.load(f)
    d = copy.deepcopy(d)

    nodes  = d["grid"]["nodes"]
    edges  = d["grid"]["edges"]
    meta   = d["metadata"]

    _, _, _, from_bus, to_bus, *_ = worst

    # ── build adjacency ───────────────────────────────────────────────────────
    n_buses = len(nodes["bus"])
    adj = {i: set() for i in range(n_buses)}
    for fam in ("ac_line", "transformer"):
        for s, r in zip(edges[fam]["senders"], edges[fam]["receivers"]):
            adj[s].add(r)
            adj[r].add(s)

    def _bfs(start):
        visited, frontier = {start}, {start}
        for _ in range(n_hops):
            frontier = {nb for node in frontier for nb in adj[node]} - visited
            visited |= frontier
        return visited

    neighbourhood = _bfs(from_bus) | _bfs(to_bus)

    # ── locate batteries in neighbourhood ─────────────────────────────────────
    ll_s = edges["load_link"]["senders"]
    ll_r = edges["load_link"]["receivers"]
    load_to_bus = {ll_s[i]: ll_r[i] for i in range(len(ll_s))}

    schema     = meta.get("feature_schema", {})
    load_feats = schema.get("load", ["pd_pu", "qd_pu"])
    idx_pd = load_feats.index("pd_pu") if "pd_pu" in load_feats else 0
    idx_qd = load_feats.index("qd_pu") if "qd_pu" in load_feats else 1

    # collect all eligible charging batteries in the neighbourhood
    candidates = []
    skipped_discharging = 0
    for load_idx, bus_row in sorted(load_to_bus.items()):
        if bus_row not in neighbourhood:
            continue
        feat    = nodes["load"][load_idx]
        orig_pd = feat[idx_pd]
        orig_qd = feat[idx_qd]
        if orig_pd <= 0:
            skipped_discharging += 1
            continue
        candidates.append((load_idx, orig_pd, orig_qd))

    # pick the top-N by charging power (largest impact first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    if max_batteries is not None:
        candidates = candidates[:max_batteries]

    shutdown_details = []
    for load_idx, orig_pd, orig_qd in candidates:
        nodes["load"][load_idx][idx_pd] = 0.0
        nodes["load"][load_idx][idx_qd] = 0.0
        shutdown_details.append((load_idx, orig_pd, orig_qd))

    cap_str = f"  (capped at {max_batteries})" if max_batteries is not None else ""
    print(f"\nshutdown_neighborhood_batteries  ({n_hops}-hop neighbourhood)")
    print(f"  Worst line       : idx {worst[0]}  {worst[1]}  "
          f"bus {from_bus}→{to_bus}  S/rate_a = {worst[7]:.3f}")
    print(f"  Neighbourhood    : {len(neighbourhood)} buses")
    print(f"  Charging stopped : {len(shutdown_details)}{cap_str}  "
          f"(skipped {skipped_discharging} discharging/idle)")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"  Saved → {output_path}")

    return d, shutdown_details


def analyze_mitigation(loading_before, loading_after, worst, shutdown_details,
                       cfg=None):
    """
    Compare line flows before and after battery curtailment on the worst edge.
    Determines Case A/B/C, computes Impact Factor, minimum-curtailment subset,
    and cost comparison.

    Case A — full overload cleared by battery action alone
    Case B — partial improvement; peaker still needed for remainder
    Case C — battery had no effect; full peaker dispatch required
    """
    if cfg is None:
        cfg = CONFIG

    global_idx = worst[0]
    s_before   = float(worst[5])
    rate_a     = float(worst[6])
    s_after    = float(loading_after["S_max"][global_idx])

    overload     = max(0.0, s_before - rate_a)
    delta_s      = s_before - s_after                  # positive = improvement
    total_delta_p = sum(pd for _, pd, _ in shutdown_details)
    impact_factor = delta_s / total_delta_p if total_delta_p > 1e-9 else 0.0

    if s_after < rate_a:
        case = "A"
    elif delta_s > 1e-4:
        case = "B"
    else:
        case = "C"

    # ── minimum curtailment subset (Case A: we may have over-curtailed) ────────
    # Estimate: overload / impact_factor = minimum ΔP needed (pu).
    # Greedily pick largest-charging batteries first until sum covers that ΔP.
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

    redispatch_cost  = overload * base_mva * cfg["emergency_peaker_cost_per_mwh"] * duration
    battery_cost     = min_curtailment_mw  * cfg["battery_curtailment_cost_per_mwh"] * duration
    savings          = redispatch_cost - battery_cost

    return {
        "case":                  case,
        "s_before_pu":           round(s_before, 6),
        "s_after_pu":            round(s_after, 6),
        "rate_a_pu":             round(rate_a, 6),
        "delta_s_pu":            round(delta_s, 6),
        "overload_cleared_pct":  round(delta_s / overload * 100, 1) if overload > 1e-9 else 100.0,
        "impact_factor":         round(impact_factor, 6),
        "batteries_curtailed":   len(shutdown_details),
        "total_curtailment_mw":  round(total_delta_p * base_mva, 2),
        "min_curtailment_mw":    round(min_curtailment_mw, 2),
        "min_battery_subset":    [idx for idx, _, _ in min_subset],
        "redispatch_cost_usd":   round(redispatch_cost, 2),
        "battery_comp_usd":      round(battery_cost, 2),
        "savings_usd":           round(savings, 2),
        "action":                "BATTERY_CURTAILMENT" if case in ("A", "B") else "PEAKER_DISPATCH",
        "economically_viable":   savings > 0 and case in ("A", "B"),
    }


def generate_report(worst, analysis, graph_filename, output_path=None):
    """Emit a structured JSON report and print a human-readable summary."""
    a = analysis
    report = {
        "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
        "grid_file":  graph_filename,
        "isolated_line": {
            "global_idx": worst[0],
            "family":     worst[1],
            "from_bus":   worst[3],
            "to_bus":     worst[4],
        },
        "status":              "CONGESTED" if a["s_before_pu"] > a["rate_a_pu"] else "SAFE",
        "baseline_loading_pu": a["s_before_pu"],
        "rate_a_pu":           a["rate_a_pu"],
        "mitigation": {
            "case":                   a["case"],
            "action":                 a["action"],
            "economically_viable":    a["economically_viable"],
            "batteries_curtailed":    a["batteries_curtailed"],
            "min_battery_subset":     a["min_battery_subset"],
            "min_curtailment_mw":     a["min_curtailment_mw"],
            "predicted_loading_after":a["s_after_pu"],
            "overload_cleared_pct":   a["overload_cleared_pct"],
            "impact_factor":          a["impact_factor"],
        },
        "economics": {
            "redispatch_cost_usd":      a["redispatch_cost_usd"],
            "battery_compensation_usd": a["battery_comp_usd"],
            "estimated_savings_usd":    a["savings_usd"],
        },
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report → {output_path}")

    w = "─" * 62
    viable = "✓ viable" if a["economically_viable"] else "✗ not viable"
    print(f"\n{w}")
    print(f"  CONGESTION MITIGATION REPORT")
    print(f"  Line     : {worst[1]} idx {worst[0]}  bus {worst[3]}→{worst[4]}")
    print(f"  Loading  : {a['s_before_pu']:.4f} pu  →  {a['s_after_pu']:.4f} pu"
          f"  (limit {a['rate_a_pu']:.4f})")
    print(f"  Case     : {a['case']}  →  {a['action']}")
    print(f"  ΔS       : {a['delta_s_pu']:.4f} pu  "
          f"({a['overload_cleared_pct']:.1f}% of overload cleared)")
    print(f"  Min curtailment : {a['min_curtailment_mw']:.1f} MW  "
          f"({len(a['min_battery_subset'])} batteries: {a['min_battery_subset']})")
    print(f"  Redispatch cost : ${a['redispatch_cost_usd']:>10,.0f}")
    print(f"  Battery comp.   : ${a['battery_comp_usd']:>10,.0f}")
    print(f"  Savings         : ${a['savings_usd']:>10,.0f}  {viable}")
    print(f"{w}\n")

    return report


def plot_mitigation_report(worst, analysis, graph_filename, save_path=None):
    """
    Non-technical summary dashboard: status card, loading gauge, cost comparison,
    and a labelled neighbourhood map of the congested area.
    """
    with open(graph_filename) as f:
        g = json.load(f)
    nodes = g["grid"]["nodes"]
    edges = g["grid"]["edges"]

    a          = analysis
    rate_a     = a["rate_a_pu"]
    pct_before = a["s_before_pu"] / rate_a * 100 if rate_a > 0 else 0
    pct_after  = a["s_after_pu"]  / rate_a * 100 if rate_a > 0 else 0

    case_color  = {"A": "#2a9d8f", "B": "#f4a261", "C": "#e63946"}[a["case"]]
    case_label  = {"A": "Overload\ncleared ✓", "B": "Partially\ncleared ~", "C": "No\neffect ✗"}[a["case"]]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#f0f2f5")

    gs_outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.4],
                                 hspace=0.38, top=0.91, bottom=0.05,
                                 left=0.04, right=0.97)
    gs_top  = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0], wspace=0.28)
    ax_s    = fig.add_subplot(gs_top[0])   # status
    ax_l    = fig.add_subplot(gs_top[1])   # loading
    ax_e    = fig.add_subplot(gs_top[2])   # economics
    ax_net  = fig.add_subplot(gs_outer[1]) # network

    grid_name = graph_filename.split("/")[-1].replace(".pyg.json", "")
    fig.suptitle(
        f"Congestion Mitigation Report  ·  Grid: {grid_name}  ·  "
        f"{worst[1].replace('_',' ').title()} #{worst[0]}  "
        f"(bus {worst[3]} → bus {worst[4]})",
        fontsize=13, fontweight="bold", color="#1a1a2e", y=0.97,
    )

    # ── Card helper ───────────────────────────────────────────────────────────
    for ax in (ax_s, ax_l, ax_e):
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ── Panel 1: Status ───────────────────────────────────────────────────────
    ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.set_xticks([]); ax_s.set_yticks([])

    ax_s.add_patch(plt.Circle((0.22, 0.70), 0.14, color="#e63946", zorder=3, transform=ax_s.transData))
    ax_s.text(0.22, 0.70, "!", ha="center", va="center",
              fontsize=22, fontweight="bold", color="white", zorder=4)
    ax_s.text(0.22, 0.50, "CONGESTED", ha="center", fontsize=8,
              color="#e63946", fontweight="bold")

    ax_s.annotate("", xy=(0.60, 0.70), xytext=(0.40, 0.70),
                  arrowprops=dict(arrowstyle="-|>", color="#888", lw=2.0,
                                  mutation_scale=18))

    ax_s.add_patch(plt.Circle((0.78, 0.70), 0.14, color=case_color, zorder=3, transform=ax_s.transData))
    ax_s.text(0.78, 0.70, case_label.split("\n")[1][0],
              ha="center", va="center", fontsize=22, fontweight="bold",
              color="white", zorder=4)
    ax_s.text(0.78, 0.50, case_label, ha="center", fontsize=8,
              color=case_color, fontweight="bold", multialignment="center")

    action = "BATTERY CURTAILMENT" if a["economically_viable"] else "EMERGENCY PEAKER"
    bg     = "#2a9d8f22" if a["economically_viable"] else "#e6394622"
    ec     = "#2a9d8f"   if a["economically_viable"] else "#e63946"
    ax_s.text(0.50, 0.24, f"Action: {action}", ha="center", va="center",
              fontsize=8.5, fontweight="bold", color=ec,
              bbox=dict(boxstyle="round,pad=0.4", facecolor=bg, edgecolor=ec, lw=1.2))
    ax_s.set_title("Grid Status", fontweight="bold", fontsize=11, pad=8)

    # ── Panel 2: Loading gauge ────────────────────────────────────────────────
    ax_l.set_xlim(-10, 240); ax_l.set_ylim(0, 1); ax_l.set_xticks([]); ax_l.set_yticks([])

    def _gauge_bar(ax, y, pct, label, color):
        # background track
        ax.barh(y, 200, height=0.14, color="#e0e0e0", left=0, zorder=1)
        ax.barh(y, min(pct, 200), height=0.14, color=color, left=0, zorder=2, alpha=0.88)
        ax.text(-8, y, label, va="center", ha="right", fontsize=9, color="#444")
        ax.text(min(pct, 200) + 4, y, f"{pct:.0f}%",
                va="center", fontsize=10, fontweight="bold", color=color)

    _gauge_bar(ax_l, 0.72, pct_before, "Before", "#e63946" if pct_before > 100 else "#f4a261")
    _gauge_bar(ax_l, 0.38, pct_after,  "After",  "#e63946" if pct_after  > 100 else "#2a9d8f")

    # limit line
    ax_l.axvline(100, ymin=0.18, ymax=0.95, color="#333", lw=1.8, linestyle="--", zorder=5)
    ax_l.text(100, 0.96, "Thermal\nlimit", ha="center", va="top",
              fontsize=7.5, color="#333", multialignment="center")

    ax_l.set_title("Line Loading  (% of thermal limit)", fontweight="bold", fontsize=11, pad=8)

    # ── Panel 3: Economics ────────────────────────────────────────────────────
    labels = ["Redispatch\n(peaker)", "Battery\ncomp."]
    vals   = [a["redispatch_cost_usd"], a["battery_comp_usd"]]
    colors = ["#e63946", "#2a9d8f"]
    bars   = ax_e.bar(labels, vals, color=colors, alpha=0.85, width=0.45,
                      edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax_e.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + max(vals) * 0.03,
                  f"${v:,.0f}", ha="center", fontsize=10, fontweight="bold")

    if a["savings_usd"] > 0:
        ax_e.text(0.50, 0.97, f"Savings: ${a['savings_usd']:,.0f}",
                  ha="center", va="top", transform=ax_e.transAxes,
                  fontsize=11, fontweight="bold", color="#2a9d8f",
                  bbox=dict(boxstyle="round,pad=0.35", facecolor="#2a9d8f18",
                            edgecolor="#2a9d8f", lw=1.2))
    ax_e.set_ylabel("Cost per dispatch hour (USD)", fontsize=8.5)
    ax_e.tick_params(labelsize=9); ax_e.spines[["top","right"]].set_visible(False)
    ax_e.set_title("Cost Comparison", fontweight="bold", fontsize=11, pad=8)

    # ── Panel 4: Neighbourhood network ───────────────────────────────────────
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
    shut_buses  = {load_to_bus[li] for li in a.get("min_battery_subset", [])
                   if li in load_to_bus}

    G_sub = nx.Graph()
    G_sub.add_nodes_from(viz_buses)
    for fam in ("ac_line", "transformer"):
        for s, r in zip(edges[fam]["senders"], edges[fam]["receivers"]):
            if s in viz_buses and r in viz_buses:
                G_sub.add_edge(s, r)

    pos_sub  = nx.spring_layout(G_sub, seed=42)
    edgelist = list(G_sub.edges())
    worst_set = {from_bus, to_bus}

    nc = []
    for n in G_sub.nodes():
        if n in worst_set:    nc.append("#e63946")   # congested endpoints
        elif n in shut_buses: nc.append("#f4a261")   # curtailed batteries
        else:                 nc.append("#aec6cf")   # other buses

    ec_sub = ["#e63946" if set(e) == worst_set else "#bbbbbb" for e in edgelist]
    ew_sub = [4.5       if set(e) == worst_set else 1.0       for e in edgelist]

    nx.draw_networkx_nodes(G_sub, pos_sub, node_color=nc, node_size=160, ax=ax_net)
    nx.draw_networkx_edges(G_sub, pos_sub, edgelist=edgelist,
                           edge_color=ec_sub, width=ew_sub, ax=ax_net)
    nx.draw_networkx_labels(G_sub, pos_sub, font_size=6, font_color="white", ax=ax_net)
    ax_net.axis("off")

    legend_items = [
        mpatches.Patch(color="#e63946",
                       label=f"Congested line endpoints (bus {from_bus} & {to_bus})"),
        mpatches.Patch(color="#f4a261",
                       label=f"Batteries stopped charging  "
                             f"({len(a.get('min_battery_subset', []))} units · "
                             f"{a['min_curtailment_mw']:.1f} MW curtailed)"),
        mpatches.Patch(color="#aec6cf", label="Other buses in affected area"),
    ]
    ax_net.legend(handles=legend_items, loc="lower right", fontsize=9,
                  framealpha=0.92, edgecolor="#ccc")
    ax_net.set_title(
        f"Affected area  ·  3-hop neighbourhood around congested line  "
        f"·  {len(viz_buses)} buses shown",
        fontweight="bold", fontsize=10, pad=8,
    )

    out = save_path or "mitigation_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"Dashboard → {out}")


import itertools

def find_vicinity_generators(graph_data, worst, n_hops=10):
    """
    Return (gens, idx_pg) where gens is a list of
    (gen_idx, bus_idx, hop_distance, pg_pu) for all active generators
    within n_hops of the worst congested line endpoints, sorted closest-first.
    """
    nodes = graph_data["grid"]["nodes"]
    edges = graph_data["grid"]["edges"]
    meta  = graph_data["metadata"]

    _, _, _, from_bus, to_bus, *_ = worst

    n_buses = len(nodes["bus"])
    adj = {i: set() for i in range(n_buses)}
    for fam in ("ac_line", "transformer"):
        for s, r in zip(edges[fam]["senders"], edges[fam]["receivers"]):
            adj[s].add(r); adj[r].add(s)

    def _bfs_dist(start):
        dist, frontier = {start: 0}, {start}
        for hop in range(1, n_hops + 1):
            nxt = {nb for nd in frontier for nb in adj[nd]} - set(dist)
            for nd in nxt: dist[nd] = hop
            frontier = nxt
        return dist

    d_from = _bfs_dist(from_bus)
    d_to   = _bfs_dist(to_bus)

    schema    = meta.get("feature_schema", {})
    gen_feats = schema.get("generator", [])
    idx_pg    = gen_feats.index("pg_pu") if "pg_pu" in gen_feats else 0

    gl_s = edges["generator_link"]["senders"]
    gl_r = edges["generator_link"]["receivers"]

    gens = []
    for i in range(len(gl_s)):
        gen_idx  = gl_s[i]
        bus_idx  = gl_r[i]
        min_dist = min(d_from.get(bus_idx, 999), d_to.get(bus_idx, 999))
        if min_dist > n_hops:
            continue
        pg = nodes["generator"][gen_idx][idx_pg]
        if pg > 0:
            gens.append((gen_idx, bus_idx, min_dist, pg))

    gens.sort(key=lambda x: x[2])
    return gens, idx_pg


def _predict_feas(model, graph_data, tmp_path="/tmp/_gridsfm_opt.pyg.json"):
    """Serialize graph_data, call predict(), return scalar feas value."""
    with open(tmp_path, "w") as f:
        json.dump(graph_data, f)
    out = predict(model, tmp_path)
    raw = out.get("feas", 0.0)
    return raw.item() if hasattr(raw, "item") else float(raw)


def run_optimization_scenarios(graph_filename, worst, shutdown_details, model,
                                gen_n_hops=10, base_mva=100.0):
    """
    Iterate all 2^N − 1 non-empty subsets of the selected batteries,
    and for each subset step down nearby generator infeed in 5%-of-total-
    battery-power increments (closest generator first).  At each step feas
    is recorded.  The optimal step is the one just before feas first declines.

    Returns a list of result dicts, one per scenario.
    """
    with open(graph_filename) as f:
        base_graph = json.load(f)

    meta       = base_graph["metadata"]
    schema     = meta.get("feature_schema", {})
    load_feats = schema.get("load", ["pd_pu", "qd_pu"])
    idx_pd = load_feats.index("pd_pu") if "pd_pu" in load_feats else 0
    idx_qd = load_feats.index("qd_pu") if "qd_pu" in load_feats else 1

    total_batt_pw = sum(pd for _, pd, _ in shutdown_details)
    step_pu       = 0.05 * total_batt_pw

    gens, idx_pg = find_vicinity_generators(base_graph, worst, n_hops=gen_n_hops)
    if not gens:
        print("No active generators found in vicinity — aborting optimization.")
        return []

    total_gen_pw = sum(pg for *_, pg in gens)
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"Optimization setup")
    print(f"  Total battery power : {total_batt_pw * base_mva:.2f} MW")
    print(f"  Step size (5 %)     : {step_pu * base_mva:.2f} MW")
    print(f"  Generators found    : {len(gens)}  ({total_gen_pw * base_mva:.2f} MW total)")
    for gid, bid, dist, pg in gens:
        print(f"    Gen {gid:>4}  bus {bid:>4}  {dist} hops  {pg * base_mva:.2f} MW")
    print(sep)

    results = []
    n_batt  = len(shutdown_details)

    for r in range(1, n_batt + 1):
        for combo in itertools.combinations(range(n_batt), r):
            label       = "Bat " + "+".join(str(i + 1) for i in combo)
            batt_subset = [shutdown_details[i] for i in combo]

            # apply battery shutdowns to a fresh copy of the base graph
            d_batt = copy.deepcopy(base_graph)
            for load_idx, _, _ in batt_subset:
                d_batt["grid"]["nodes"]["load"][load_idx][idx_pd] = 0.0
                d_batt["grid"]["nodes"]["load"][load_idx][idx_qd] = 0.0

            # generator power state for this scenario
            gen_pw    = {gid: pg for gid, _, _, pg in gens}
            gen_order = [gid for gid, *_ in gens]
            ptr       = 0
            total_red = 0.0

            feas_series = []
            red_series  = []
            d_iter      = copy.deepcopy(d_batt)

            # step 0: batteries off, no generator change
            feas_series.append(_predict_feas(model, d_iter))
            red_series.append(0.0)

            while ptr < len(gen_order):
                gid = gen_order[ptr]
                cur = gen_pw[gid]
                if cur <= 0:
                    ptr += 1
                    continue

                new_pw = max(0.0, cur - step_pu)
                gen_pw[gid] = new_pw
                d_iter["grid"]["nodes"]["generator"][gid][idx_pg] = new_pw
                total_red += cur - new_pw

                if new_pw <= 0:
                    ptr += 1

                feas_val = _predict_feas(model, d_iter)
                feas_series.append(feas_val)
                red_series.append(total_red)

                # stop as soon as feas declines for the first time
                if feas_series[-1] < feas_series[-2]:
                    break

            # peak = index just before first decline
            peak_idx = len(feas_series) - 1
            for i in range(1, len(feas_series)):
                if feas_series[i] < feas_series[i - 1]:
                    peak_idx = i - 1
                    break

            opt_feas = feas_series[peak_idx]
            opt_red  = red_series[peak_idx]

            results.append({
                "scenario":             label,
                "combo":                combo,
                "optimal_feas":         opt_feas,
                "optimal_reduction_pu": opt_red,
                "optimal_reduction_mw": opt_red * base_mva,
                "feas_series":          feas_series,
                "reduction_series_mw":  [v * base_mva for v in red_series],
            })
            print(f"  {label:<18}  peak feas = {opt_feas:.6f}  "
                  f"gen reduction = {opt_red * base_mva:.2f} MW")

    return results


def plot_optimization_table(results, save_path=None):
    """
    Render a ranked table of optimization results.
    Top row = scenario with minimum generator reduction needed for peak feas.
    Columns: generator reduction (MW) | battery scenario | feas (optimal)
    """
    if not results:
        print("No optimization results to display.")
        return

    # sort ascending by generator reduction (minimum first)
    ranked = sorted(results, key=lambda x: x["optimal_reduction_mw"])

    col_labels = ["Gen. reduction (MW)", "Battery scenario", "Feas (optimal)"]
    table_data = [
        [f"{r['optimal_reduction_mw']:.2f}", r["scenario"], f"{r['optimal_feas']:.6f}"]
        for r in ranked
    ]

    n_rows  = len(table_data)
    fig_h   = max(3.5, 0.6 * n_rows + 2.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("#f0f2f5")

    row_colors = []
    for i in range(n_rows):
        base = "#d4edda" if i == 0 else ("#f8f9fa" if i % 2 == 0 else "#ffffff")
        row_colors.append([base] * 3)

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=row_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 1.8)

    for j in range(len(col_labels)):
        tbl[(0, j)].set_text_props(fontweight="bold", color="white")
        tbl[(0, j)].set_facecolor("#343a40")

    # highlight top row
    for j in range(len(col_labels)):
        tbl[(1, j)].set_text_props(fontweight="bold")

    ax.set_title(
        "Optimization Results — Battery Shutdown × Generator Reduction\n"
        "sorted by minimum generator reduction required for peak feas  "
        "(top = best scenario)",
        fontsize=12, fontweight="bold", pad=18, color="#1a1a2e",
    )

    plt.tight_layout()
    out = save_path or "optimization_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"Optimization table → {out}")


#%%
import torch
from gridsfm import load_model, predict

GRAPH_FILE   = "samples/case1354_pegase.pyg.json"
MODIFIED_FILE = "samples/case1354_batteries_off.pyg.json"
REPORT_FILE   = "mitigation_report.json"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model  = load_model("checkpoints/gridsfm_open_v1.1.pt", device=device)

# ── baseline run ──────────────────────────────────────────────────────────────
#%%
out_before     = predict(model, GRAPH_FILE)
loading_before = compute_line_loading(out_before)
G, worst, congested_sorted = plot_line_loading(
    loading_before, GRAPH_FILE, save_path="line_loading_before.png"
)

# ── curtail charging batteries, run remedial prediction ───────────────────────
#%%
if worst is not None:
    d_modified, shutdown_details = shutdown_neighborhood_batteries(
        GRAPH_FILE, worst, output_path=MODIFIED_FILE, n_hops=2, max_batteries=3,
    )

    out_after     = predict(model, MODIFIED_FILE)
    loading_after = compute_line_loading(out_after)
    plot_line_loading(loading_after, MODIFIED_FILE, save_path="line_loading_after.png")

    # ── analysis & report ─────────────────────────────────────────────────────
    analysis = analyze_mitigation(loading_before, loading_after, worst, shutdown_details)
    report   = generate_report(worst, analysis, GRAPH_FILE, output_path=REPORT_FILE)
    plot_mitigation_report(worst, analysis, GRAPH_FILE, save_path="mitigation_dashboard.png")

# ── optimization: all battery combinations × generator reduction steps ────────
#%%
if worst is not None and shutdown_details:
    opt_results = run_optimization_scenarios(
        GRAPH_FILE, worst, shutdown_details, model,
        gen_n_hops=10, base_mva=CONFIG["base_mva"],
    )
    plot_optimization_table(opt_results, save_path="optimization_results.png")
# %%
