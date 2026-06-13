#%%
import torch
from gridsfm import load_model, predict

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model("checkpoints/gridsfm_open_v1.1.pt", device=device)
out = predict(model, "samples/case1803_snem.pyg.json")

# %%
print(out)
#%%
print(out["feas"])
# %%
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_grid(json_path):
    with open(json_path) as f:
        d = json.load(f)

    _DEFAULT_SCHEMA = {
        "bus":          ["base_kv", "bus_type_1_pq_2_pv_3_ref", "vmin_pu", "vmax_pu"],
        "solution_bus": ["va_rad", "vm_pu"],
    }
    schema  = d["metadata"].get("feature_schema", _DEFAULT_SCHEMA)
    nodes   = d["grid"]["nodes"]
    edges   = d["grid"]["edges"]
    sol     = d["solution"]["nodes"]

    # bus types and solution voltage magnitudes
    bt_idx  = schema["bus"].index("bus_type_1_pq_2_pv_3_ref")
    vm_idx  = schema["solution_bus"].index("vm_pu")
    buses   = nodes["bus"]
    n_buses = len(buses)
    bus_types = [int(b[bt_idx]) for b in buses]
    vm_pu     = [sol["bus"][i][vm_idx] for i in range(n_buses)]
    vm_min, vm_range = min(vm_pu), max(vm_pu) - min(vm_pu) + 1e-9

    gen_bus  = d["metadata"]["gen_bus_map"]
    load_bus = d["metadata"]["load_bus_map"]

    # build graph
    G = nx.Graph()
    for i in range(n_buses):
        G.add_node(f"b{i}", kind="bus")
    for gi, bi in enumerate(gen_bus):
        G.add_node(f"g{gi}", kind="gen")
        G.add_edge(f"g{gi}", f"b{bi}", etype="gen_link")
    for li, bi in enumerate(load_bus):
        G.add_node(f"l{li}", kind="load")
        G.add_edge(f"l{li}", f"b{bi}", etype="load_link")
    for s, r in zip(edges["ac_line"]["senders"], edges["ac_line"]["receivers"]):
        G.add_edge(f"b{s}", f"b{r}", etype="ac_line")
    for s, r in zip(edges["transformer"]["senders"], edges["transformer"]["receivers"]):
        G.add_edge(f"b{s}", f"b{r}", etype="transformer")

    pos = nx.spring_layout(G, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(13, 10))

    # edges
    edge_groups = {"ac_line": [], "transformer": [], "gen_link": [], "load_link": []}
    for u, v, data in G.edges(data=True):
        edge_groups[data["etype"]].append((u, v))

    nx.draw_networkx_edges(G, pos, edgelist=edge_groups["ac_line"],
                           edge_color="#aaaaaa", width=1.2, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edge_groups["transformer"],
                           edge_color="#c77c2e", width=2.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edge_groups["gen_link"],
                           edge_color="#2a9d8f", width=1.0, style="dashed", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edge_groups["load_link"],
                           edge_color="#9b59b6", width=1.0, style="dashed", ax=ax)

    # bus nodes colored by type, sized by voltage magnitude
    bus_style = {1: ("#457b9d", "PQ bus"), 2: ("#2a9d8f", "PV bus"), 3: ("#e63946", "Slack/Ref")}
    for btype, (color, _) in bus_style.items():
        nl = [f"b{i}" for i in range(n_buses) if bus_types[i] == btype]
        if nl:
            sizes = [150 + 400 * (vm_pu[int(n[1:])] - vm_min) / vm_range for n in nl]
            nx.draw_networkx_nodes(G, pos, nodelist=nl, node_color=color,
                                   node_size=sizes, ax=ax)

    # generator and load nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[f"g{i}" for i in range(len(gen_bus))],
                           node_color="#f4a261", node_shape="^", node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[f"l{i}" for i in range(len(load_bus))],
                           node_color="#9b59b6", node_shape="s", node_size=220, ax=ax)

    nx.draw_networkx_labels(G, pos, labels={f"b{i}": str(i) for i in range(n_buses)},
                            font_size=7, ax=ax)

    legend = [
        mpatches.Patch(color="#457b9d", label="PQ bus"),
        mpatches.Patch(color="#2a9d8f", label="PV bus"),
        mpatches.Patch(color="#e63946", label="Slack/Ref bus"),
        mpatches.Patch(color="#f4a261", label="Generator"),
        mpatches.Patch(color="#9b59b6", label="Load"),
        mpatches.Patch(color="#aaaaaa", label="AC line"),
        mpatches.Patch(color="#c77c2e", label="Transformer"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.85)
    name = json_path.split("/")[-1]
    ax.set_title(f"{name}  —  {n_buses} buses · {len(gen_bus)} generators · {len(load_bus)} loads\n"
                 f"node size ∝ voltage magnitude (solution)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("grid_viz.png", dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved grid_viz.png")


visualize_grid("samples/case500_goc.pyg.json")

# %%
import json

with open("samples/cas30e_oriented_like_case1354.pyg.json") as f:
    d = json.load(f)

loads = d["grid"]["nodes"]["load"]
generators = d["grid"]["nodes"]["generator"]

loads[5][0] = 1000       # set pd of load 5 to 125 MW (per-unit on 100 MVA)
loads[5][1] = 400        # set qd to 41 MVAr
#loads[12][0] *= 100      # or scale: +10% on load 12

generators[5][0] = 1000  # set pg of generator 0 to 500 MW (per-unit on 100 MVA)

generators[5][1] = 400


with open("samples/case30_modified.pyg.json", "w") as f:
    json.dump(d, f)

# %%
import random

def subset_pyg_json(json_path, n_buses, output_path=None, seed=None):
    """
    Return a random subset of n_buses buses from a pyg.json, in the same format.
    Generators, loads, shunts and edges are filtered to those connected to the
    selected buses. All indices are remapped to be contiguous from 0.
    """
    with open(json_path) as f:
        d = json.load(f)

    nodes = d["grid"]["nodes"]
    edges = d["grid"]["edges"]
    sol   = d["solution"]["nodes"]

    total_buses = len(nodes["bus"])
    if n_buses >= total_buses:
        return d

    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_buses), n_buses))
    bus_set  = set(selected)
    bus_remap = {old: new for new, old in enumerate(selected)}

    # ── filter element indices by bus membership ──────────────────────────────
    # generator_link.receivers uses 0-indexed bus rows (same space as bus_set)
    # gen_bus_map stores original PowerModels bus IDs — different indexing
    gl_s = edges["generator_link"]["senders"]
    gl_r = edges["generator_link"]["receivers"]
    gen_to_row  = {gl_s[i]: gl_r[i] for i in range(len(gl_s))}

    ll_s = edges["load_link"]["senders"]
    ll_r = edges["load_link"]["receivers"]
    load_to_row = {ll_s[i]: ll_r[i] for i in range(len(ll_s))}

    sl_s = edges["shunt_link"].get("senders",  [])
    sl_r = edges["shunt_link"].get("receivers", [])
    shunt_to_row = {sl_s[i]: sl_r[i] for i in range(len(sl_s))}

    sel_gens   = [gi for gi in range(len(nodes["generator"])) if gen_to_row.get(gi)   in bus_set]
    sel_loads  = [li for li in range(len(nodes["load"]))      if load_to_row.get(li)  in bus_set]
    sel_shunts = [si for si in range(len(nodes.get("shunt", []))) if shunt_to_row.get(si) in bus_set]

    gen_remap   = {old: new for new, old in enumerate(sel_gens)}
    load_remap  = {old: new for new, old in enumerate(sel_loads)}
    shunt_remap = {old: new for new, old in enumerate(sel_shunts)}

    # ── helper: filter branch edges (both endpoints must be in subset) ────────
    def _filter_branches(edict):
        snd, rcv = edict["senders"], edict["receivers"]
        mask = [i for i, (s, r) in enumerate(zip(snd, rcv)) if s in bus_set and r in bus_set]
        out  = {"senders":   [bus_remap[snd[i]] for i in mask],
                "receivers": [bus_remap[rcv[i]] for i in mask]}
        if "features" in edict:
            out["features"] = [edict["features"][i] for i in mask]
        return out, mask

    ac_edges, ac_mask = _filter_branches(edges["ac_line"])
    tr_edges, tr_mask = _filter_branches(edges["transformer"])

    # ── helper: filter link edges (sender = element idx, receiver = bus idx) ──
    def _filter_links(edict, elem_remap):
        snd, rcv = edict["senders"], edict["receivers"]
        mask = [i for i, s in enumerate(snd) if s in elem_remap]
        return {"senders":   [elem_remap[snd[i]]  for i in mask],
                "receivers": [bus_remap[rcv[i]]   for i in mask]}

    # ── solution edges (branch flows, same senders/receivers/features layout) ──
    sol_edge_out = {}
    for key, branch_mask in (("ac_line", ac_mask), ("transformer", tr_mask)):
        src = d["solution"].get("edges", {}).get(key, {})
        if src:
            sol_edge_out[key] = {
                "senders":   [bus_remap[src["senders"][i]]   for i in branch_mask],
                "receivers": [bus_remap[src["receivers"][i]] for i in branch_mask],
                "features":  [src["features"][i]             for i in branch_mask],
            }

    subset = {
        "grid": {
            "nodes": {
                "bus":       [nodes["bus"][i]       for i in selected],
                "generator": [nodes["generator"][i] for i in sel_gens],
                "load":      [nodes["load"][i]      for i in sel_loads],
                "shunt":     [nodes["shunt"][i]     for i in sel_shunts] if nodes.get("shunt") else [],
            },
            "edges": {
                "ac_line":        ac_edges,
                "transformer":    tr_edges,
                "generator_link": _filter_links(edges["generator_link"], gen_remap),
                "load_link":      _filter_links(edges["load_link"],      load_remap),
                "shunt_link":     _filter_links(edges["shunt_link"],     shunt_remap),
            },
            "context": d["grid"].get("context", {}),
        },
        "solution": {
            "nodes": {
                "bus":       [sol["bus"][i]       for i in selected],
                "generator": [sol["generator"][i] for i in sel_gens],
            },
            "edges": sol_edge_out,
            "duals": {},
        },
        "metadata": {
            **{k: v for k, v in d["metadata"].items()
               if k not in {"gen_bus_map", "load_bus_map", "gen_id_map", "load_id_map",
                             "bus_id_map", "ac_line_branch_ids", "transformer_branch_ids"}},
            # gen_bus_map keeps original PowerModels bus IDs (not row indices)
            "gen_bus_map":  [d["metadata"]["gen_bus_map"][i]  for i in sel_gens],
            "load_bus_map": [d["metadata"]["load_bus_map"][i] for i in sel_loads],
        },
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(subset, f, indent=2)
        print(f"Subset written: {output_path} "
              f"({n_buses} buses, {len(sel_gens)} gens, {len(sel_loads)} loads, "
              f"{len(ac_edges['senders'])} ac_lines, {len(tr_edges['senders'])} trafos)")
    return subset


# example: take 15 random buses from the oriented case30
subset_pyg_json(
    "samples/case1803_snem.pyg.json",
    n_buses=20,
    output_path="samples/case1803_subset20.pyg.json",
    seed=42,
)
# %%
