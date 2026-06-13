"""
SimBench EHV grid: Snapshot 02.01.2020 19:00, congested lines (>90%) in red,
plus demo for adding loads and generators at selected buses.

Tested with:  pandapower 3.4.0  |  simbench 1.6.2
Installation: pip install pandapower simbench networkx matplotlib
"""

import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.plotting as plot
import simbench as sb

SB_CODE   = "1-EHV-mixed--0-sw"   # mixed EHV grid Germany, scenario 0 (base year 2016)
TIMESTAMP = "02.01.2016 19:00"    # format as in net.profiles['load']['time']
THRESHOLD = 300.0                  # lines above this loading [%] are considered congested


# ----------------------------------------------------------------------------
# 1) Load network
# ----------------------------------------------------------------------------
def load_network():
    """Loads the SimBench EHV grid. Data is bundled in the simbench package,
    no download or account required."""
    net = sb.get_simbench_net(SB_CODE)
    # Latitude/longitude per bus from GeoJSON (pandapower 3.x: coordinates in net.bus['geo'])
    net.bus["lat"] = net.bus["geo"].apply(
        lambda g: json.loads(g)["coordinates"][1] if isinstance(g, str) else None)
    net.bus["lon"] = net.bus["geo"].apply(
        lambda g: json.loads(g)["coordinates"][0] if isinstance(g, str) else None)
    return net


# ----------------------------------------------------------------------------
# 2) Set time step
# ----------------------------------------------------------------------------
def set_timestep(net, timestamp=TIMESTAMP):
    """Writes the absolute load/generation values for the requested time step
    from the annual profiles (15-min grid) into the element tables."""
    abs_vals = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    # Find time step index robustly via the time column (instead of hardcoding "128").
    time_col = net.profiles["load"]["time"]
    matches = time_col.index[time_col == timestamp]
    if len(matches) == 0:
        raise ValueError(f"Timestamp {timestamp!r} not found in profile.")
    ts = matches[0]

    for (table, column), df in abs_vals.items():
        if len(df.columns) and ts in df.index:
            net[table].loc[df.columns, column] = df.loc[ts]
    return ts


# ----------------------------------------------------------------------------
# 3) (optional) Add loads and generators at selected buses
# ----------------------------------------------------------------------------
def _well_meshed_380kv_buses(net):
    """380 kV buses with degree >= 3 (avoids divergence at weakly connected
    fringe buses like coastal/alpine stubs)."""
    deg = pd.concat([net.line.from_bus, net.line.to_bus]).value_counts()
    net.bus["deg"] = net.bus.index.map(deg).fillna(0)
    return net.bus[(net.bus.vn_kv > 300) & (net.bus.deg >= 3)]


def _buses_without_voltage_control(net, candidates):
    """Filters out buses that are merged with an already voltage-controlled bus
    (gen/ext_grid) via closed busbar switches.
    Needed because two gen/ext_grid on the same merged bus with different vm_pu
    causes 'different setpoints' errors."""
    g = nx.Graph()
    g.add_nodes_from(net.bus.index)
    bb = net.switch[(net.switch.et == "b") & (net.switch.closed)]
    g.add_edges_from(zip(bb.bus, bb.element))
    controlled = set(net.gen.bus) | set(net.ext_grid.bus)
    controlled_merged = set()
    for component in nx.connected_components(g):
        if component & controlled:
            controlled_merged |= component
    return [b for b in candidates if b not in controlled_merged]


def add_elements(net):
    """Example: additional wind farm + conventional power plant in the north,
    additional industrial load in the south. Buses chosen to be well-meshed."""
    b380 = _well_meshed_380kv_buses(net)
    north_bus = b380[b380.lat >= b380.lat.quantile(0.75)].sort_values("deg").index[-1]
    south_bus = b380[b380.lat <= b380.lat.quantile(0.25)].sort_values("deg").index[-1]

    # WIND -> create_sgen: static generator, PQ injection (like negative load).
    # Correct choice for wind/PV. Never causes voltage setpoint conflicts.
    pp.create_sgen(net, bus=north_bus, p_mw=1000, q_mvar=0,
                   name="Wind Farm North", type="WP")

    # LOAD -> create_load: active and reactive power demand at a bus.
    pp.create_load(net, bus=south_bus, p_mw=800, q_mvar=160,
                   name="Industrial Load South")

    # CONVENTIONAL PLANT -> create_gen: voltage-controlled PV bus (holds vm_pu).
    # Must be connected to a bus WITHOUT existing voltage control.
    north_gen_candidates = b380[b380.lat >= b380.lat.quantile(0.60)].sort_values("deg").index[::-1]
    gen_bus = _buses_without_voltage_control(net, north_gen_candidates)[0]
    pp.create_gen(net, bus=gen_bus, p_mw=500, vm_pu=1.0, name="Power Plant North")

    print(f"  + Wind Farm (sgen)  at bus {north_bus}")
    print(f"  + Power Plant (gen) at bus {gen_bus}")
    print(f"  + Industrial Load   at bus {south_bus}")


# ----------------------------------------------------------------------------
# 4) Plot: congested lines (>THRESHOLD %) in red, nodes coloured by type
# ----------------------------------------------------------------------------
def plot_congestion(net, threshold=THRESHOLD, filename="ehv_congestion.png"):
    congested = net.res_line.index[net.res_line.loading_percent > threshold]
    normal    = net.res_line.index[net.res_line.loading_percent <= threshold]

    bus_size = plot.get_collection_sizes(net, bus_size=0.4)["bus"]

    lc_normal    = plot.create_line_collection(net, normal,    color="lightgrey",
                                               linewidths=0.6, use_bus_geodata=True)
    lc_congested = plot.create_line_collection(net, congested, color="red",
                                               linewidths=2.2, use_bus_geodata=True)

    # Classify buses by element type (priority: ext_grid > gen > sgen > load > plain)
    ext_buses   = set(net.ext_grid.bus)
    gen_buses   = set(net.gen.bus)  - ext_buses
    sgen_buses  = set(net.sgen.bus) - ext_buses - gen_buses
    load_buses  = set(net.load.bus) - ext_buses - gen_buses - sgen_buses
    plain_buses = set(net.bus.index) - ext_buses - gen_buses - sgen_buses - load_buses

    node_types = [
        ("Ext. Grid", sorted(ext_buses),   "#e63946"),
        ("Generator", sorted(gen_buses),   "#2a9d8f"),
        ("Stat. Gen", sorted(sgen_buses),  "#f4a261"),
        ("Load",      sorted(load_buses),  "#457b9d"),
        ("Bus",       sorted(plain_buses), "#aaaaaa"),
    ]

    collections = [lc_normal, lc_congested]
    legend_handles = []
    for label, buses, color in node_types:
        if buses:
            collections.append(
                plot.create_bus_collection(net, buses, size=bus_size, color=color, zorder=10)
            )
            legend_handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color, markersize=8, label=label)
            )

    fig, ax = plt.subplots(figsize=(8, 9))
    plot.draw_collections(collections, ax=ax)
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.85)
    ax.set_title(f"SimBench EHV · {TIMESTAMP} · Loading > {threshold:.0f}% in red")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=130, bbox_inches="tight")
    print(f"Plot saved: {filename}  ({len(congested)} congested lines)")
    plt.show()


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ADD_EXTRA_ELEMENTS = False   # set to True to activate step 3

    net = load_network()
    ts = set_timestep(net, TIMESTAMP)
    print(f"Time step {ts} set ({TIMESTAMP}). "
          f"Total load: {net.load.p_mw.sum():.0f} MW")

    if ADD_EXTRA_ELEMENTS:
        add_elements(net)

    try:
        pp.runpp(net)
    except pp.LoadflowNotConverged:
        print("Power flow did not converge. Tip: reduce extra generation, "
              "choose well-meshed buses, or try runpp(net, init='dc').")
        raise

    n_congested = int((net.res_line.loading_percent > THRESHOLD).sum())
    print(f"Power flow OK. Max. line loading: "
          f"{net.res_line.loading_percent.max():.0f} %  |  Congested >"
          f"{THRESHOLD:.0f}%: {n_congested}")

    plot_congestion(net)
