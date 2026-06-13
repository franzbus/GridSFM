#%%
import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pandapower.networks.power_system_test_cases import case30, case145

net = case145()
print(net)
pp.to_json(net, "case145.pyg.json")
#%%



def plot_simple(net):
    """Quick overview using pandapower's built-in simple_plot."""
    plot.simple_plot(net, show_plot=True)


def plot_detailed(net, threshold=80.0, filename="case145_plot.png"):
    """Custom plot: buses colored by type, lines colored by loading level."""

    pp.runpp(net)

    bus_size = plot.get_collection_sizes(net, bus_size=0.4)["bus"]

    # lines split by loading
    congested = net.res_line.index[net.res_line.loading_percent > threshold]
    normal    = net.res_line.index[net.res_line.loading_percent <= threshold]

    lc_normal    = plot.create_line_collection(net, normal,    color="#aaaaaa",
                                               linewidths=1.0, use_bus_geodata=True)
    lc_congested = plot.create_line_collection(net, congested, color="#e63946",
                                               linewidths=2.5, use_bus_geodata=True)

    # buses by element type (priority: ext_grid > gen > load > plain)
    ext_buses   = set(net.ext_grid.bus)
    gen_buses   = set(net.gen.bus)  - ext_buses
    load_buses  = set(net.load.bus) - ext_buses - gen_buses
    plain_buses = set(net.bus.index) - ext_buses - gen_buses - load_buses

    node_types = [
        ("Ext. Grid", sorted(ext_buses),   "#e63946"),
        ("Generator", sorted(gen_buses),   "#2a9d8f"),
        ("Load",      sorted(load_buses),  "#457b9d"),
        ("Bus",       sorted(plain_buses), "#cccccc"),
    ]

    collections = [lc_normal, lc_congested]
    legend_handles = [
        mpatches.Patch(color="#aaaaaa", label="Normal line"),
        mpatches.Patch(color="#e63946", label=f"Congested line (>{threshold:.0f}%)"),
    ]

    for label, buses, color in node_types:
        if buses:
            collections.append(
                plot.create_bus_collection(net, buses, size=bus_size, color=color, zorder=10)
            )
            legend_handles.append(mpatches.Patch(color=color, label=label))

    fig, ax = plt.subplots(figsize=(10, 8))
    plot.draw_collections(collections, ax=ax)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_title(
        f"IEEE Case 30  —  {len(net.bus)} buses · {len(net.gen)} generators · {len(net.load)} loads\n"
        f"Max line loading: {net.res_line.loading_percent.max():.1f}%  |  threshold: {threshold:.0f}%"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved {filename}  ({len(congested)} congested lines)")


# %%

plot_simple(net)
plot_detailed(net, threshold=80.0, filename="case145_detailed.png")

# %%
from pandapower.plotting.simple_plot import simple_plot

# we want to highlight overhead power liens and connected buses
ol_lines = net.line.loc[net.line.type=="ol"].index
ol_buses = net.bus.index[net.bus.index.isin(net.line.from_bus.loc[ol_lines]) |
                         net.bus.index.isin(net.line.to_bus.loc[ol_lines])]

simple_plot(net, highlight_lines=ol_lines, highlight_buses=ol_buses, enable_hover=True)
#%%