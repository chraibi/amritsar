import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pedpy
from pathlib import Path
import numpy as np

from plot_utils import load_results, walkable_area


# ---------------------------
def plot_causality_grid(
    walkable_area,
    fallen_positions,
    output_file,
    grid_size=3,
    min_x=0,
    max_x=220,
    min_y=0,
    max_y=130,
    vmax=None,
):
    """Fallen agents per cell, averaged over runs, on a log colour scale.

    vmax: common upper limit so panels of one sweep are comparable; defaults to
    the maximum of this grid.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    grid = count_grid(fallen_positions, grid_size, min_x, max_x, min_y, max_y)
    vmax = grid.max() if vmax is None else vmax
    masked = np.ma.masked_less(grid.T, 1.0)  # empty cells drawn in black

    extent = [min_x, max_x, min_y, max_y]
    im = ax.imshow(
        masked,
        origin="lower",
        extent=extent,
        cmap="inferno",
        interpolation="nearest",
        norm=LogNorm(vmin=1.0, vmax=max(vmax, 2.0)),
    )
    ax.set_facecolor("black")

    pedpy.plot_walkable_area(
        walkable_area=pedpy.WalkableArea(walkable_area),
        line_width=2,
        line_color="white",
        axes=ax,
    )
    fs = 20
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label(f"Fallen agents per {grid_size} m cell (mean over runs)", fontsize=fs)
    ax.set_xlabel("X [m]", fontsize=fs)
    ax.set_ylabel("Y [m]", fontsize=fs)
    ax.set_xticklabels(ax.get_xticks(), fontsize=fs)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fs)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    plt.tight_layout()

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def count_grid(fallen_positions, grid_size, min_x, max_x, min_y, max_y):
    """Mean number of fallen agents per cell over the runs in fallen_positions."""
    width = int(np.ceil((max_x - min_x) / grid_size))
    height = int(np.ceil((max_y - min_y) / grid_size))
    grid = np.zeros((width, height))
    for run in fallen_positions:
        for x, y in run:
            gx, gy = int((x - min_x) // grid_size), int((y - min_y) // grid_size)
            if 0 <= gx < width and 0 <= gy < height:
                grid[gx, gy] += 1
    return grid / max(len(fallen_positions), 1)


# ---------------------------
loaded_data, stem, output_dir = load_results()

evac_times = loaded_data["evac_times"]
dead = loaded_data["dead"]
fallen_time_series = loaded_data["fallen_time_series"]
cl = loaded_data["results"]

print("Simulation data successfully loaded.")

walkable_area = walkable_area()

# ---------------------------
# 1. Plot Dead Agents vs Lambda for Different Num_Agents
fig, ax = plt.subplots()


# ---------------------------
# 2. Plot Causality Heatmaps per (lambda, num_agents)

min_x, min_y, max_x, max_y = walkable_area.bounds
grid_size = 3
# Common colour limit per crowd size so alpha/kappa panels are comparable
vmax_by_n = {}
for (num_agents, *_), fallen_positions in cl.items():
    g = count_grid(fallen_positions, grid_size, min_x, max_x, min_y, max_y)
    vmax_by_n[num_agents] = max(vmax_by_n.get(num_agents, 0), g.max())

for (num_agents, _lambda_decay, alpha, kappa), fallen_positions in cl.items():
    folder = Path(output_dir) / f"N_{num_agents}"
    folder.mkdir(parents=True, exist_ok=True)
    heatmap_file = (
        folder
        / f"{stem}_causality_alpha_{alpha}_kappa_{kappa}_N_{num_agents}.pdf"
    )

    plot_causality_grid(
        walkable_area=walkable_area,
        fallen_positions=fallen_positions,
        output_file=heatmap_file,
        grid_size=grid_size,
        vmax=vmax_by_n[num_agents],
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
    )
    print(f">> Saved heatmap: {heatmap_file}")
