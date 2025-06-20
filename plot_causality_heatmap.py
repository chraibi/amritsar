import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import read_geometry as rr
import pedpy
from shapely import Polygon, LinearRing
from pathlib import Path
import pickle
import numpy as np
import sys
import os


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
):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Flatten all runs
    positions = [pos for run in fallen_positions for pos in run]

    width = int(np.ceil((max_x - min_x) / grid_size))
    height = int(np.ceil((max_y - min_y) / grid_size))
    grid = np.zeros((width, height), dtype=int)

    color_map = "inferno"

    for x, y in positions:
        grid_x = int((x - min_x) // grid_size)
        grid_y = int((y - min_y) // grid_size)

        if 0 <= grid_x < width and 0 <= grid_y < height:
            grid[grid_x, grid_y] += 1

    extent = [min_x, max_x, min_y, max_y]
    im = ax.imshow(
        grid.T,
        origin="lower",
        extent=extent,
        cmap=color_map,
        interpolation="lanczos",
    )

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
    cbar.set_label("Number of Fallen Agents", fontsize=fs)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x)}")
    )  # Format as int
    ax.set_xlabel("X [m]", fontsize=fs)
    ax.set_ylabel("Y [m]", fontsize=fs)
    ax.set_xticklabels(ax.get_xticks(), fontsize=fs)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fs)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    plt.tight_layout()

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
if len(sys.argv) == 1:
    sys.exit(f"Usage {sys.argv[0]} pickle_file")

save_path = sys.argv[1]
output_dir = "fig_results"
path = Path(save_path)
stem = path.stem

# Load data
with open(save_path, "rb") as f:
    loaded_data = pickle.load(f)

evac_times = loaded_data["evac_times"]
dead = loaded_data["dead"]
fallen_time_series = loaded_data["fallen_time_series"]
cl = loaded_data["results"]

print("Simulation data successfully loaded.")

# Read walkable area
wkt = rr.parse_geo_file("./Jaleanwala_Bagh.xml")
walkable_area0 = wkt[0]
holes = walkable_area0.interiors[1:]
holes.append(LinearRing([(84, 90), (84, 87), (90, 87), (90, 90), (84, 90)]))
holes.append(LinearRing([(170, 80), (171, 80), (171, 81), (170, 81), (170, 80)]))
holes.append(LinearRing([(100, 40), (101, 40), (101, 41), (100, 41), (100, 40)]))
walkable_area = Polygon(shell=walkable_area0.exterior, holes=holes)

# ---------------------------
# 1. Plot Dead Agents vs Lambda for Different Num_Agents
fig, ax = plt.subplots()


# ---------------------------
# 2. Plot Causality Heatmaps per (lambda, num_agents)

min_x, min_y, max_x, max_y = walkable_area.bounds

for (num_agents, lambda_decay, _), fallen_positions in cl.items():
    folder = Path(output_dir) / f"N_{num_agents}"
    folder.mkdir(parents=True, exist_ok=True)
    heatmap_file = folder / f"{stem}_causality_lambda_{lambda_decay}_N_{num_agents}.pdf"

    plot_causality_grid(
        walkable_area=walkable_area,
        fallen_positions=fallen_positions,
        output_file=heatmap_file,
        grid_size=3,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
    )
    print(f">> Saved heatmap: {heatmap_file}")
