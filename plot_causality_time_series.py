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
        cmap="jet",
        interpolation="lanczos",
    )

    pedpy.plot_walkable_area(
        walkable_area=pedpy.WalkableArea(walkable_area),
        line_width=2,
        line_color="white",
        axes=ax,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label="Number of Fallen Agents")

    ax.set_xlabel("X [m]", fontsize=18)
    ax.set_ylabel("Y [m]", fontsize=18)
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
cl = loaded_data["cl"]

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

scenarios = list(dead.keys())
unique_num_agents = sorted(set(num_agents for num_agents, _ in scenarios))
lambda_decays = sorted(set(lambda_decay for _, lambda_decay in scenarios))

fig, ax = plt.subplots()

for num_agents in unique_num_agents:
    lambdas = []
    means = []
    stds = []
    for n, l in scenarios:
        if n == num_agents:
            lambdas.append(l)
            means.append(np.mean(dead[(n, l)]))
            stds.append(np.std(dead[(n, l)]))

    lambdas, means, stds = map(np.array, (lambdas, means, stds))
    sorted_idx = np.argsort(lambdas)
    ax.errorbar(
        lambdas[sorted_idx],
        means[sorted_idx],
        yerr=stds[sorted_idx],
        label=f"N={num_agents}",
        fmt="o-",
        capsize=5,
    )

ax.set_xlabel(r"$\lambda$", fontsize=18)
ax.set_xticks(lambda_decays)
ax.set_ylabel("Number of Dead Agents", fontsize=18)
ax.grid(alpha=0.1)
ax.legend(title="Number of Agents")
plt.tight_layout()

plot1_path = Path(output_dir) / f"{stem}_dead_vs_lambda_allN.pdf"
fig.savefig(
    plot1_path,
    bbox_inches="tight",
    pad_inches=0.1,
)
print(f">> Saved plot: {plot1_path}")
plt.close(fig)

# ---------------------------
# 2. Plot Causality Heatmaps per (lambda, num_agents)

min_x, min_y, max_x, max_y = walkable_area.bounds

for (num_agents, lambda_decay), fallen_positions in cl.items():
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
