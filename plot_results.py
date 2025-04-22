"""Plot the results of the simulation"""
# Usage: python plot_results.py <pickle file>
# 1. Evacuation time vs lambda
# 2. Number of dead agents vs lambda
# 3. Time series of fallen agents
# 4. Heatmap of fallen agents
# 5. GIF of heatmap sequence

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import read_geometry as rr
from PIL import Image
import glob
import pedpy
from shapely import Polygon, Point, LinearRing, intersection

from sys import argv
import sys
import pickle
import numpy as np
import re


def plot_causality_grids_by_lambda(
    fig,
    ax,
    walkable_area,
    data_by_lambda,
    grid_size=1.0,
    min_x=0,
    max_x=220,
    min_y=0,
    max_y=130,
):
    """
    data_by_lambda: {lambda_value: [ [(x, y), (x, y), ...], [(x, y), ...], ... ]}
    grid_size: resolution of the grid
    """

    for lambda_val, runs in data_by_lambda.items():
        print(f"Plotting for λ = {lambda_val}")

        # Flatten all positions from all runs
        positions = [pos for run in runs for pos in run]

        # Create grid
        width = int(np.ceil((max_x - min_x) / grid_size))
        height = int(np.ceil((max_y - min_y) / grid_size))
        grid = np.zeros((width, height), dtype=int)

        for x, y in positions:
            grid_x = int((x - min_x) // grid_size)
            grid_y = int((y - min_y) // grid_size)

            if 0 <= grid_x < width and 0 <= grid_y < height:
                grid[grid_x, grid_y] += 1

        eps = 0
        extent = [min_x - eps, max_x + eps, min_y - eps, max_y + eps]
        vmin, vmax = np.min(grid), np.max(grid)

        im = ax.imshow(
            grid.T,
            origin="lower",
            extent=extent,
            cmap="jet",
            interpolation="lanczos",
            vmin=vmin,
            vmax=vmax,
        )

        pedpy.plot_walkable_area(
            walkable_area=pedpy.WalkableArea(walkable_area),
            line_width=2,
            line_color="white",
            axes=ax,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label="Number of Fallen Agents")
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x)}")
        )  # Format as int
        cbar.set_ticks(np.linspace(vmin, vmax, num=2))  # Set 5 evenly spaced ticks
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

        plt.tight_layout()
        plt.show()

    # Set labels and title
    ax.set_xlabel("X [m]", fontsize=18)
    ax.set_ylabel("Y [m]", fontsize=18)
    # ax.set_title(f"Locations of Fallen Agents (λ={lambda_decay}, N={num_agents})")
    return fig


if len(argv) == 1:
    sys.exit(f"Usage {argv[0]} pickle file")

save_path = argv[1]
output_dir = "fig_results"

# num_agents = save_path.split("_")[-1].split(".")[0]

match = re.search(r"simulation_data_(\d+)_\d+", save_path)

# Get the number of agents if found
num_agents = int(match.group(1)) if match else None

print(f"{num_agents = }")


wkt = rr.parse_geo_file("./Jaleanwala_Bagh.xml")
# walkable_area = wkt[0]
walkable_area0 = wkt[0]
holes = walkable_area0.interiors[1:]
holes.append(LinearRing([(84, 90), (84, 87), (90, 87), (90, 90), (84, 90)]))
holes.append(LinearRing([(170, 80), (171, 80), (171, 81), (170, 81), (170, 80)]))
holes.append(LinearRing([(100, 40), (101, 40), (101, 41), (100, 41), (100, 40)]))
walkable_area = Polygon(shell=walkable_area0.exterior, holes=holes)

# Load the saved data
with open(save_path, "rb") as f:
    loaded_data = pickle.load(f)


# print(loaded_data)
# Extract variables
evac_times = loaded_data["evac_times"]
dead = loaded_data["dead"]
fallen_time_series = loaded_data["fallen_time_series"]
print(dead)
cl = loaded_data["cl"]
print("Simulation data successfully loaded.")

lambda_decays = [0.5]
# %% Ploting
mean_evac_times = {scenario: np.mean(times) for scenario, times in evac_times.items()}
std_dev_evac_times = {scenario: np.std(times) for scenario, times in evac_times.items()}

mean_dead = {scenario: np.mean(dead) for scenario, dead in dead.items()}
std_dead = {scenario: np.std(dead) for scenario, dead in dead.items()}

means = [mean_evac_times[scenario] for scenario in lambda_decays]
std_devs = [std_dev_evac_times[scenario] for scenario in lambda_decays]

means1 = [mean_dead[scenario] for scenario in lambda_decays]
std_devs1 = [std_dead[scenario] for scenario in lambda_decays]

fig1, ax1 = plt.subplots(nrows=1, ncols=1)
fig2, ax2 = plt.subplots(nrows=1, ncols=1)

ax1.errorbar(lambda_decays, means, yerr=std_devs, fmt="o-", ecolor="blue")
ax1.set_xlabel(r"$\lambda$")
ax1.set_ylabel("max. simulation itme [min]")

ax2.errorbar(lambda_decays, means1, yerr=std_devs1, fmt="o-", ecolor="red")
ax2.set_xlabel(r"$\lambda$")
ax2.set_ylabel("Number of agents lying on the ground")

ax2.set_xticks(lambda_decays)
ax1.set_xticks(lambda_decays)
ax2.grid(alpha=0.1)
ax1.grid(alpha=0.1)

# plt.tight_layout()

fig1.savefig(f"{output_dir}/result1_{num_agents}.pdf")
fig2.savefig(f"{output_dir}/result2_{num_agents}.pdf")

fig3, ax3 = plt.subplots(figsize=(8, 5))
# heatmaps
fig4, ax4 = plt.subplots(figsize=(10, 6))

# Get a colormap with distinct colors

colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_decays)))
color = "gray"
output_file_stats = f"{output_dir}/fallen_agents_stats_{num_agents}.txt"
for i, lambda_decay in enumerate(lambda_decays):
    sums = []
    print(f"Plot with Lambda {lambda_decay}")
    time_series, fallen_series = fallen_time_series[lambda_decay]
    #    print(time_series)
    for time_serie, fallen_serie in zip(time_series, fallen_series):
        # ax3.plot(time_serie, fallen_serie, color=color, alpha=0.3, linewidth=0.8)
        cumulative_fallen = np.cumsum(fallen_serie)
        ax3.plot(
            time_serie,
            cumulative_fallen,
            color=color,
            alpha=0.3,
            linewidth=0.8,
            linestyle="--",
        )
        sums.append(np.sum(fallen_serie))

    # Find the index of the longest time series
    longest_index = np.argmax([len(ts) for ts in time_series])
    # Use the longest series as the representative
    representative_time = time_series[longest_index]
    representative_fallen = fallen_series[longest_index]
    representative_cumulative_fallen = np.cumsum(fallen_series[longest_index])
    mean_fallen = int(np.mean(sums))
    std_fallen = int(np.std(sums))
    ax3.set_title(
        rf"Fallen Agents $\approx$ {mean_fallen} $\pm$ {std_fallen}  (N={num_agents})"
    )
    ax3.plot(
        representative_time,
        representative_cumulative_fallen,
        label=rf"Cumulative $\lambda = {lambda_decay}$",
        color=color,
        linestyle="-",
        linewidth=2,
    )

    with open(output_file_stats, "w") as f:
        f.write(f"{lambda_decay},{mean_fallen},{std_fallen},{num_agents}")
    print(">> ", output_file_stats)

ax3.set_xlabel("Time [seconds]")
ax3.set_ylabel("New Fallen Agents per Time Step")
# ax3.set_title(rf"Fallen Agents $\approx$ {int(np.sum(fallen_series[0]))}")

ax3.grid(alpha=0.3)
ax3.legend()
plt.tight_layout()
fig3.savefig(f"{output_dir}/fallen_agents_time_series_{num_agents}.pdf")

grid_size = 3
for lambda_decay in lambda_decays:
    min_x, min_y, max_x, max_y = walkable_area.bounds
    # Initialize the casualty grid
    causality_locations = cl[lambda_decay]
    fig4 = plot_causality_grids_by_lambda(
        fig=fig4,
        ax=ax4,
        walkable_area=walkable_area,
        data_by_lambda=cl,
        grid_size=grid_size,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
    )
    # Save heatmap
    print(f"{output_dir}/Casualty_Locations_{num_agents}_lambda_{lambda_decay}.pdf")
    fig4.savefig(
        f"{output_dir}/Casualty_Locations_{num_agents}_lambda_{lambda_decay}.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig4)
