import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
import read_geometry as rr

from sys import argv
import sys
import pickle
import numpy as np

if len(argv) == 1:
    sys.exit(f"Usage {argv[0]} pickle file")

save_path = argv[1]
output_dir = "fig_results"

num_agents = save_path.split("_")[-1].split(".")[0]
print(f"{num_agents = }")


wkt = rr.parse_geo_file("./Jaleanwala_Bagh.xml")
walkable_area = wkt[0]


# Load the saved data
with open(save_path, "rb") as f:
    loaded_data = pickle.load(f)

# Extract variables
evac_times = loaded_data["evac_times"]
dead = loaded_data["dead"]
fallen_time_series = loaded_data["fallen_time_series"]
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

fig4, ax4 = plt.subplots(figsize=(10, 6))

# Get a colormap with distinct colors

colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_decays)))
color = "gray"
for i, lambda_decay in enumerate(lambda_decays):
    print(f"Plot with Lambda {lambda_decay}")
    time_series, fallen_series = fallen_time_series[lambda_decay]
    for time_serie, fallen_serie in zip(time_series, fallen_series):
        ax3.plot(time_serie, fallen_serie, color=color, alpha=0.3, linewidth=0.8)
        cumulative_fallen = np.cumsum(fallen_serie)
        ax3.plot(
            time_serie,
            cumulative_fallen,
            color=color,
            alpha=0.3,
            linewidth=0.8,
            linestyle="--",
        )

    representative_time = time_series[0]
    representative_fallen = fallen_series[0]
    representative_cumulative_fallen = np.cumsum(fallen_series[0])
    ax3.plot(
        representative_time,
        representative_fallen,
        label=rf"$\lambda = {lambda_decay}$",
        color=color,
        linewidth=2,
    )

    ax3.plot(
        representative_time,
        representative_cumulative_fallen,
        label=rf"Cumulative $\lambda = {lambda_decay}$",
        color=color,
        linestyle="--",
        linewidth=2,
    )

ax3.set_xlabel("Time [seconds]")
ax3.set_ylabel("New Fallen Agents per Time Step")
ax3.set_title(rf"Fallen Agents $\approx$ {int(np.sum(fallen_series[0]))}")
ax3.grid(alpha=0.3)
ax3.legend()
plt.tight_layout()
fig3.savefig(f"{output_dir}/fallen_agents_time_series_{num_agents}.pdf")

for lambda_decay in lambda_decays:
    min_x, min_y, max_x, max_y = walkable_area.bounds

    grid_size_x = int(max_x // 5)  # Scale down grid for visualization
    grid_size_y = int(max_y // 5)

    # Initialize the casualty grid
    causality_locations = cl[lambda_decay]
    for ic, causality_location in enumerate(causality_locations):
        fig4, ax4 = plt.subplots(figsize=(8, 10))
        causality_grid = np.zeros((grid_size_x, grid_size_y))
        for (x, y), count in causality_location.items():
            grid_x = int(x // 5)  # Scale coordinates for grid
            grid_y = int(y // 5)
            if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
                causality_grid[x, y] += count

        # Plot heatmap
        im = ax4.imshow(
            causality_grid.T, cmap="jet", origin="lower", interpolation="lanczos"
        )
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig4.colorbar(im, cax=cax, label="Number of Fallen Agents")
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x)}")
        )  # Format as int
        vmin, vmax = np.min(causality_grid), np.max(causality_grid)
        cbar.set_ticks(np.linspace(vmin, vmax, num=2))  # Set 5 evenly spaced ticks

        ax4.set_xlabel("X [m]")
        ax4.set_ylabel("Y [m]")
        ax4.set_title(f"Locations of Fallen Agents (λ={lambda_decay})")
        # Save heatmap
        fig4.savefig(
            f"{output_dir}/Casualty_Locations_{num_agents}_lambda_{lambda_decay}_{ic:03d}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig4)
