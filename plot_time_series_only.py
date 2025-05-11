"""Plot time series of causalities for pickle files with more than one lambda value."""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from pathlib import Path
import read_geometry as rr
from shapely import Polygon, LinearRing

if len(sys.argv) == 1:
    sys.exit(f"Usage {sys.argv[0]} pickle_file")

save_path = sys.argv[1]
output_dir = "fig_results"
path = Path(save_path)
stem = path.stem

# Load saved data
with open(save_path, "rb") as f:
    loaded_data = pickle.load(f)

evac_times = loaded_data["evac_times"]
dead = loaded_data["dead"]
fallen_time_series = loaded_data["fallen_time_series"]
cl = loaded_data["cl"]

print("Simulation data successfully loaded.")

# Parse geometry
wkt = rr.parse_geo_file("./Jaleanwala_Bagh.xml")
walkable_area0 = wkt[0]
holes = walkable_area0.interiors[1:]
holes.append(LinearRing([(84, 90), (84, 87), (90, 87), (90, 90), (84, 90)]))
holes.append(LinearRing([(170, 80), (171, 80), (171, 81), (170, 81), (170, 80)]))
holes.append(LinearRing([(100, 40), (101, 40), (101, 41), (100, 41), (100, 40)]))
walkable_area = Polygon(shell=walkable_area0.exterior, holes=holes)

# Extract scenarios
scenarios = list(fallen_time_series.keys())
print(scenarios)
unique_num_agents = sorted(set(num_agents for num_agents, _ in scenarios))
lambda_decays = sorted(set(lambda_decay for _, lambda_decay in scenarios))

print(f"Unique number of agents: {unique_num_agents}")
print(f"Unique lambda decays: {lambda_decays}")

# --------- Time Series Plot per num_agents ----------

max_time = 600  # seconds

for num_agents in unique_num_agents:
    fig, ax = plt.subplots(figsize=(8, 5))

    # Pick all lambda values corresponding to this num_agents
    relevant_lambdas = [
        lambda_decay for (n, lambda_decay) in scenarios if n == num_agents
    ]

    for lambda_decay in relevant_lambdas:
        time_series_list, fallen_series_list = fallen_time_series[
            (num_agents, lambda_decay)
        ]

        color = plt.cm.viridis(lambda_decay / max(lambda_decays))  # Nice color mapping

        for time_series, fallen_series in zip(time_series_list, fallen_series_list):
            cumulative_fallen = np.cumsum(fallen_series)

            if time_series[-1] < max_time:
                extra_times = np.arange(time_series[-1] + 1, max_time + 1)
                extended_time = np.concatenate([time_series, extra_times])
                last_value = cumulative_fallen[-1]
                extended_cumulative = np.concatenate(
                    [cumulative_fallen, np.full(len(extra_times), last_value)]
                )
            else:
                extended_time = time_series
                extended_cumulative = cumulative_fallen

            ax.plot(
                extended_time,
                extended_cumulative,
                color=color,
                alpha=0.6,
                linewidth=1,
                linestyle="-",
                label=f"λ={lambda_decay}"
                if lambda_decay not in ax.get_legend_handles_labels()[1]
                else "",
            )

    ax.set_xlabel("Time [s]", fontsize=18)
    ax.set_ylabel("Cumulative Fallen Agents", fontsize=18)
    # ax.set_title(f"Fallen Agents Over Time (N = {num_agents})")
    ax.grid(alpha=0.3)

    # Create legend only once per lambda
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="")

    output_file = Path(output_dir) / f"{stem}_fallen_time_series_N{num_agents}.pdf"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f">> Saved: {output_file}")
