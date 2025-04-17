"""
This script generates a sequence of heatmaps showing the probability of survival.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# Define the probability function
def prob_function(d, t, lambda_val, time_scale):
    distance_factor = 1 - np.exp(-2 * d)
    time_factor = np.exp(-lambda_val * (t / time_scale))
    return distance_factor * time_factor


# Define parameters
time_scale = 600  # 10 minutes
grid_width, grid_height = 100, 100
distances = np.linspace(0, 1, grid_width)
times = np.linspace(0, time_scale, grid_height)

# Directory to save images
output_dir = "heatmap_frames"
os.makedirs(output_dir, exist_ok=True)

# Lambda values for the heatmap sequence

lambda_values1 = np.linspace(0.05, 2.0, 50)
lambda_values2 = np.linspace(2.0, 0.5, 50)  # Corrected linspace

lambda_values = np.concatenate(
    (lambda_values1, lambda_values2)
)  # Correct concatenation

fig_width, fig_height = 10, 10  # Inches
dpi = 150  # Resolution

# Generate heatmaps for each lambda value
for idx, lambda_val in enumerate(lambda_values):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    P = np.array(
        [
            [prob_function(d, t, lambda_val, time_scale) for d in distances]
            for t in times
        ]
    )

    c = ax.imshow(
        P, origin="lower", cmap="jet_r", extent=[0, 1, 0, time_scale], aspect="auto"
    )

    # Labels and title
    #    ax.set_xlabel("Distance")  # Distance on X-axis
    ax.set_ylabel("Time [min]")  # Time on Y-axis
    ax.set_title(f"λ = {lambda_val:.2f}")
    ax.set_xticks([0, 0.5])
    ax.set_xticklabels(
        [r"$\uparrow$ danger line", r"$\longrightarrow$ Increasing distance"],
        fontsize=10,
        rotation=0,
        ha="left",
    )
    ax.set_yticks([0, time_scale])
    ax.set_yticklabels(
        [0, 10],
        fontsize=10,
        # rotation=90,
        # ha="left",
    )

    fig.colorbar(c, ax=ax, orientation="vertical", label="Probability of survival")

    filename = os.path.join(output_dir, f"heatmap_{idx:03d}.png")
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

print(f"Heatmaps saved in: {output_dir}")
