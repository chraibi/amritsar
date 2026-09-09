"""
This script generates a sequence of heatmaps showing the probability of survival.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# Define the probability function
def prob_function(d, t, lambda_val, time_scale):
    distance_factor = 1 - np.exp(-2 * d)
    d_crit = 10
    k = 10
    distance_factor = 1 / (1 + np.exp(-(d - d_crit) / k))

    time_factor = np.exp(-lambda_val * (t / time_scale))
    return distance_factor * time_factor


# Define parameters
time_scale = 600  # 10 minutes
grid_width, grid_height = 100, 100
distances = np.linspace(0, 50, grid_width)
times = np.linspace(0, time_scale, grid_height)

# Directory to save images
output_dir = "heatmap_frames"
os.makedirs(output_dir, exist_ok=True)

# Lambda values for the heatmap sequence


lambda_values = [0.5, 1.0, 2.0]

fig_width, fig_height = 10, 10  # Inches
dpi = 150  # Resolution

# Generate heatmaps for each lambda value
for idx, lambda_val in enumerate(lambda_values):
    print(
        f"Generating heatmap for λ = {lambda_val:.2f} ({idx + 1}/{len(lambda_values)})"
    )
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    P = np.array(
        [
            [prob_function(d, t, lambda_val, time_scale) for d in distances]
            for t in times
        ]
    )

    c = ax.imshow(
        P,
        origin="lower",
        cmap="jet_r",
        extent=[0, 1, 0, time_scale],
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    ax.set_xticks([0, 0.5])
    ax.set_yticks([0, time_scale])

    # Labels and title
    #    ax.set_xlabel("Distance")  # Distance on X-axis
    ax.set_ylabel("Time [min]", fontsize=20)  # Time on Y-axis
    # ax.set_title(f"λ = {lambda_val:.2f}", fontsize=16)

    ax.set_xticklabels(
        [r"$\uparrow$ danger line", r"$\longrightarrow$ Increasing distance"],
        fontsize=20,
        rotation=0,
        ha="left",
    )
    ax.set_yticklabels(
        [0, 10],
        fontsize=20,
        # rotation=90,
        # ha="left",
    )
    cb = fig.colorbar(c, ax=ax, orientation="vertical", label=r"$p$")
    cb.ax.yaxis.label.set_size(18)
    cb.ax.tick_params(labelsize=18)
    filename = f"heatmap_lambda_{lambda_val}.pdf"
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Heatmaps saved in: {filename}")
