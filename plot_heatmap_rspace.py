"""Plot the survival probability field p(x, t) at several times, using the model in utils.py.

Usage: python plot_heatmap_rspace.py [config.json]
Parameters (sigma, lambda, firing line, p_min/p_max, n_shooters) are read from the
sweep configuration so the figure matches the simulations.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from shapely import Point

from utils import calculate_probability, setup_geometry, shooter_positions


class NoNoise:
    """Replaces the rng so the field is drawn without the multiplicative noise."""

    def uniform(self, low, high):
        return 1.0


config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
with open(config_file) as f:
    config = json.load(f)
sigma = config["sigma"]
lambda_decay = config["lambda_decay_list"][0]
time_scale = config["time_scale"]
firing_line = tuple(map(tuple, config.get("firing_line", [[12, 11], [38, 90]])))
n_shooters = config.get("n_shooters", 50)
p_min, p_max = config.get("p_min", 0.05), config.get("p_max", 0.95)
times = [0, 600]
contour_level = 0.5

walkable_area = setup_geometry()[0]
min_x, min_y, max_x, max_y = walkable_area.bounds
shooters = shooter_positions(firing_line, n_shooters)

nx = ny = 500
x = np.linspace(min_x, max_x, nx)
y = np.linspace(min_y, max_y, ny)
X, Y = np.meshgrid(x, y)

for t in times:
    Z = np.full_like(X, np.nan)
    for i in range(ny):
        for j in range(nx):
            pt = Point(X[i, j], Y[i, j])
            if not walkable_area.contains(pt):
                continue
            Z[i, j] = calculate_probability(
                point=pt,
                time_elapsed=t,
                lambda_decay=lambda_decay,
                time_scale=time_scale,
                firing_line=firing_line,
                shielding=0,
                gamma=0,
                alpha=0,
                rng=NoNoise(),
                sigma=sigma,
                p_min=p_min,
                p_max=p_max,
                n_shooters=n_shooters,
            )
    fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
    im = ax.imshow(
        Z,
        origin="lower",
        extent=(min_x, max_x, min_y, max_y),
        cmap="inferno",
        vmin=p_min,
        vmax=p_max,
    )
    fs = 20
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label("Survival Probability", fontsize=fs)
    cbar.set_ticks([p_min, contour_level, p_max])
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    ax.set_title(f"time = {t} s", fontsize=fs)
    ax.set_xlabel("X [m]", fontsize=fs)
    ax.set_ylabel("Y [m]", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))

    x_outer, y_outer = walkable_area.exterior.xy
    ax.plot(x_outer, y_outer, color="black", linewidth=1)
    for interior in walkable_area.interiors:
        x_hole, y_hole = interior.xy
        ax.plot(x_hole, y_hole, color="black", linewidth=1)
    ax.plot(shooters[:, 0], shooters[:, 1], "w.", markersize=3, label="firing line")
    cs = ax.contour(X, Y, Z, levels=[contour_level], colors="white", linewidths=2, linestyles="--")
    ax.clabel(
        cs, fmt=lambda v: f"p = {v:.1f}", fontsize=fs - 4, colors="white", rightside_up=True
    )

    fig.tight_layout()
    fig.savefig(f"rspace_at_time_{t}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"rspace_at_time_{t}.pdf")
