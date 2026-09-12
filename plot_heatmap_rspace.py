"""Plot the collapse hazard field P(x) per update, using the model in utils.py.

Usage: python plot_heatmap_rspace.py [config.json]
Parameters (sigma, lambda, tau_line, firing line, n_shooters) are read from the
sweep configuration so the figure matches the simulations. Crowding factor c = 1.
One panel per entry of `times`; with lambda = 0 the field is stationary and a
single panel (t = 0) is drawn.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from shapely import Point

from utils import collapse_hazard, setup_geometry, shooter_positions


config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
with open(config_file) as f:
    config = json.load(f)
sigma = config["sigma"]
lambda_decay = config["lambda_decay_list"][0]
time_scale = config["time_scale"]
firing_line = tuple(map(tuple, config.get("firing_line", [[12, 11], [38, 90]])))
n_shooters = config.get("n_shooters", 50)
tau_line, update_time = config["tau_line"], config["update_time"]
gamma = config["gamma"]
times = [0] if lambda_decay == 0 else [0, time_scale]
p_max = update_time / tau_line * (1 + lambda_decay)  # hazard on the line at t = T
p_min = 0.0
contour_level = 0.5 * update_time / tau_line

walkable_area = setup_geometry()[0]
min_x, min_y, max_x, max_y = walkable_area.bounds
shooters = shooter_positions(firing_line, n_shooters)

nx = ny = 400
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
            Z[i, j] = collapse_hazard(
                pt, t, 0.5, lambda_decay, time_scale, firing_line, sigma, gamma, 0.5,
                tau_line, update_time, n_shooters,
            )
    fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
    im = ax.imshow(
        Z,
        origin="lower",
        extent=(min_x, max_x, min_y, max_y),
        cmap="inferno_r",
        interpolation="bilinear",
        vmin=p_min,
        vmax=p_max,
    )
    fs = 20
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label(f"Collapse probability per {update_time} s", fontsize=fs)
    cbar.set_ticks([p_min, contour_level, p_max])
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    if lambda_decay != 0:
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
    ax.plot(shooters[:, 0], shooters[:, 1], "k.", markersize=3, label="firing line")
    cs = ax.contour(X, Y, Z, levels=[contour_level], colors="black", linewidths=2, linestyles="--")
    # Label the contour with horizontal text just right of its easternmost point
    cx, cy = max(cs.allsegs[0], key=len).T
    ax.text(
        cx.max() + 3, cy[cx.argmax()], f"P = {contour_level:.2f}",
        color="black", fontsize=fs - 4, ha="left", va="center",
    )

    fig.tight_layout()
    out = "hazard_field.pdf" if lambda_decay == 0 else f"rspace_at_time_{t}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)
