"""Plot the spatial exposure r_space(x) over the Bagh, using the model in utils.py.

Usage: python plot_heatmap_rspace.py [config.json]
Parameters (sigma, firing line, n_shooters) are read from the sweep configuration
so the figure matches the simulations. The exposure is normalised to one on the
firing line; the dashed contour marks half that value. Writes hazard_field.pdf
and .png.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from shapely import Point, contains_xy

from plot_utils import for_article, save_figure
from utils import exposure_factor, setup_geometry, shooter_positions

article = for_article()

# --- Data ---
config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
with open(config_file) as f:
    config = json.load(f)
sigma = config["sigma"]
firing_line = tuple(map(tuple, config.get("firing_line", [[12, 11], [38, 90]])))
n_shooters = config.get("n_shooters", 50)
contour_level = 0.5

walkable_area = setup_geometry()[0]
min_x, min_y, max_x, max_y = walkable_area.bounds
shooters = shooter_positions(firing_line, n_shooters)

nx = ny = 400
x = np.linspace(min_x, max_x, nx)
y = np.linspace(min_y, max_y, ny)
X, Y = np.meshgrid(x, y)
inside = contains_xy(walkable_area, X, Y)
Z = np.full_like(X, np.nan)
for i, j in zip(*np.nonzero(inside), strict=True):
    Z[i, j] = exposure_factor(Point(X[i, j], Y[i, j]), firing_line, sigma, n_shooters)

# --- Style Setup ---
sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
im = ax.imshow(Z, origin="lower", extent=(min_x, max_x, min_y, max_y), cmap="inferno_r", interpolation="bilinear", vmin=0, vmax=1)
bx, by = walkable_area.exterior.xy
ax.plot(bx, by, color="dimgrey", lw=1.2)
for hole in walkable_area.interiors:
    ax.fill(*hole.xy, color="lightgrey", lw=0)
ax.scatter(shooters[:, 0], shooters[:, 1], s=8, color="#bd0c0c", zorder=5)
cs = ax.contour(X, Y, Z, levels=[contour_level], colors="dimgrey", linewidths=1.5, linestyles="--")
cx, cy = max(cs.allsegs[0], key=len).T  # label just right of the easternmost point
ax.text(cx.max() + 3, cy[cx.argmax()], rf"$r_\mathrm{{space}} = {contour_level:.1f}$", color="dimgrey", fontsize=10, ha="left", va="center")

ax.set_aspect("equal")
ax.set_xlim(min_x - 5, max_x + 5)
ax.set_ylim(min_y - 5, max_y + 5)
ax.set_xlabel("x (m)", fontsize=12, labelpad=8, color="dimgrey")
ax.set_ylabel("y (m)", fontsize=12, labelpad=8, color="dimgrey")
if not article:
    ax.set_title(rf"Spatial exposure to the firing line, $\sigma$ = {sigma:g} m", fontsize=14, loc="left", pad=7, color="dimgrey")
    fig.text(
        0.98, -0.03, f"Red dots: the {n_shooters} shooters; exposure falls to about {np.nanmin(Z):.2f} at the far wall",
        ha="right", va="bottom", fontsize=9, color="dimgrey", style="italic",
    )

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0, contour_level, 1])
cbar.set_label(r"Spatial exposure $r_\mathrm{space}$ (1 on the firing line)", color="dimgrey")
cbar.ax.tick_params(length=0, labelcolor="dimgrey")
cbar.outline.set_visible(False)
ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
ax.grid(False)
sns.despine(left=True, bottom=True)

# --- Save ---
print(save_figure(fig, "hazard_field.pdf"))
