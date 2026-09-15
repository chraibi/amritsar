"""Visualise the movement rules: exit choice (Eq. exit_prob), persistence kappa, and
capacity-limited openings.

Usage: python plot_exit_model.py [config.json]
Writes exit_choice_map and exit_persistence as .pdf and .png.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from shapely import Point, contains_xy

from plot_utils import save_figure
from utils import setup_geometry

# --- Data ---
config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
with open(config_file) as f:
    config = json.load(f)
beta = config["exit_choice_exponent"]
kappas = config["kappa_list"]
dt, T = config["update_time"], config["time_scale"]

walkable_area, exit_areas, _ = setup_geometry(config.get("extra_exits", []))
min_x, min_y, max_x, max_y = walkable_area.bounds

# --- Style Setup ---
sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
cmap = sns.cubehelix_palette(rot=-0.25, light=0.9, as_cmap=True)

# ---------- 1. Probability of heading for the nearest opening, over the Bagh
nx = ny = 300
xs = np.linspace(min_x, max_x, nx)
ys = np.linspace(min_y, max_y, ny)
X, Y = np.meshgrid(xs, ys)
inside = contains_xy(walkable_area, X, Y)
Z = np.full_like(X, np.nan)
for i, j in zip(*np.nonzero(inside), strict=True):
    d = np.array([Point(X[i, j], Y[i, j]).distance(e) for e in exit_areas])
    p = (d + 1e-6) ** -beta
    Z[i, j] = p.max() / p.sum()

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
im = ax.imshow(Z, origin="lower", extent=(min_x, max_x, min_y, max_y), cmap=cmap, vmin=0.2, vmax=1.0)
bx, by = walkable_area.exterior.xy
ax.plot(bx, by, color="dimgrey", lw=1.2)
for hole in walkable_area.interiors:
    ax.fill(*hole.xy, color="lightgrey", lw=0)
for e in exit_areas:
    ax.plot(e.centroid.x, e.centroid.y, marker="s", ms=7, mfc="white", mec="dimgrey", mew=1.2, zorder=6)
ax.set_aspect("equal")
ax.set_xlim(min_x - 5, max_x + 5)
ax.set_ylim(min_y - 5, max_y + 5)
ax.set_xlabel("x (m)", fontsize=12, labelpad=8, color="dimgrey")
ax.set_ylabel("y (m)", fontsize=12, labelpad=8, color="dimgrey")
ax.set_title(rf"Probability of heading for the nearest opening, $\beta$ = {beta:g}", fontsize=14, loc="left", pad=7, color="dimgrey")
fig.text(
    0.98, -0.03, f"White squares: the {len(exit_areas)} openings; in the centre the choice is close to even, {np.nanmin(Z):.2f} for the nearest",
    ha="right", va="bottom", fontsize=9, color="dimgrey", style="italic",
)
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Probability of the nearest opening", color="dimgrey")
cbar.ax.tick_params(length=0, labelcolor="dimgrey")
cbar.outline.set_visible(False)
ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
ax.grid(False)
sns.despine(left=True, bottom=True)
print(save_figure(fig, "exit_choice_map.pdf"))
plt.close(fig)

# ---------- 2. Persistence: how long a target is held, and how often it changes
kap = np.linspace(0, 0.99, 200)
hold = dt / (1 - kap)  # mean holding time (s)
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.plot(kap, hold, color=pal[5], lw=2.5, zorder=3)
ax.set_xlabel(r"Persistence $\kappa$", fontsize=12, labelpad=8, color="dimgrey")
ax.set_ylabel("Mean time a target is kept (s)", fontsize=12, labelpad=8, color="dimgrey")
ax.set_title("Persistence of the chosen opening", fontsize=14, loc="left", pad=7, color="dimgrey")
ax.set_yscale("log")
ax.set_ylim(dt, T)
for k in kappas:
    if k >= 1:  # never reconsiders: holding time is the whole event
        ax.scatter(k, T, s=60, color=pal[2], edgecolors="white", zorder=4)
        ax.annotate(r"$\kappa = 1$: keeps the first choice", (k, T), textcoords="offset points", xytext=(-10, -14), ha="right", fontsize=10, color="dimgrey")
        continue
    ax.scatter(k, dt / (1 - k), s=60, color=pal[2], edgecolors="white", zorder=4)
    ax.annotate(
        rf"$\kappa = {k}$: {dt / (1 - k):.0f} s, ~{(T / dt) * (1 - k):.0f} changes in {T} s",
        (k, dt / (1 - k)), textcoords="offset points", xytext=(-10, 12), ha="right", fontsize=10, color="dimgrey",
    )
ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
ax.grid(False)
ax.grid(axis="y", which="major", alpha=0.7, linewidth=1)
sns.despine(left=True, bottom=True)
ax.patch.set_edgecolor("lightgrey")
ax.patch.set_linewidth(0.8)
print(save_figure(fig, "exit_persistence.pdf"))
plt.close(fig)
