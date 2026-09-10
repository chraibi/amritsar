"""Visualise the movement rules: exit choice (Eq. exit_prob), persistence kappa, and
capacity-limited openings.

Usage: python plot_exit_model.py [config.json]
Writes exit_choice_map.pdf, exit_persistence.pdf and exit_capacity.pdf.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely import Point

from utils import exit_capacity_per_update, setup_geometry

config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
with open(config_file) as f:
    config = json.load(f)
beta = config["determinism_strength_exits"]
kappas = config["kappa_list"]
dt, T = config["update_time"], config["time_scale"]
flow, width = config["exit_flow_rate"], config["exit_width"]
crowd_sizes = config["num_agents_list"]

walkable_area, exit_areas, _ = setup_geometry()
min_x, min_y, max_x, max_y = walkable_area.bounds
fs = 14

# ---------- 1. Probability of heading for the nearest opening, over the Bagh
nx = ny = 300
xs = np.linspace(min_x, max_x, nx)
ys = np.linspace(min_y, max_y, ny)
X, Y = np.meshgrid(xs, ys)
Z = np.full_like(X, np.nan)
for i in range(ny):
    for j in range(nx):
        pt = Point(X[i, j], Y[i, j])
        if not walkable_area.contains(pt):
            continue
        d = np.array([pt.distance(e) for e in exit_areas])
        p = (d + 1e-6) ** -beta
        Z[i, j] = p.max() / p.sum()
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(Z, origin="lower", extent=(min_x, max_x, min_y, max_y), cmap="viridis", vmin=0.2, vmax=1.0)
x_outer, y_outer = walkable_area.exterior.xy
ax.plot(x_outer, y_outer, color="black", lw=1)
for interior in walkable_area.interiors:
    ax.plot(*interior.xy, color="black", lw=1)
for e in exit_areas:
    ax.plot(e.centroid.x, e.centroid.y, marker="s", color="red", ms=8)
ax.set_xlabel("X [m]", fontsize=fs)
ax.set_ylabel("Y [m]", fontsize=fs)
ax.tick_params(labelsize=fs - 2)
cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(rf"Probability of choosing the nearest opening ($\beta = {beta:g}$)", fontsize=fs - 2)
cbar.ax.tick_params(labelsize=fs - 2)
fig.tight_layout()
fig.savefig("exit_choice_map.pdf", bbox_inches="tight")
plt.close(fig)
print("exit_choice_map.pdf")

# ---------- 2. Persistence: how long a target is held, and how often it changes
kap = np.linspace(0, 0.99, 200)
hold = dt / (1 - kap)  # mean holding time (s)
changes = (T / dt) * (1 - kap)  # expected re-decisions over the event
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(kap, hold, color="black", lw=2)
ax.set_xlabel(r"Persistence $\kappa$", fontsize=fs)
ax.set_ylabel("Mean time a target is kept [s]", fontsize=fs)
ax.set_yscale("log")
ax.set_ylim(dt, T)
ax.grid(alpha=0.3, which="both")
for k in kappas:
    ax.plot(k, dt / (1 - k), "o", color="black", ms=8)
    ax.annotate(
        rf"$\kappa = {k}$: {dt / (1 - k):.0f} s, ~{(T / dt) * (1 - k):.0f} changes in {T} s",
        (k, dt / (1 - k)), textcoords="offset points", xytext=(-10, 12), ha="right", fontsize=fs - 3,
    )
ax.tick_params(labelsize=fs - 2)
fig.tight_layout()
fig.savefig("exit_persistence.pdf", bbox_inches="tight")
plt.close(fig)
print("exit_persistence.pdf")

# ---------- 3. Capacity: the most people that can leave through the five openings
cap = exit_capacity_per_update(flow, width, dt) * len(exit_areas) / dt  # persons per s
t = np.linspace(0, T, 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, cap * t, color="black", lw=2, label=rf"{len(exit_areas)} openings, $J = {flow}$ /m/s, $w = {width}$ m")
for n in crowd_sizes:
    ax.axhline(n, color="gray", ls="--", lw=1)
    ax.text(5, n, f"N = {n}", va="bottom", ha="left", fontsize=fs - 3, color="gray")
ax.set_xlabel("Time [s]", fontsize=fs)
ax.set_ylabel("Maximum number of people that can have left", fontsize=fs)
ax.set_xlim(0, T)
ax.set_ylim(0, max(crowd_sizes) * 1.05)
ax.tick_params(labelsize=fs - 2)
ax.grid(alpha=0.3)
ax.legend(fontsize=fs - 2, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.9))
fig.tight_layout()
fig.savefig("exit_capacity.pdf", bbox_inches="tight")
plt.close(fig)
print(f"exit_capacity.pdf  (max outflow {cap:.1f}/s, {cap * T:.0f} in {T} s)")
