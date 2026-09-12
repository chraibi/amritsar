"""Calibration of the hazard model: survival of a stationary agent over the event.

Usage: python plot_hazard_calibration.py [config.json]
Plots the probability that an agent who does not move survives the whole event,
as a function of its distance from the firing line, for several values of
tau_line (mean time to collapse on the firing line). Crowding factor c = 1
(alpha = 0.5); the band shows c in [1 - gamma, 1 + gamma].
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from shapely import LineString, Point

from utils import collapse_hazard, setup_geometry

config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
with open(config_file) as f:
    config = json.load(f)
firing_line = tuple(map(tuple, config["firing_line"]))
sigma, gamma = config["sigma"], config["gamma"]
lam = config["lambda_decay_list"][0]
T, dt = config["time_scale"], config["update_time"]
n_shooters = config.get("n_shooters", 50)
taus = [30, 60, 120, 240]

walkable_area = setup_geometry()[0]
line = LineString(firing_line)
# Probe points along a horizontal transect at mid-height of the firing line
y0 = 0.5 * (firing_line[0][1] + firing_line[1][1])
xs = np.arange(0, 220, 2.0)
probes = [Point(x, y0) for x in xs if walkable_area.contains(Point(x, y0))]
dist = np.array([line.distance(p) for p in probes])


def survival_600(p, tau, shielding=0.5, alpha=0.5):
    keep = 1.0
    for t in range(0, T, dt):
        keep *= 1 - collapse_hazard(
            p, t, shielding, lam, T, firing_line, sigma, gamma, alpha, tau, dt, n_shooters
        )
    return keep


fs = 14
fig, ax = plt.subplots(figsize=(8, 5))
styles = ["-", "--", "-.", ":"]
for tau, ls in zip(taus, styles, strict=True):
    mid = np.array([survival_600(p, tau) for p in probes])
    lo = np.array([survival_600(p, tau, shielding=0, alpha=1.0) for p in probes])  # c = 1 + gamma
    hi = np.array([survival_600(p, tau, shielding=1, alpha=1.0) for p in probes])  # c = 1 - gamma
    ax.plot(dist, mid, ls=ls, lw=2, color="black", label=rf"$\tau = {tau}$ s")
    ax.fill_between(dist, lo, hi, color="gray", alpha=0.15)
ax.set_xlabel("Distance from the firing line [m]", fontsize=fs)
ax.set_ylabel(f"Survival of a stationary agent over {T} s", fontsize=fs)
ax.set_ylim(0, 1)
ax.set_xlim(0, dist.max())
ax.tick_params(labelsize=fs - 2)
ax.grid(alpha=0.3)
ax.legend(fontsize=fs - 2, frameon=False, title=rf"$\lambda = {lam}$, $\sigma = {sigma}$ m")
fig.tight_layout()
fig.savefig("hazard_calibration.pdf", bbox_inches="tight")
print("hazard_calibration.pdf")
