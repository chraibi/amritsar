"""Illustrate the dual role of crowding: the factor c(s, alpha) in the collapse hazard.

Usage: python plot_shielding_effect.py [config.json]
Crowding factor versus local density for several alpha; each line is labelled at
its right end.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from utils import crowding_factor

config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
with open(config_file) as f:
    config = json.load(f)
gamma = config["gamma"]
radius = config["radius_around"]
n_max = config["n_max"]

area = np.pi * radius**2
density = np.linspace(0, 2.5, 300)  # persons per m^2
shielding = np.minimum(1.0, density * area / n_max)
alphas = [0.0, 0.3, 0.5, 0.7, 1.0]


def p_collapse(s, alpha):
    return crowding_factor(s, gamma=gamma, alpha=alpha)


fs = 14
fig, ax1 = plt.subplots(figsize=(8, 5))

styles = ["-", "--", "-.", ":", (0, (5, 1))]
for alpha, ls in zip(alphas, styles, strict=True):
    y = [p_collapse(s, alpha) for s in shielding]
    ax1.plot(density, y, ls=ls, lw=2, color="black")
    ax1.text(density[-1] + 0.05, y[-1], rf"$\alpha = {alpha:.1f}$", va="center", ha="left", fontsize=fs - 2)
ax1.axhline(1.0, color="gray", lw=1, ls="-", alpha=0.5)
ax1.axvline(n_max / area, color="gray", lw=1, ls=":")
ax1.text(n_max / area + 0.03, 1.55, r"$s = 1$", color="gray", ha="left", va="center", fontsize=fs - 2)
ax1.set_xlabel(r"Local density $\rho$ [persons/m$^2$]", fontsize=fs)
ax1.set_ylabel(r"Crowding factor $c(s, \alpha)$", fontsize=fs)
ax1.tick_params(labelsize=fs - 2)
ax1.set_xlim(density[0], density[-1] + 0.6)
ax1.set_xticks(np.arange(0, density[-1] + 0.01, 0.5))
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("shielding_effect.pdf", bbox_inches="tight")
print("shielding_effect.pdf")
