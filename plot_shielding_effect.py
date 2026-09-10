"""Illustrate the dual role of crowding in the collapse probability (Eq. collapse).

Usage: python plot_shielding_effect.py [config.json]
Left: collapse probability versus local density for several alpha, at a fixed
survival probability p. Right: the same over the (density, alpha) plane.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import collapse_probability

config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
with open(config_file) as f:
    config = json.load(f)
gamma = config["gamma"]
crowding_model = config.get("crowding_model", "risk")
radius = config["radius_around"]
n_max = config["n_max"]
p_base = 0.5  # survival probability p(x, t) before the crowding term

area = np.pi * radius**2
density = np.linspace(0, 2.5, 300)  # persons per m^2
shielding = np.minimum(1.0, density * area / n_max)
alphas = [0.0, 0.3, 0.5, 0.7, 1.0]


def p_collapse(s, alpha):
    return collapse_probability(p_base, s, gamma=gamma, alpha=alpha, crowding_model=crowding_model)


fs = 14
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: curves per alpha
styles = ["-", "--", "-.", ":", (0, (5, 1))]
for alpha, ls in zip(alphas, styles, strict=True):
    y = [p_collapse(s, alpha) for s in shielding]
    ax1.plot(density, y, ls=ls, lw=2, color="black", label=rf"$\alpha = {alpha:.1f}$")
ax1.axhline(1 - p_base, color="gray", lw=1, ls="-", alpha=0.5)
ax1.text(density[-1], 1 - p_base, r"$1 - p$", color="gray", ha="right", va="bottom", fontsize=fs - 2)
ax1.axvline(n_max / area, color="gray", lw=1, ls=":")
ax1.text(n_max / area, ax1.get_ylim()[1], r"$s = 1$", color="gray", ha="left", va="top", fontsize=fs - 2)
ax1.set_xlabel(r"Local density $\rho$ [persons/m$^2$]", fontsize=fs)
ax1.set_ylabel(r"Collapse probability $P_\mathrm{collapse}$", fontsize=fs)
ax1.tick_params(labelsize=fs - 2)
ax1.legend(fontsize=fs - 2, frameon=False)
ax1.set_xlim(density[0], density[-1])

# Right: heatmap over (density, alpha)
alpha_grid = np.linspace(0, 1, 201)
Z = np.array([[p_collapse(s, a) for s in shielding] for a in alpha_grid])
im = ax2.imshow(
    Z,
    origin="lower",
    aspect="auto",
    extent=(density[0], density[-1], 0, 1),
    cmap="inferno_r",
)
ax2.contour(density, alpha_grid, Z, levels=[1 - p_base], colors="white", linestyles="--")
ax2.set_xlabel(r"Local density $\rho$ [persons/m$^2$]", fontsize=fs)
ax2.set_ylabel(r"$\alpha$", fontsize=fs)
ax2.tick_params(labelsize=fs - 2)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="4%", pad=0.08)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"$P_\mathrm{collapse}$", fontsize=fs)
cbar.ax.tick_params(labelsize=fs - 2)

fig.tight_layout()
out = f"shielding_effect_{crowding_model}.pdf"
fig.savefig(out, bbox_inches="tight")
print(out)
