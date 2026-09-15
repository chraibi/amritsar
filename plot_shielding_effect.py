"""Illustrate the dual role of crowding: the factor c(s, alpha) in the collapse weight.

Usage: python plot_shielding_effect.py [config.json]
Crowding factor versus local density for several alpha; each line is labelled at
its right end. Writes shielding_effect.pdf and .png.
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_utils import for_article, save_figure
from utils import crowding_factor

article = for_article()

# --- Data ---
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

# --- Style Setup ---
sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
pal = sns.cubehelix_palette(len(alphas), rot=-0.25, light=0.7)
styles = ["-", "--", "-.", ":", (0, (5, 1))]  # line style encodes alpha a second time

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
for alpha, ls, color in zip(alphas, styles, pal, strict=True):
    y = [crowding_factor(s, gamma=gamma, alpha=alpha) for s in shielding]
    ax.plot(density, y, ls=ls, lw=2, color=color, zorder=3)
    ax.text(density[-1] + 0.05, y[-1], rf"$\alpha = {alpha:.1f}$", va="center", ha="left", fontsize=11, color=color, weight="medium")
ax.axhline(1.0, color="lightgrey", lw=0.8, zorder=1)
ax.axvline(n_max / area, color="lightgrey", lw=0.8, ls=":", zorder=1)
ax.text(n_max / area + 0.03, 1.55, r"$s = 1$", color="dimgrey", ha="left", va="center", fontsize=10)
ax.set_xlabel(r"Local density $\rho$ (persons/m$^2$)", fontsize=12, labelpad=8, color="dimgrey")
ax.set_ylabel(r"Crowding factor $c(s, \alpha)$", fontsize=12, labelpad=8, color="dimgrey")
if not article:
    ax.set_title("Crowding factor versus local density", fontsize=14, loc="left", pad=7, color="dimgrey")
ax.set_xlim(density[0], density[-1] + 0.6)
ax.set_xticks(np.arange(0, density[-1] + 0.01, 0.5))
if not article:
    fig.text(
    0.98, -0.03,
    rf"Dense groups draw {1 + gamma:.1f}× the rounds when targeted ($\alpha = 0$) and {1 - gamma:.1f}× when they shield ($\alpha = 1$)",
    ha="right", va="bottom", fontsize=9, color="dimgrey", style="italic",
)
ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
ax.grid(False)
sns.despine(left=True, bottom=True)
ax.patch.set_edgecolor("lightgrey")
ax.patch.set_linewidth(0.8)

# --- Save ---
print(save_figure(fig, "shielding_effect.pdf"))
