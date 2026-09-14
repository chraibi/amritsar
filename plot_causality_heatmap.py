"""Maps of where the agents fell, one per (N, alpha, kappa) of a sweep.

Usage: python plot_causality_heatmap.py <sweep pickle>

Positions are binned on 1 m cells, averaged over the repetitions, smoothed with a
Gaussian kernel of 2 m and rescaled to agents per 3 m x 3 m, then drawn on a
square-root colour scale with a common upper limit per crowd size. The smoothing
shows the diffuse hits in the interior next to the dense queues at the openings.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from shapely import contains_xy

from plot_utils import load_results
from utils import setup_geometry, shooter_positions

SMOOTH_SIGMA = 2.0  # m
REPORT_CELL = 3.0  # m, colourbar unit


def gaussian_smooth(grid, sigma):
    """Separable Gaussian blur of a 2-D array with reflecting borders (numpy only)."""
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    def blur_axis(a, axis):
        padded = np.pad(a, [(radius, radius) if i == axis else (0, 0) for i in range(2)], mode="reflect")
        return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), axis, padded)

    return blur_axis(blur_axis(grid, 0), 1)


def density_grid(fallen_positions, x_edges, y_edges):
    """Smoothed mean number of fallen agents per 3 m x 3 m over the runs."""
    counts = np.zeros((len(x_edges) - 1, len(y_edges) - 1))
    for run in fallen_positions:
        if len(run) == 0:
            continue
        pts = np.asarray(run)
        counts += np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges])[0]
    counts /= max(len(fallen_positions), 1)
    return gaussian_smooth(counts, SMOOTH_SIGMA) * REPORT_CELL**2


def plot_fallen_map(walkable, exits, shooters, density, x_edges, y_edges, inside, vmax, output_file):
    """Draw one smoothed map of the fallen agents over the geometry."""
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    masked = np.ma.masked_where(~inside | (density < 0.05), density)
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        masked.T,
        cmap="inferno_r",
        norm=PowerNorm(0.5, vmin=0, vmax=vmax),
        shading="flat",
        rasterized=True,
    )

    bx, by = walkable.exterior.xy
    ax.plot(bx, by, color="dimgrey", lw=1.2)
    for hole in walkable.interiors:
        hx, hy = hole.xy
        ax.fill(hx, hy, color="lightgrey", lw=0)
    ax.scatter(shooters[:, 0], shooters[:, 1], s=8, color="#bd0c0c", zorder=5)
    for exit_area in exits:
        c = exit_area.centroid
        ax.plot(c.x, c.y, marker="s", ms=7, mfc="white", mec="dimgrey", mew=1.2, zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(x_edges[0] - 5, x_edges[-1] + 5)
    ax.set_ylim(y_edges[0] - 5, y_edges[-1] + 5)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    ax.set_xlabel("x [m]", color="dimgrey")
    ax.set_ylabel("y [m]", color="dimgrey")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f"Collapsed agents per {REPORT_CELL:.0f} m × {REPORT_CELL:.0f} m", color="dimgrey")
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()
    cbar.ax.tick_params(length=0, labelcolor="dimgrey")
    cbar.outline.set_visible(False)
    sns.despine(left=True, bottom=True)

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


loaded_data, stem, output_dir = load_results()
config = loaded_data["config"]
results = loaded_data["results"]

walkable, exits, _ = setup_geometry(extra_exits=config.get("extra_exits", ()))
shooters = shooter_positions(config["firing_line"], config["n_shooters"])

min_x, min_y, max_x, max_y = walkable.bounds
x_edges = np.arange(np.floor(min_x), np.ceil(max_x) + 1, 1.0)
y_edges = np.arange(np.floor(min_y), np.ceil(max_y) + 1, 1.0)
cx, cy = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2, indexing="ij")
inside = contains_xy(walkable, cx, cy)

# Common colour limit per crowd size so alpha/kappa panels are comparable
densities = {key: density_grid(positions, x_edges, y_edges) for key, positions in results.items()}
vmax_by_n = {}
for (num_agents, *_), density in densities.items():
    vmax_by_n[num_agents] = max(vmax_by_n.get(num_agents, 0), density[inside].max())

for (num_agents, _lambda_decay, alpha, kappa), density in densities.items():
    folder = Path(output_dir) / f"N_{num_agents}"
    folder.mkdir(parents=True, exist_ok=True)
    heatmap_file = folder / f"{stem}_causality_alpha_{alpha}_kappa_{kappa}_N_{num_agents}.pdf"
    plot_fallen_map(walkable, exits, shooters, density, x_edges, y_edges, inside, vmax_by_n[num_agents], heatmap_file)
    print(f">> Saved heatmap: {heatmap_file}")
