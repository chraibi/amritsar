"""Cumulative fallen agents over time from a sweep pickle (mean and std over repetitions).

Usage: python plot_fallen_time_series.py <pickle> [--vary alpha|kappa]
One figure per crowd size and per value of the parameter that is not varied.
Sweep keys are (num_agents, lambda_decay, alpha, kappa).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_utils import load_results, save_figure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pickle")
parser.add_argument("--vary", choices=["alpha", "kappa"], default="alpha")
args = parser.parse_args()

# --- Data ---
data, stem, output_dir = load_results(["", args.pickle])
fallen_time_series = data["fallen_time_series"]
max_time = data["config"]["time_scale"]
update_time = data["config"]["update_time"]
keys = sorted(fallen_time_series)
idx = {"alpha": 2, "kappa": 3}
vary, fixed = args.vary, ("kappa" if args.vary == "alpha" else "alpha")
symbol = {"alpha": r"\alpha", "kappa": r"\kappa"}

# --- Style Setup ---
sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
n_values = len({k[idx[vary]] for k in keys})
# Two values: the diverging pair; more: cubehelix. Line styles encode the value a second time.
colors = ["#4575b4", "#d73027"] if n_values == 2 else sns.cubehelix_palette(n_values, rot=-0.25, light=0.7)
linestyles = ["-", "--", "-.", ":"]


def cumulative_on_grid(time_series, fallen_series):
    """Cumulative fallen count resampled on a 1 s grid up to max_time."""
    cumulative = np.cumsum(fallen_series)
    grid = np.arange(0, max_time + 1)
    return np.interp(grid, time_series, cumulative, right=cumulative[-1]), grid


def insight(means):
    """One sentence on the time course: the rate, and when the Bagh empties if it does."""
    rate = np.mean([(m[300] - m[0]) / 300 * update_time for m in means])
    reached = [np.argmax(m >= 0.99 * m[-1]) for m in means]
    if max(reached) >= 0.95 * max_time:
        return f"≈ {rate:.0f} collapses per {update_time} s until the firing stops"
    when = f"{min(reached):.0f} s" if max(reached) - min(reached) < 20 else f"{min(reached):.0f} to {max(reached):.0f} s"
    return f"≈ {rate:.0f} collapses per {update_time} s; the Bagh is empty after ≈ {when}"


# --- Plot ---
for num_agents in sorted({k[0] for k in keys}):
    for fixed_value in sorted({k[idx[fixed]] for k in keys if k[0] == num_agents}):
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        group = [k for k in keys if k[0] == num_agents and k[idx[fixed]] == fixed_value]
        means = []
        for i, key in enumerate(sorted(group, key=lambda k: k[idx[vary]])):
            color, ls = colors[i], linestyles[i % len(linestyles)]
            times_list, fallen_list = fallen_time_series[key]
            runs = []
            for t, f in zip(times_list, fallen_list, strict=True):
                series, grid = cumulative_on_grid(t, f)
                runs.append(series)
                ax.plot(grid, series, color=color, alpha=0.3, lw=0.7, zorder=2)
            runs = np.array(runs)
            mean, std = runs.mean(axis=0), runs.std(axis=0)
            means.append(mean)
            ax.plot(
                grid, mean, color=color, ls=ls, lw=2.5, zorder=4,
                label=rf"${symbol[vary]} = {key[idx[vary]]:.1f}$  ({mean[-1]:.0f} $\pm$ {std[-1]:.0f})",
            )
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15, lw=0, zorder=1)
        ax.set_xlabel("Time (s)", fontsize=12, labelpad=8, color="dimgrey")
        ax.set_ylabel("Cumulative collapsed agents", fontsize=12, labelpad=8, color="dimgrey")
        n_label = f"{num_agents:,}".replace(",", " ")
        ax.set_title(
            rf"Collapses over time, $N$ = {n_label}, ${symbol[fixed]} = {fixed_value:.1f}$",
            fontsize=14, loc="left", pad=7, color="dimgrey",
        )
        ax.set_xlim(0, max_time)
        ax.set_ylim(bottom=0)
        ax.legend(
            loc="lower right", fontsize=10, title="mean ± std at the end", title_fontsize=9,
            frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey", labelcolor="dimgrey",
        )
        ax.text(0.02, 0.97, insight(means), transform=ax.transAxes, ha="left", va="top", fontsize=9, color="dimgrey", style="italic")
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
        ax.grid(False)
        sns.despine(left=True, bottom=True)
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)

        # --- Save ---
        out = Path(output_dir) / f"{stem}_fallen_time_series_N{num_agents}_{fixed}{fixed_value:.1f}_vary_{vary}.pdf"
        print(f">> Saved: {save_figure(fig, out)}")
        plt.close(fig)
