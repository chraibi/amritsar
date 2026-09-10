"""Cumulative fallen agents over time from a sweep pickle (mean and std over repetitions).

Usage: python plot_fallen_time_series.py <pickle> [--vary alpha|kappa]
One figure per crowd size and per value of the parameter that is not varied.
Sweep keys are (num_agents, lambda_decay, alpha, kappa).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import load_results

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pickle")
parser.add_argument("--vary", choices=["alpha", "kappa"], default="alpha")
args = parser.parse_args()

data, stem, output_dir = load_results(["", args.pickle])
fallen_time_series = data["fallen_time_series"]
max_time = data["config"]["time_scale"]
keys = sorted(fallen_time_series)
idx = {"alpha": 2, "kappa": 3}
vary, fixed = args.vary, ("kappa" if args.vary == "alpha" else "alpha")
symbol = {"alpha": r"\alpha", "kappa": r"\kappa"}
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len({k[idx[vary]] for k in keys})))


def cumulative_on_grid(time_series, fallen_series):
    """Cumulative fallen count resampled on a 1 s grid up to max_time."""
    cumulative = np.cumsum(fallen_series)
    grid = np.arange(0, max_time + 1)
    return np.interp(grid, time_series, cumulative, right=cumulative[-1]), grid


for num_agents in sorted({k[0] for k in keys}):
    for fixed_value in sorted({k[idx[fixed]] for k in keys if k[0] == num_agents}):
        fig, ax = plt.subplots(figsize=(10, 6))
        group = [k for k in keys if k[0] == num_agents and k[idx[fixed]] == fixed_value]
        for color, key in zip(colors, sorted(group, key=lambda k: k[idx[vary]]), strict=False):
            times_list, fallen_list = fallen_time_series[key]
            runs = []
            for t, f in zip(times_list, fallen_list, strict=True):
                series, grid = cumulative_on_grid(t, f)
                runs.append(series)
                ax.plot(grid, series, color=color, alpha=0.25, lw=0.8)
            runs = np.array(runs)
            mean, std = runs.mean(axis=0), runs.std(axis=0)
            ax.plot(
                grid, mean, color=color, lw=3,
                label=rf"${symbol[vary]} = {key[idx[vary]]:.1f}$  (mean: {mean[-1]:.0f} $\pm$ {std[-1]:.0f})",
            )
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_xlabel("Time [s]", fontsize=18)
        ax.set_ylabel("Cumulative fallen agents", fontsize=18)
        ax.set_xlim(0, max_time)
        ax.grid(alpha=0.4, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=16)
        ax.legend(
            fontsize=14, loc="upper left", framealpha=0.9,
            title=rf"$N = {num_agents}$, ${symbol[fixed]} = {fixed_value:.1f}$", title_fontsize=14,
        )
        out = Path(output_dir) / f"{stem}_fallen_time_series_N{num_agents}_{fixed}{fixed_value:.1f}_vary_{vary}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f">> Saved: {out}")
