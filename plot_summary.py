"""One-figure summary of every sweep: simulated collapses against the historical numbers.

Usage: python plot_summary.py <results_dir> [--official 379] [--estimates 1000 1500] [--rounds 1650]
Reads <results_dir>/report.csv (written by make_report.py) and writes
<results_dir>/summary.pdf and .png. One row per sweep and crowd size; one marker
per alpha (colour and shape), error bars are one standard deviation over
repetitions, kappa = 0.9 above and kappa = 0.5 below the row centre.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from plot_utils import for_article, save_figure

article = for_article()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results_dir")
parser.add_argument("--official", type=int, default=379)
parser.add_argument("--estimates", type=int, nargs=2, default=[1000, 1500])
parser.add_argument("--rounds", type=int, default=1650)
args = parser.parse_args()

# --- Data ---
with open(Path(args.results_dir) / "report.csv") as f:
    rows = list(csv.DictReader(f))
labels = {
    "main": "reference model",
    "hits_0p5": "0.5 hits per round",
    "hits_1p5": "1.5 hits per round",
    "open_gates_w3": "openings 3 m",
    "open_gates_w4": "openings 4 m",
    "sixth_door": "north door open",
    "kappa_extremes": r"$\kappa = 0$ and $1$",
    "alpha_extremes": r"$\alpha = 0$ and $1$",
    "exit_zone_5": "exit zone 5 m",
    "exit_zone_15": "exit zone 15 m",
    "n20000": "reference model",
    "sigma_20": r"$\sigma = 20$ m",
    "sigma_40": r"$\sigma = 40$ m",
}
order = [r for r in ["main", "n20000", "hits_0p5", "hits_1p5", "kappa_extremes", "alpha_extremes", "exit_zone_5", "exit_zone_15", "sigma_20", "sigma_40", "open_gates_w3", "open_gates_w4", "sixth_door"] if r in {row["run"] for row in rows}]
order += sorted({row["run"] for row in rows} - set(order))
labels = {k: labels.get(k, k) for k in order}
groups = []
for run in order:
    for n in sorted({int(r["N"]) for r in rows if r["run"] == run}):
        groups.append((run, n))
lowest = min(rows, key=lambda r: float(r["fallen_mean"]))
highest = max(rows, key=lambda r: float(r["fallen_mean"]))

# --- Style Setup ---
sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
style = {  # alpha -> (colour, marker): meaning encoded twice
    "targeted": (pal[5], "o"),
    "protective": (pal[2], "s"),
}

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 0.42 * len(groups) + 2.2), dpi=150)
for y, (run, n) in enumerate(groups):
    for r in rows:
        if r["run"] != run or int(r["N"]) != n:
            continue
        alpha, kappa = float(r["alpha"]), float(r["kappa"])
        mean, std = float(r["fallen_mean"]), float(r["fallen_std"])
        dy = 0.16 if kappa > 0.7 else -0.16
        color, marker = style["targeted"] if alpha < 0.5 else style["protective"]
        ax.errorbar(mean, y + dy, xerr=std, fmt=marker, ms=7, color=color, mec="white", mew=0.6, ecolor=color, capsize=2, lw=1, zorder=3)
ax.set_yticks(range(len(groups)))
ax.set_yticklabels([f"{labels[run]}, $N$ = {n:,}".replace(",", " ") for run, n in groups], fontsize=9.5)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(250, 14000)
ax.set_xlabel("Collapsed agents (log scale)", fontsize=12, labelpad=8, color="dimgrey")
if not article:
    ax.set_title("Simulated collapses against the historical numbers", fontsize=14, loc="left", pad=7, color="dimgrey")

ymax = len(groups) - 0.5
box = dict(facecolor="white", edgecolor="none", pad=1.5)
ax.axvline(args.official, color="#bd0c0c", lw=1.5, zorder=1)
ax.text(args.official / 1.06, -0.8, f"official\n{args.official}", color="#bd0c0c", ha="right", va="bottom", fontsize=9, bbox=box)
ax.axvspan(args.estimates[0], args.estimates[1], color="#4575b4", alpha=0.15, lw=0, zorder=0)
ax.text(args.estimates[0] / 1.04, -0.8, f"Indian estimates\n{args.estimates[0]}–{args.estimates[1]}", color="#4575b4", ha="right", va="bottom", fontsize=9, bbox=box)
ax.axvline(args.rounds, color="grey", lw=1.5, ls="--", zorder=1)
ax.text(args.rounds * 1.05, -0.8, f"rounds fired\n{args.rounds}", color="grey", ha="left", va="bottom", fontsize=9, bbox=box)
ax.set_ylim(ymax, -1.6)

handles = [
    Line2D([], [], marker=style["targeted"][1], color=style["targeted"][0], ls="", label=r"$\alpha = 0.3$, or $0$ in the $\alpha$ run (crowds targeted)"),
    Line2D([], [], marker=style["protective"][1], color=style["protective"][0], ls="", label=r"$\alpha = 0.7$, or $1$ in the $\alpha$ run (crowds protect)"),
]
ax.legend(
    handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10,
    frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey", labelcolor="dimgrey",
    title=r"per row: $\kappa = 0.9$ (or $1$) above, $\kappa = 0.5$ (or $0$) below", title_fontsize=9,
)
ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
ax.grid(False)
ax.grid(axis="x", which="major", alpha=0.7, linewidth=1)
sns.despine(left=True, bottom=True)

# --- Insight annotation ---
def describe(r):
    n_label = f"{int(r['N']):,}".replace(",", " ")
    return f"{float(r['fallen_mean']):.0f} ({labels[r['run']]}, $N$ = {n_label})"


if not article:
    fig.text(
        0.98, -0.01,
        f"Every run exceeds the official toll: lowest {describe(lowest)}, highest {describe(highest)}",
        ha="right", va="bottom", fontsize=9, color="dimgrey", style="italic",
    )

# --- Save ---
print(save_figure(fig, Path(args.results_dir) / "summary.pdf"))
