"""One-figure summary of every sweep: simulated collapses against the historical numbers.

Usage: python plot_summary.py <results_dir> [--official 379] [--estimates 1000 1500] [--rounds 1650]
Reads <results_dir>/report.csv (written by make_report.py) and writes
<results_dir>/summary.pdf. One row per sweep and crowd size; markers per alpha,
error bars are one standard deviation over repetitions (both kappa values shown).
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results_dir")
parser.add_argument("--official", type=int, default=379)
parser.add_argument("--estimates", type=int, nargs=2, default=[1000, 1500])
parser.add_argument("--rounds", type=int, default=1650)
args = parser.parse_args()

with open(Path(args.results_dir) / "report.csv") as f:
    rows = list(csv.DictReader(f))
labels = {
    "main": "reference model",
    "rate_half": "half the rate of fire",
    "hits_1p5": "1.5 hits per round",
    "open_gates_w3": "openings 3 m",
    "open_gates_w4": "openings 4 m",
    "sixth_door": "north door open",
    "kappa_extremes": r"$\kappa = 0$ and $1$",
    "exit_zone_5": "exit zone 5 m",
    "exit_zone_15": "exit zone 15 m",
    "n20000": "reference model",
    "sigma_20": r"$\sigma = 20$ m",
    "sigma_40": r"$\sigma = 40$ m",
}
order = [r for r in ["main", "n20000", "rate_half", "hits_1p5", "kappa_extremes", "exit_zone_5", "exit_zone_15", "sigma_20", "sigma_40", "open_gates_w3", "open_gates_w4", "sixth_door"] if r in {row["run"] for row in rows}]
order += sorted({row["run"] for row in rows} - set(order))
labels = {k: labels.get(k, k) for k in order}
groups = []
for run in order:
    for n in sorted({int(r["N"]) for r in rows if r["run"] == run}):
        groups.append((run, n))

fs = 13
fig, ax = plt.subplots(figsize=(9, 0.5 * len(groups) + 1.8))
markers = {0.3: ("o", "black", "filled"), 0.7: ("o", "black", "open")}
for y, (run, n) in enumerate(groups):
    for r in rows:
        if r["run"] != run or int(r["N"]) != n:
            continue
        alpha, kappa = float(r["alpha"]), float(r["kappa"])
        mean, std = float(r["fallen_mean"]), float(r["fallen_std"])
        dy = 0.14 if kappa > 0.7 else -0.14
        face = "black" if alpha < 0.5 else "white"
        ax.errorbar(mean, y + dy, xerr=std, fmt="o", ms=7, mfc=face, mec="black", ecolor="black", capsize=2, lw=1)
ax.set_yticks(range(len(groups)))
ax.set_yticklabels([f"{labels[run]}, $N$ = {n:,}".replace(",", " ") for run, n in groups], fontsize=fs - 2)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(250, 14000)
ax.set_xlabel("Collapsed agents (log scale)", fontsize=fs)
ax.tick_params(labelsize=fs - 2)
ax.grid(axis="x", alpha=0.3, which="both")

ymax = len(groups) - 0.5
ax.axvline(args.official, color="crimson", lw=1.5)
box = dict(facecolor="white", edgecolor="none", pad=1.5)
ax.text(args.official / 1.06, -0.8, f"official\n{args.official}", color="crimson", ha="right", va="bottom", fontsize=fs - 3, bbox=box)
ax.axvspan(args.estimates[0], args.estimates[1], color="tab:blue", alpha=0.15, lw=0)
ax.text(args.estimates[0] / 1.04, -0.8, f"Indian estimates\n{args.estimates[0]}–{args.estimates[1]}", color="tab:blue", ha="right", va="bottom", fontsize=fs - 3, bbox=box)
ax.axvline(args.rounds, color="gray", lw=1.5, ls="--")
ax.text(args.rounds * 1.05, -0.8, f"rounds fired\n{args.rounds}", color="gray", ha="left", va="bottom", fontsize=fs - 3, bbox=box)
ax.set_ylim(ymax, -1.6)

handles = [
    Line2D([], [], marker="o", color="black", mfc="black", ls="", label=r"$\alpha = 0.3$ (crowds targeted)"),
    Line2D([], [], marker="o", color="black", mfc="white", ls="", label=r"$\alpha = 0.7$ (crowds protect)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=fs - 3, frameon=False, title=r"two markers per row: $\kappa = 0.9$ above, $\kappa = 0.5$ below", title_fontsize=fs - 4)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
out = Path(args.results_dir) / "summary.pdf"
fig.savefig(out, bbox_inches="tight")
print(out)
