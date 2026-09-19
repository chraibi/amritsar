"""Collapses against crowd size for the three hits-per-round values.

Usage: python plot_hits_comparison.py <results_dir> [--official 379] [--rounds 1650]
Reads <results_dir>/report.csv (written by make_report.py) and writes
<results_dir>/hits_comparison.pdf and .png. One line per hits-per-round value
(sweeps hits_0p5, main, hits_1p5 at kappa = 0.9), one marker per alpha, error
bars are one standard deviation over repetitions. The label of each line gives the ceiling hits per round x rounds fired that
the toll reaches once the crowd is large enough.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from plot_utils import for_article, save_figure

article = for_article()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results_dir")
parser.add_argument("--official", type=int, default=379)
parser.add_argument("--rounds", type=int, default=1650)
args = parser.parse_args()

# --- Data ---
with open(Path(args.results_dir) / "report.csv") as f:
    rows = [r for r in csv.DictReader(f) if r["run"] in ("hits_0p5", "main", "hits_1p5") and float(r["kappa"]) > 0.7]
hits_values = sorted({float(r["hits_per_round"]) for r in rows})
sizes = sorted({int(r["N"]) for r in rows})

# --- Style Setup ---
sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)
colors = dict(zip(hits_values, [pal[1], pal[3], pal[5]], strict=True))
markers = {0.3: "o", 0.7: "s"}  # alpha -> marker: encoded by shape, colour is the hits value
offset = {0.3: -0.012, 0.7: 0.012}  # relative x shift so the two alphas do not overlap

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
for h in hits_values:
    color = colors[h]
    means = []
    for n in sizes:
        at_n = [r for r in rows if float(r["hits_per_round"]) == h and int(r["N"]) == n]
        means.append(sum(float(r["fallen_mean"]) for r in at_n) / len(at_n))
        for r in at_n:
            alpha = float(r["alpha"])
            ax.errorbar(
                n * (1 + offset[alpha]), float(r["fallen_mean"]), yerr=float(r["fallen_std"]),
                fmt=markers[alpha], ms=7, color=color, mec="white", mew=0.6, ecolor=color, capsize=2, lw=1, zorder=3,
            )
    ax.plot(sizes, means, color=color, lw=1.8, zorder=2)
    ceiling = h * args.rounds
    ax.text(sizes[-1] * 1.08, ceiling, f"{h:g} hit{'s' if h != 1 else ''} per round\nceiling {ceiling:,.0f}".replace(",", " "),
            color=color, ha="left", va="center", fontsize=9)

ax.axhline(args.official, color="#bd0c0c", lw=1.5, zorder=1)
ax.text(sizes[0] * 0.78, args.official * 1.03, f"official toll {args.official}", color="#bd0c0c", ha="left", va="bottom", fontsize=9)

ax.set_xscale("log")
ax.set_xticks(sizes)
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([f"{n:,}".replace(",", " ") for n in sizes])
ax.set_xlim(sizes[0] * 0.75, sizes[-1] * 1.6)
ax.set_ylim(0, max(hits_values) * args.rounds * 1.12)
ax.set_xlabel("Crowd size $N$", fontsize=12, labelpad=8, color="dimgrey")
ax.set_ylabel("Collapsed agents", fontsize=12, labelpad=8, color="dimgrey")
if not article:
    ax.set_title("The toll scales with the hits per round, not with the crowd", fontsize=14, loc="left", pad=7, color="dimgrey")

handles = [
    Line2D([], [], marker=markers[0.3], color="dimgrey", ls="", label=r"$\alpha = 0.3$ (crowds targeted)"),
    Line2D([], [], marker=markers[0.7], color="dimgrey", ls="", label=r"$\alpha = 0.7$ (crowds protect)"),
    Line2D([], [], color="dimgrey", lw=1.8, label="mean over $\\alpha$"),
]
ax.legend(
    handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9.5,
    frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey", labelcolor="dimgrey",
    title=r"$\kappa = 0.9$", title_fontsize=9,
)
ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
ax.grid(False)
ax.grid(axis="y", which="major", alpha=0.7, linewidth=1)
sns.despine(left=True, bottom=True)

# --- Insight annotation ---
if not article:
    fig.text(
        0.98, -0.11,
        f"From $N$ = 10 000 on, every round finds its victims: the toll is fixed by hits per round × {args.rounds} rounds fired",
        ha="right", va="bottom", fontsize=9, color="dimgrey", style="italic",
    )

# --- Save ---
print(save_figure(fig, Path(args.results_dir) / "hits_comparison.pdf"))
