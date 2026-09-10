"""Summarise all sweep pickles under a results directory into a Markdown report and a CSV.

Usage: python make_report.py <results_dir> [--official 379] [--rounds 1650]

For every <results_dir>/<run>/sweep_simulation_data_<run>.pkl the report lists, per
(N, alpha, kappa): fallen (mean +- std and share of N), exited, still inside at the
end, exits per opening, and the ratio of fallen to rounds fired. It also states the
most conservative case across all sweeps.
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results_dir")
parser.add_argument("--official", type=int, default=379, help="Official death toll")
parser.add_argument("--rounds", type=int, default=1650, help="Rounds fired")
args = parser.parse_args()

results_dir = Path(args.results_dir)
pickles = sorted(results_dir.glob("*/sweep_simulation_data_*.pkl"))
if not pickles:
    raise SystemExit(f"No sweep pickles under {results_dir}")

rows = []
for pkl in pickles:
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    run = pkl.parent.name
    cfg = data["config"]
    per_exit = data.get("exited_per_exit") or {}
    for key in sorted(data["dead"]):
        n, lam, alpha, kappa = key
        fallen = np.array([sum(f) for f in data["fallen_time_series"][key][1]])
        remaining = np.array(data["dead"][key])
        exited = n - remaining
        inside = remaining - fallen
        exits = np.array(per_exit.get(key, [])) if per_exit else np.array([])
        rows.append(
            {
                "run": run,
                "N": n,
                "alpha": alpha,
                "kappa": kappa,
                "tau_line": cfg.get("tau_line"),
                "exit_width": cfg.get("exit_widths") or cfg.get("exit_width"),
                "openings": len(exits[0]) if exits.size else None,
                "reps": len(fallen),
                "fallen_mean": fallen.mean(),
                "fallen_std": fallen.std(),
                "fallen_share": fallen.mean() / n,
                "exited_mean": exited.mean(),
                "inside_mean": inside.mean(),
                "fallen_per_round": fallen.mean() / args.rounds,
                "exits_per_opening": exits.mean(axis=0).round(0).tolist() if exits.size else None,
            }
        )

csv_file = results_dir / "report.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

lines = ["# Simulation results", ""]
lines.append(f"Official death toll for comparison: {args.official}. Rounds fired: {args.rounds}.")
lines.append("")
for run in sorted({r["run"] for r in rows}):
    sub = [r for r in rows if r["run"] == run]
    r0 = sub[0]
    lines.append(f"## {run}")
    lines.append("")
    lines.append(
        f"tau_line = {r0['tau_line']} s, exit width = {r0['exit_width']} m, "
        f"openings = {r0['openings']}, repetitions = {r0['reps']}"
    )
    lines.append("")
    lines.append("| N | alpha | kappa | fallen (mean ± std) | share | exited | inside at end | fallen / round | exits per opening |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sub:
        lines.append(
            f"| {r['N']} | {r['alpha']} | {r['kappa']} | {r['fallen_mean']:.0f} ± {r['fallen_std']:.0f} "
            f"| {100 * r['fallen_share']:.0f}% | {r['exited_mean']:.0f} | {r['inside_mean']:.0f} "
            f"| {r['fallen_per_round']:.2f} | {r['exits_per_opening']} |"
        )
    lines.append("")

best = min(rows, key=lambda r: r["fallen_mean"])
worst = max(rows, key=lambda r: r["fallen_mean"])
lines.append("## Bounds across all sweeps")
lines.append("")
lines.append(
    f"Most conservative case: {best['run']}, N = {best['N']}, alpha = {best['alpha']}, "
    f"kappa = {best['kappa']}: {best['fallen_mean']:.0f} ± {best['fallen_std']:.0f} collapses, "
    f"{best['fallen_mean'] / args.official:.1f} times the official toll, "
    f"{best['fallen_per_round']:.2f} per round fired."
)
lines.append("")
lines.append(
    f"Highest case: {worst['run']}, N = {worst['N']}, alpha = {worst['alpha']}, "
    f"kappa = {worst['kappa']}: {worst['fallen_mean']:.0f} ± {worst['fallen_std']:.0f} collapses."
)
lines.append("")

report_file = results_dir / "report.md"
report_file.write_text("\n".join(lines))
print(f"{report_file}\n{csv_file}")
