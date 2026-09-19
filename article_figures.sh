#!/usr/bin/env bash
# Article copies of the figures: the same plots as in results/, drawn with
# --for-article (no title, no insight line; the captions carry these), renamed
# as main.tex expects. See FIGURES.md for the mapping.
#
#   ./article_figures.sh [results_dir] [figs_dir]     defaults: results ../article/Amritsar/figs
set -euo pipefail
cd "$(dirname "$0")"
RESULTS="$(cd "${1:-results}" && pwd)"
FIGS="$(cd "${2:-../article/Amritsar/figs}" && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/main"
ln -s "$RESULTS/main/sweep_simulation_data_main.pkl" "$TMP/main/"
cp "$RESULTS/report.csv" "$TMP/"
PKL="$TMP/main/sweep_simulation_data_main.pkl"

$PYTHON plot_summary.py "$TMP" --for-article >/dev/null
$PYTHON plot_hits_comparison.py "$TMP" --for-article >/dev/null
$PYTHON plot_fallen_time_series.py "$PKL" --vary alpha --for-article >/dev/null
$PYTHON plot_causality_heatmap.py "$PKL" --for-article >/dev/null
( cd "$TMP" \
  && $PYTHON "$OLDPWD/plot_heatmap_rspace.py" "$OLDPWD/config.json" --for-article \
  && $PYTHON "$OLDPWD/plot_shielding_effect.py" "$OLDPWD/config.json" --for-article \
  && $PYTHON "$OLDPWD/plot_exit_model.py" "$OLDPWD/config.json" --for-article ) >/dev/null

cp "$TMP/summary.pdf" "$FIGS/results_summary.pdf"
cp "$TMP/hits_comparison.pdf" "$FIGS/results_hits.pdf"
for f in hazard_field shielding_effect exit_choice_map exit_persistence; do
  cp "$TMP/$f.pdf" "$FIGS/$f.pdf"
done
for n in 5000 10000 15000; do
  cp "$TMP/main/figures/sweep_simulation_data_main_fallen_time_series_N${n}_kappa0.9_vary_alpha.pdf" "$FIGS/results_time_series_N$n.pdf"
done
for n in 5000 15000; do for a in 0.3 0.7; do
  cp "$TMP/main/figures/N_$n/sweep_simulation_data_main_causality_alpha_${a}_kappa_0.9_N_$n.pdf" "$FIGS/results_map_N${n}_alpha$a.pdf"
done; done
echo "article figures written to $FIGS"
