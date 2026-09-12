#!/usr/bin/env bash
# Reproduce every simulation, figure and table of the paper from scratch.
#
#   ./reproduce.sh              full sweeps (many hours; runs use all cores)
#   ./reproduce.sh --quick      same pipeline with tiny crowds and short runs (minutes)
#   RESULTS=/data/amritsar ./reproduce.sh     write to another directory
#   JOBS=8 ./reproduce.sh                      limit parallel workers (default: all cores)
#   SWEEPS="n20000 sigma_20" ./reproduce.sh    run a subset (finished sweeps are always skipped)
#
# All sweeps: main rate_half hits_1p5 open_gates_w3 open_gates_w4 sixth_door kappa_extremes
#             exit_zone_5 exit_zone_15 n20000 sigma_20 sigma_40  (config_<name>.json)
#
# Progress: one line per finished run, "[sweep] 12/60 runs done, 01:23:45 elapsed".
#
# Output layout (default RESULTS=results):
#   results/<sweep>/sweep_simulation_data_<sweep>.pkl   raw results of one sweep
#   results/<sweep>/simulation_summary_<sweep>.json     human-readable summary
#   results/<sweep>/figures/                            time series and fatality maps
#   results/model_figures/                              figures that illustrate the model
#   results/report.md, results/report.csv               tables of all sweeps
#   results/summary.pdf                                 all sweeps against the historical numbers
#   results/environment.txt                             git commit, python, pip freeze
#   results/traj/<sweep>/                               sqlite trajectories (main sweep only)
#   results.zip                                         everything above except traj/ and the logs
set -euo pipefail
cd "$(dirname "$0")"

RESULTS="${RESULTS:-results}"
PYTHON="${PYTHON:-python}"
JOBS="${JOBS:--1}"
START=$(date +%s)
elapsed() { local s=$(( $(date +%s) - START )); printf "%02d:%02d:%02d" $((s/3600)) $((s%3600/60)) $((s%60)); }
runs_in() {  # number of simulations a config defines
  $PYTHON - "$1" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
print(len(c["num_agents_list"]) * len(c["lambda_decay_list"]) * len(c["alpha_list"]) * len(c["kappa_list"]) * c["num_reps"])
EOF
}
QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

ALL_SWEEPS="main rate_half hits_1p5 open_gates_w3 open_gates_w4 sixth_door kappa_extremes exit_zone_5 exit_zone_15 n20000 sigma_20 sigma_40"
SWEEPS="${SWEEPS:-$ALL_SWEEPS}"
config_for() {  # sweep name -> config file (bash 3 compatible, no associative arrays)
  case "$1" in
    main) echo config.json ;;
    *) echo "config_$1.json" ;;
  esac
}

mkdir -p "$RESULTS"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

# --- record the environment
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git status: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') modified files"
  echo "python: $($PYTHON --version 2>&1)"
  echo "quick mode: $QUICK"
  echo
  $PYTHON -m pip freeze
} > "$RESULTS/environment.txt"

# --- quick mode: derive small configs
if [[ $QUICK -eq 1 ]]; then
  mkdir -p "$RESULTS/quick_configs"
  for s in $ALL_SWEEPS; do
    $PYTHON - "$(config_for "$s")" "$RESULTS/quick_configs/$s.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
c.update(num_reps=1, num_agents_list=[300], time_scale=60)
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
  done
fi
config_path() {  # resolves to the quick config in quick mode
  if [[ $QUICK -eq 1 ]]; then echo "$RESULTS/quick_configs/$1.json"; else config_for "$1"; fi
}

# --- sweeps
for s in $SWEEPS; do
  if [[ -f "$RESULTS/$s/sweep_simulation_data_$s.pkl" ]]; then
    echo "== $s: results exist, skipping simulation"
    continue
  fi
  traj="none"
  [[ "$s" == "main" ]] && traj="$RESULTS/traj/$s"
  total=$(runs_in "$(config_path "$s")")
  echo "== $(date '+%F %T') [$s] starting $total runs with config $(config_path "$s") ($(elapsed) elapsed)"
  $PYTHON main.py --config "$(config_path "$s")" --output-dir "$RESULTS" --run-name "$s" \
    --trajectory-dir "$traj" --log-level INFO --jobs "$JOBS" 2>&1 \
    | tee "$RESULTS/$s.log" \
    | { done=0; while IFS= read -r line; do
          case "$line" in
            *"Simulation finished"*) done=$((done + 1)); echo "[$s] $done/$total runs done, $(elapsed) elapsed" ;;
            *Traceback*|*Error*|*"saved to"*) echo "$line" ;;
          esac
        done; } || true
  echo "== $(date '+%F %T') [$s] finished ($(elapsed) elapsed)"
done

# --- figures from the sweeps (every sweep with results, not only those run now)
for s in $ALL_SWEEPS; do
  pkl="$RESULTS/$s/sweep_simulation_data_$s.pkl"
  [[ -f "$pkl" ]] || continue
  echo "== $(date '+%F %T') [$s] figures ($(elapsed) elapsed)"
  $PYTHON plot_fallen_time_series.py "$pkl" --vary alpha
  $PYTHON plot_fallen_time_series.py "$pkl" --vary kappa
  $PYTHON plot_causality_heatmap.py "$pkl"
done

# --- figures that illustrate the model (use the main config)
mkdir -p "$RESULTS/model_figures"
REPO="$PWD"
MAIN_CONFIG="$(cd "$(dirname "$(config_path main)")" && pwd)/$(basename "$(config_path main)")"
( cd "$RESULTS/model_figures" \
  && $PYTHON "$REPO/plot_heatmap_rspace.py" "$MAIN_CONFIG" \
  && $PYTHON "$REPO/plot_shielding_effect.py" "$MAIN_CONFIG" \
  && $PYTHON "$REPO/plot_exit_model.py" "$MAIN_CONFIG" )

# --- report and one-figure summary
$PYTHON make_report.py "$RESULTS"
$PYTHON plot_summary.py "$RESULTS"

# --- archive (pickles, figures, report, environment; trajectories stay on disk)
ZIP="${RESULTS%/}.zip"
$PYTHON - "$RESULTS" "$ZIP" <<'EOF'
import sys, zipfile
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if f.is_file() and rel.parts[0] != "traj" and f.suffix != ".log":
            z.write(f, Path(root.name) / rel)
print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")
EOF
echo "== $(date '+%F %T') done in $(elapsed): $RESULTS/report.md and $ZIP"
