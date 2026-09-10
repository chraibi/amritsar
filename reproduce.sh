#!/usr/bin/env bash
# Reproduce every simulation, figure and table of the paper from scratch.
#
#   ./reproduce.sh              full sweeps (many hours; runs use all cores)
#   ./reproduce.sh --quick      same pipeline with tiny crowds and short runs (minutes)
#   RESULTS=/data/amritsar ./reproduce.sh     write to another directory
#
# Output layout (default RESULTS=results):
#   results/<sweep>/sweep_simulation_data_<sweep>.pkl   raw results of one sweep
#   results/<sweep>/simulation_summary_<sweep>.json     human-readable summary
#   results/<sweep>/figures/                            time series and fatality maps
#   results/model_figures/                              figures that illustrate the model
#   results/report.md, results/report.csv               tables of all sweeps
#   results/environment.txt                             git commit, python, pip freeze
#   results/traj/<sweep>/                               sqlite trajectories (main sweep only)
set -euo pipefail
cd "$(dirname "$0")"

RESULTS="${RESULTS:-results}"
PYTHON="${PYTHON:-python}"
QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

SWEEPS="main tau120 open_gates_w3 open_gates_w4 sixth_door"
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
  for s in $SWEEPS; do
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
  echo "== $s: running $(config_path "$s")"
  $PYTHON main.py --config "$(config_path "$s")" --output-dir "$RESULTS" --run-name "$s" \
    --trajectory-dir "$traj" --log-level INFO 2>&1 | tee "$RESULTS/$s.log" | grep -E "Simulation finished|saved to" || true
done

# --- figures from the sweeps
for s in $SWEEPS; do
  pkl="$RESULTS/$s/sweep_simulation_data_$s.pkl"
  echo "== $s: figures"
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
  && $PYTHON "$REPO/plot_hazard_calibration.py" "$MAIN_CONFIG" \
  && $PYTHON "$REPO/plot_shielding_effect.py" "$MAIN_CONFIG" \
  && $PYTHON "$REPO/plot_exit_model.py" "$MAIN_CONFIG" )

# --- report
$PYTHON make_report.py "$RESULTS"
echo "== done: $RESULTS/report.md"
