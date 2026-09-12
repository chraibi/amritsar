# Figures in the article and their sources

Reference dataset: `data_to_publish/20250707_224324/sweep_simulation_data_20250707_224324.pkl`
(sweep with N = 5000, 10000, 15000; lambda = 0.2; alpha = 0.3, 0.7; 5 repetitions;
config in the accompanying `simulation_summary_*.json`).

Scripts that take a pickle are run as `python <script> <pickle>` and write to a `figures/` directory next to the pickle. `./reproduce.sh` runs all sweeps, all scripts below and `make_report.py`; see README.
Scripts without input compute the figure analytically from the model equations.

| Article figure (`figs/`) | Script | Input | Notes |
|---|---|---|---|
| `amritsar_geometry.png` | none in repo | `Jaleanwala_Bagh.xml` | Geometry rendering from Wagner's map, produced outside the scripts |
| `massacre_art.jpg` | none | external artwork | |
| `hazard_field.pdf` | `plot_heatmap_rspace.py` | `config.json` | Spatial exposure over the Bagh with the firing line; in the hazard model the collapse probability per interval |
| `shielding_effect.pdf` | `plot_shielding_effect.py` | `config.json` | Crowding factor c(s, alpha) vs local density |
| `exit_choice_map.pdf`, `exit_persistence.pdf` | `plot_exit_model.py` | `config.json` | Probability of heading for the nearest opening; holding time vs kappa |
| `results_summary.pdf` | `plot_summary.py` | `results/report.csv` | Every sweep against the official toll, the Indian estimates and the rounds fired |
| `results_time_series_N*.pdf` | `plot_fallen_time_series.py` | sweep pickle | Cumulative collapses over time, `--vary alpha` or `--vary kappa` |
| `results_map_N*_alpha*.pdf` | `plot_causality_heatmap.py` | sweep pickle | Positions of collapsed agents, mean per 3 m cell, log colour scale |

All of these are produced by `./reproduce.sh` (see README); `results/report.md` and `report.csv` come from `make_report.py`.

## Other scripts

| Script | Purpose | Used in article |
|---|---|---|
| `plot_hazard_calibration.py` | Survival of a stationary agent over the event vs distance, hazard model only | no (documents the hazard model's tau) |
| `sqlite_to_jpsvis.py` | Convert a trajectory sqlite file for JPSvis | no |
| `plot_utils.py` | Shared loading of the sweep pickle and the walkable area | helper |

## Submitted version (SAFETY-D-26-02824, July 2025)

The figures of the submitted manuscript were made with the legacy survival model from the
dataset in `data_to_publish/20250707_224324/` by the scripts as of commit 9b1e8c8
(`plot_cumulative_fallen_agents_time_alpha.py`, `plot_causality_heatmap.py`,
`plot_heatmap_rspace.py`, `plot_heatmap_once.py`, `plot_exit_probabilities.py`). The
current scripts expect the four-element sweep key and cannot read that dataset. The
exploratory scripts and the original notebook were removed from the repository on
2026-09-12 and remain in the git history.
