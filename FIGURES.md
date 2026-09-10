# Figures in the article and their sources

Reference dataset: `data_to_publish/20250707_224324/sweep_simulation_data_20250707_224324.pkl`
(sweep with N = 5000, 10000, 15000; lambda = 0.2; alpha = 0.3, 0.7; 5 repetitions;
config in the accompanying `simulation_summary_*.json`).

Scripts that take a pickle are run as `python <script> <pickle>` and write to `fig_results/`.
Scripts without input compute the figure analytically from the model equations.

| Article figure (`figs/`) | Script | Input | Notes |
|---|---|---|---|
| `amritsar_geometry.png` | none in repo | `Jaleanwala_Bagh.xml` | Geometry rendering, produced outside the scripts |
| `massacre_art.jpg` | none | external artwork | |
| `exit_probability.pdf` | `plot_exit_probabilities.py` | none | Exit selection probability vs distance for beta = 0.1, 1.0 |
| `heatmap_lambda_{0.5,1.0,2.0}.pdf` | `plot_heatmap_once.py` | none | Survival probability over distance and time; lambda values set in the script |
| `hazard_field.pdf` (replaces `rspace_at_time_*.pdf`) | `plot_heatmap_rspace.py` | `config.json` | Collapse hazard per update P(x) at c = 1 (hazard model, lambda = 0 so the field is stationary), sweep parameters, firing line as dots, contour at half the on-line hazard. The submitted four-panel figure was made with a separate copy of the legacy survival model (sigma=40, lambda=0.5). |
| `sweep_simulation_data_20250707_224324_causality_lambda_0.2_N_{5000,10000,15000}.pdf` | `plot_causality_heatmap.py` | reference pickle (old 3-key format) | **alpha = 0.7.** The script used to write one file per (N, lambda) and the alpha = 0.7 run overwrote alpha = 0.3; the article figures are pixel-identical to the alpha = 0.7 outputs. The script now expects keys (N, lambda, alpha, kappa) and names files `_alpha_<a>_kappa_<k>`. |
| `sweep_simulation_data_20250707_224324_enhanced_fallen_time_series_N{5000,10000,15000}_alpha.pdf` | `plot_fallen_time_series.py` (replaces `plot_cumulative_fallen_agents_time_{alpha,lambda}.py`) | reference pickle (old 3-key format) | Cumulative fallen agents over time. The new script expects 4-element keys; the submitted figures are reproducible from the scripts in git history at commit 9b1e8c8. |

## Other scripts

| Script | Purpose | Used in article |
|---|---|---|
| `heatmap.py` + `make_heatmap.sh` | Frame sequence and video of the survival heatmap | no |
| `analysis.py` | Aggregate `fallen_agents_stats*.txt` files | no |
| `sqlite_to_jpsvis.py` | Convert a trajectory sqlite file for JPSvis | no |
| `plot_shielding_effect.py` | Crowding factor c(s, alpha) vs local density and alpha, `shielding_effect.pdf` | candidate for the revision |
| `plot_hazard_calibration.py` | Survival of a stationary agent over the event vs distance from the firing line for several tau_line, `hazard_calibration.pdf` | calibration of tau_line; candidate for Methods or supplement |
| `plot_exit_model.py` | Movement rules: map of P(nearest opening) for beta, holding time vs kappa; `exit_choice_map.pdf`, `exit_persistence.pdf` | candidates for Methods |
| `plot_utils.py` | Shared loading of the sweep pickle and the walkable area | helper |

Untracked exploratory scripts (`plot_N_Fallen.py`, `plot_causality.py`, `plot_damping_factor.py`,
`plot_rspace.py`, `plot_survival_alpha.py`, `rspace_heatmap.py`) are not used
for any article figure.
