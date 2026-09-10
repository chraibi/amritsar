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
| `rspace_at_time_{0,200,400,600}.pdf` | `plot_heatmap_rspace.py` | `config.json` | Survival field p(x,t) without noise or shielding, computed with `utils.calculate_probability` and the sweep parameters; firing line drawn as white dots. The submitted figures were made with a separate copy of the model (sigma=40, lambda=0.5). |
| `sweep_simulation_data_20250707_224324_causality_lambda_0.2_N_{5000,10000,15000}.pdf` | `plot_causality_heatmap.py` | reference pickle | **alpha = 0.7.** The script used to write one file per (N, lambda) and the alpha = 0.7 run overwrote alpha = 0.3. Output names now include `_alpha_<value>`; the article figures are pixel-identical to the `alpha_0.7` outputs. |
| `sweep_simulation_data_20250707_224324_enhanced_fallen_time_series_N{5000,10000,15000}_alpha.pdf` | `plot_cumulative_fallen_agents_time_alpha.py` | reference pickle | Cumulative fallen agents over time, both alpha values, lambda fixed |

## Other scripts

| Script | Purpose | Used in article |
|---|---|---|
| `plot_cumulative_fallen_agents_time_lambda.py` | Same as the alpha variant but varying lambda at fixed alpha | no |
| `heatmap.py` + `make_heatmap.sh` | Frame sequence and video of the survival heatmap | no |
| `analysis.py` | Aggregate `fallen_agents_stats*.txt` files | no |
| `sqlite_to_jpsvis.py` | Convert a trajectory sqlite file for JPSvis | no |
| `plot_utils.py` | Shared loading of the sweep pickle and the walkable area | helper |

Untracked exploratory scripts (`plot_N_Fallen.py`, `plot_causality.py`, `plot_damping_factor.py`,
`plot_rspace.py`, `plot_shielding.py`, `plot_survival_alpha.py`, `rspace_heatmap.py`) are not used
for any article figure.
