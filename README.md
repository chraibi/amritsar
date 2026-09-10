# Rethinking the Amritsar Massacre through Agent-Based Modeling and Social Psychology

This repository contains an agent-based modeling (ABM) simulation that models crowd evacuation dynamics during crisis scenarios, specifically inspired by the [Amritsar massacre](https://en.wikipedia.org/wiki/Jallianwala_Bagh_massacre). 
The simulation is built using [JuPedSim](jupedsim.org), a software for simulating pedestrian dynamics, and incorporates social psychology principles to understand how crowd behavior, and spatial constraints affect evacuation outcomes.


## Overview

The simulation models agents (pedestrians) attempting to evacuate from a confined space under crisis conditions using the [JuPedSim](jupedsim.org) pedestrian dynamics software. 

Key features include:

- **Stamina decay over time**: Agents' movement speed decreases based on exposure time and distance to exits
- **Social shielding effects**: Crowd density provides protective effects against targeting
- **Targeting behavior**: Depending on the parameters shielding effect can be turns into targeting behavior.
- **Stochastic agent collapse**: Probabilistic model for agents falling due to various factors
- **Multiple exit strategies**: Agents dynamically choose exits based on distance and crowding
- **Parallel simulation runs**: Support for parameter sweeps with multiple repetitions

## Installation

### Prerequisites

- Python 3.8+
- Required packages (see requirements below)

### Environment Setup

1. **Clone the repository:**
```bash
git clone https://github.com/chraibi/amritsar.git
cd amritsar
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Configuration

The simulation is controlled through a `config.json` file.

### Configuration Parameters

| Parameter | Type | Description | Default/Example |
|-----------|------|-------------|-----------------|
| **Simulation Settings** | | | |
| `time_scale` | int | Total simulation time in seconds | 600 |
| `update_time` | int | Status update interval in seconds | 10 |
| `num_reps` | int | Number of repetitions per parameter set | 10 |
| `global_seed` | int | Base seed for reproducibility | 42 |
| **Agent Parameters** | | | |
| `num_agents_list` | list | List of agent counts to test | [100, 200, 500] |
| `v0_max` | float | Maximum agent velocity (m/s) | 3.0 |
| `exit_choice_exponent` | float | β in the exit choice P_i ∝ d_i^-β | 1.0 |
| `kappa_list` | list | Persistence: probability per update of keeping the target opening | [0.5, 0.9] |
| `exit_flow_rate` | float | J, persons per metre per second an opening passes | 1.3 |
| `exit_width` | float | w, width of an opening (m); capacity per update = J·w·update_time | 1.5 |
| `exit_widths` | list | Optional per-opening widths (m), overrides `exit_width`; one entry per opening | |
| `extra_exits` | list | Optional additional openings as [x, y] centres on the wall (1.5 m x 1 m boxes) | [] |
| `wp_radius` | float | Radius of the exit zone around an opening (m) | 10 |
| **Model Parameters** | | | |
| `lambda_decay_list` | list | Stamina decay rates to test | [0.1, 0.5, 1.0] |
| `alpha_list` | list | Shielding effectiveness values | [0.0, 0.5, 1.0] |
| `gamma` | float | Shielding decay parameter | 0.8 |
| `sigma` | float | Space factor parameter in meters | 20 |
| **Model Constants** | | | |
| `dt` | float | Simulation time step (s) | 0.01 |
| `trajectory_every_nth_frame` | int | Frames between trajectory writes (100 = 1 frame/s) | 100 |
| `agent_radius` | float | Agent body radius (m) | 0.15 |
| `v0_std` | float | Std of the desired-speed distribution (m/s) | 0.05 |
| `distance_to_agents` | float | Minimum initial spacing between agents (m) | 0.3 |
| `distance_to_polygon` | float | Minimum initial distance to walls (m) | 0.5 |
| `model` | str | `hazard`: P = h·r_space·r_time·c per update (default); `legacy`: survival form of the submitted paper | hazard |
| `tau_line` | float | Mean time to collapse of an agent on the firing line (s); h = update_time / tau_line | 60 |
| `crowding_model` | str | legacy model only: `survival` (submitted form) or `risk` | survival |
| `firing_line` | list | Endpoints [[x0, y0], [x1, y1]] of the shooters' line (m), from Wagner's map | [[12, 11], [38, 90]] |
| `n_shooters` | int | Shooter positions evenly spaced along the firing line | 50 |
| `p_min`, `p_max` | float | Bounds of the per-update survival probability | 0.05, 0.95 |
| `survival_noise` | float | Relative noise applied to the survival probability | 0.05 |


## Reproducing the results of the paper

Everything in the article (simulations, figures, tables) is produced by one command
from a clean checkout:

```bash
pip install -r requirements.txt
./reproduce.sh            # all sweeps, figures and the report; many hours on a multi-core machine
./reproduce.sh --quick    # same pipeline with tiny crowds, a few minutes, to check the setup
```

`JOBS=8 ./reproduce.sh` limits the parallel workers; the script prints one progress line per finished run.

`reproduce.sh` runs the five sweeps defined by `config.json` (main results),
`config_tau120.json` (lower-bound sensitivity), `config_open_gates_w3.json`,
`config_open_gates_w4.json` (wider openings) and `config_sixth_door.json` (the closed
door on the north wall open), then the plot scripts and `make_report.py`. Output goes
to `results/` (override with `RESULTS=/path ./reproduce.sh`):

```
results/<sweep>/sweep_simulation_data_<sweep>.pkl   raw results
results/<sweep>/figures/                            time series and fatality maps
results/model_figures/                              figures illustrating the model
results/report.md, results/report.csv               tables for all sweeps
results/environment.txt                             git commit, Python version, pip freeze
results/traj/main/                                  sqlite trajectories of the main sweep
results.zip                                         all of the above except the trajectories
```

Runs are seeded (`global_seed` in the config) and reproducible for a fixed jupedsim
version, which is pinned in `requirements.txt`. The results reported in the article
were produced with the tagged release (see Citation) by exactly this command;
`environment.txt` in the archived results records the commit and package versions.

## Development

```bash
pip install pytest ruff
ruff check .
PYTHONPATH=. pytest
```

## Usage

### Running Simulations

1. **Single simulation run:**
```bash
python main.py
```

2. **The simulation will:**
   - Load parameters from `config.json`
   - Run all parameter combinations in parallel
   - Save results to `fig_results/sweep_simulation_data_TIMESTAMP.pkl`

### Key Parameters Explained

- **λ (lambda_decay)**: Growth of the collapse hazard with exposure time, r_time = 1 + λ t/T (hazard model); decay rate of the survival probability in the legacy model.
- **α (alpha)**: Shielding effectiveness parameter. 1.0 = full physical shielding, 0.0 = targeted effects.
- **γ (gamma)**: Decay rate for shielding effectiveness.
- **σ (sigma)**: Spatial factor affecting survival probability.

## Analysis and Visualization

The repository includes several plotting scripts for analyzing simulation results:

### Plot Scripts

See [FIGURES.md](FIGURES.md) for the mapping between article figures, scripts and input data.
Scripts that read a sweep pickle share their loading code in `plot_utils.py`:

```bash
python plot_fallen_time_series.py fig_results/<run>/sweep_simulation_data_<run>.pkl [--vary alpha|kappa]
python plot_causality_heatmap.py fig_results/<run>/sweep_simulation_data_<run>.pkl
python plot_heatmap_once.py
python plot_heatmap_rspace.py
python plot_exit_probabilities.py
```

## Output Structure

```
project/
├── fig_results/
│   ├── sweep_simulation_data_TIMESTAMP.pkl  # Raw simulation data
│   ├── heatmaps/                           # Generated heatmap images
│   └── plots/                             # Time series plots
├── trajectories/                          # Individual simulation trajectories
└── config.json                           # Configuration file
```

## Key Features

### Technical Features

- **Parallel Processing**: Utilizes joblib for efficient parameter sweeps
- **Reproducible Results**: Deterministic seeding for consistent outcomes
- **Scalable Architecture**: Handles large numbers of agents and parameter combinations
- **Comprehensive Logging**: Detailed simulation progress and results tracking




## Citation

If you use this simulation in your research, please cite:

```
TBD
```

## License

MIT License.

## Contact

https://www.chraibi.de/
