# Rethinking the Amritsar Massacre through Agent-Based Modeling and Social Psychology

This repository contains an agent-based modeling (ABM) simulation that models crowd evacuation dynamics during crisis scenarios, specifically inspired by the Amritsar Massacre. 
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
| `determinism_strength_exits` | float | Exit selection randomness (0-1) | 0.2 |
| `exit_probability` | float | Probability of exiting when at exit | 0.2 |
| `wp_radius` | float | Exit detection radius (meters) | 1.0 |
| **Model Parameters** | | | |
| `lambda_decay_list` | list | Stamina decay rates to test | [0.1, 0.5, 1.0] |
| `alpha_list` | list | Shielding effectiveness values | [0.0, 0.5, 1.0] |
| `gamma` | float | Shielding decay parameter | 0.8 |
| `sigma` | float | Space factor parameter in meters | 20 |


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

- **λ (lambda_decay)**: Controls the rate of agent stamina decay over time. Higher values mean faster deterioration.
- **α (alpha)**: Shielding effectiveness parameter. 1.0 = full physical shielding, 0.0 = targeted effects.
- **γ (gamma)**: Decay rate for shielding effectiveness.
- **σ (sigma)**: Spatial factor affecting survival probability.

## Analysis and Visualization

The repository includes several plotting scripts for analyzing simulation results:

### Plot Scripts

| Script | Description | Fixed Parameters |
|--------|-------------|------------------|
| `plot_cumulative_fallen_agents_time_lambda.py` | Cumulative fallen agents over time for different λ values | α = fixed |
| `plot_cumulative_fallen_agents_time_alpha.py` | Cumulative fallen agents over time for different α values | λ = fixed |
| `heatmap.py` | Multiple PNG survival heatmaps | Various parameters |
| `plot_heatmap_once.py` | Single PDF survival heatmap | λ = fixed |
| `plot_heatmap_rspace.py` | Heatmaps for spatial analysis at 4 time points | - |

### Running Analysis

```bash
# Generate time series plots
python plot_cumulative_fallen_agents_time_lambda.py
python plot_cumulative_fallen_agents_time_alpha.py

# Generate heatmaps
python heatmap.py
python plot_heatmap_once.py
python plot_heatmap_rspace.py
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



## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -am 'Add new analysis feature'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

## Citation

If you use this simulation in your research, please cite:

```
TBD
```

## License

MIT License.

## Contact

https://www.chraibi.de/
