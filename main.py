"""This script runs a simulation of a crowd evacuation scenario with agents that can fall down due to shooting.

The agents' speed is reduced based on the distance to the exit and the time elapsed.
The simulation is run for different values of λ (lambda) which controls the rate of speed reduction.
The simulation is run multiple times to get an average evacuation time and number of fallen agents.

The time series of fallen agents is also plotted.
"""

import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from shapely import Point

from utils import (
    calculate_probability,
    convert_seconds_to_hms,
    get_nearest_exit_id,
    get_trajectory_name,
    log_simulation_status,
    maybe_remove_agent,
    setup_geometry,
    setup_simulation,
)


def run_evacuation_simulation(params):
    """Run an evacuation simulation with agent stamina decay over time."""
    np.random.seed(params["seed"])
    # Create simulation
    simulation, exit_ids, journey_ids = setup_simulation(params)
    # Unpack parameters
    update_time = params["update_time"]
    lambda_decay = params["lambda_decay"]
    time_scale = params["time_scale"]
    determinism_strength_exits = params["determinism_strength_exits"]
    exit_probability = params["exit_probability"]
    exit_areas = params["exit_areas"]
    num_agents = params["num_agents"]
    exit_radius = params["wp_radius"]
    # Constants
    MAX_SIMULATION_TIME = time_scale
    GRID_SIZE = 5  # units for causality tracking
    LAMBDA_VARIATION = 0.1  # variation in lambda values

    # Tracking data structures
    fallen_over_time = []
    time_series = []
    fallen_status_agents = {agent.id: False for agent in simulation.agents()}
    v_distribution = {agent.id: agent.model.v0 for agent in simulation.agents()}
    last_update_time = -update_time
    causality_locations = defaultdict(int)

    # Assign individual decay rates to agents
    lambda_range = (lambda_decay - LAMBDA_VARIATION, lambda_decay + LAMBDA_VARIATION)
    agent_lambdas = {
        agent.id: np.random.uniform(*lambda_range) for agent in simulation.agents()
    }

    start_time = time.time()

    while (
        simulation.agent_count() > 0
        and simulation.elapsed_time() <= MAX_SIMULATION_TIME
    ):
        simulation.iterate()
        elapsed_time = simulation.elapsed_time()

        # Only update at exact intervals
        if (elapsed_time // update_time) > (last_update_time // update_time):
            last_update_time = elapsed_time

            number_fallen_agents, number_active_agents = update_agent_statuses(
                simulation=simulation,
                fallen_status_agents=fallen_status_agents,
                v_distribution=v_distribution,
                agent_lambdas=agent_lambdas,
                causality_locations=causality_locations,
                grid_size=GRID_SIZE,
                time_scale=time_scale,
                elapsed_time=elapsed_time,
                seed=params["seed"],
            )

            remove_or_update_journey(
                simulation,
                fallen_status_agents,
                exit_areas,
                exit_ids,
                journey_ids,
                determinism_strength_exits,
                exit_probability,
                exit_radius,
            )

            # Record data
            fallen_over_time.append(number_fallen_agents)
            time_series.append(elapsed_time)

            # Log status
            log_simulation_status(
                elapsed_time,
                number_fallen_agents,
                number_active_agents,
                num_agents,
                simulation.agent_count(),
                fallen_status_agents,
            )

            if number_active_agents == 0:
                break

    # Log execution time
    execution_time = time.time() - start_time
    hours, minutes, seconds = convert_seconds_to_hms(execution_time)
    print(
        f"[INFO] Simulation finished: λ={lambda_decay}, Execution time: {hours} h {minutes} min {seconds:.1f} s"
    )

    return (
        simulation.elapsed_time() / 60,
        simulation.agent_count(),
        time_series,
        fallen_over_time,
        causality_locations,
    )


def update_agent_statuses(
    simulation,
    elapsed_time,
    fallen_status_agents,
    v_distribution,
    agent_lambdas,
    time_scale,
    causality_locations,
    grid_size,
    seed,
):
    """Update agent stamina and handle fallen agents."""
    number_fallen_agents = 0
    number_active_agents = 0

    for agent in simulation.agents():
        agent_id = agent.id
        initial_v0 = v_distribution[agent_id]

        # Calculate agent stamina
        prob = calculate_probability(
            Point(agent.position),
            elapsed_time,
            agent_lambdas[agent_id],
            time_scale,
            walkable_area,
            seed=seed,
        )

        base_speed = initial_v0 * prob
        p_collapse = 1.0 if initial_v0 == 0 else 1.0 - (base_speed / initial_v0)

        # Check if agent should fall
        if not fallen_status_agents[agent_id] and np.random.rand() < p_collapse:
            number_fallen_agents += 1
            fallen_status_agents[agent_id] = True
            agent.model.v0 = 0
            v_distribution[agent_id] = 0

            # Record casualty location
            grid_x, grid_y = (
                int(agent.position[0] // grid_size),
                int(agent.position[1] // grid_size),
            )
            causality_locations[(grid_x, grid_y)] += 1

        # Count active agents
        elif not fallen_status_agents[agent_id]:
            number_active_agents += 1

    return number_fallen_agents, number_active_agents


def remove_or_update_journey(
    simulation,
    fallen_status_agents,
    exit_areas,
    exit_ids,
    journey_ids,
    determinism_strength,
    exit_probability,
    exit_radius,
):
    """Check if agent has to be removed otherwise update journey."""
    for agent in simulation.agents():
        agent_to_be_removed = False  # assume agent is not exiting the simulation yet.

        # Only process movement for active agents
        if not fallen_status_agents[agent.id]:
            # Try to remove agent if near exit
            for exit_area, exit_id in zip(exit_areas, exit_ids):
                agent_to_be_removed = maybe_remove_agent(
                    simulation,
                    agent,
                    exit_area,
                    exit_probability=exit_probability,
                    exit_radius=exit_radius,
                )
                if agent_to_be_removed:
                    break

        if not agent_to_be_removed:
            new_journey_id, new_exit_id, *_ = get_nearest_exit_id(
                agent.position,
                exit_areas,
                exit_ids,
                journey_ids,
                determinism_strength=determinism_strength,
            )
            simulation.switch_agent_journey(agent.id, new_journey_id, new_exit_id)


def init_params(seed=None):
    """Define parameters and return parm object."""
    # ================================= MODEL PARAMETERS =========
    num_agents = 10  # 10000, 20000
    time_scale = 600  # in seconds = 10 min of shooting
    update_time = 10  # in seconds
    v0_max = 3  # m/s
    # Add some variability to avoid synchronized agent falls
    determinism_strength_exits = 0.2
    exit_probability = 0.2
    lambda_decay = 0.5  # [0.1, 0.4, 0.5]  # , 0.5, 1]
    num_reps = 1

    if not seed:
        seed = random.randint(1, 10000)

    params = {
        "num_agents": num_agents,  # Number of agents in simulation
        "v0_max": v0_max,  # Maximum agent velocity (3 m/s)
        "seed": seed,
        "walkable_area": walkable_area,
        "spawning_area": spawning_area,
        "exit_areas": exit_areas,
        "wp_radius": 10,
        # ====
        "time_scale": time_scale,  # 600 seconds = 10 min of simulation time
        "update_time": update_time,  # How often to update agent status (10 seconds)
        "determinism_strength_exits": determinism_strength_exits,  # Controls randomness in exit selection (0.2)
        "exit_probability": exit_probability,  # Probability of agent exiting when at exit (0.2)
        "lambda_decay": lambda_decay,
        "trajectory_file": "",
        "num_reps": num_reps,
    }
    params["trajectory_file"] = get_trajectory_name(params)
    return params


# ============================================================
if __name__ == "__main__":
    walkable_area, exit_areas, spawning_area = setup_geometry()
    # set seet to a constant value for reproducibility. Otherwise it will be random
    params = init_params(seed=None)
    num_reps = params["num_reps"]
    lambda_decay = params["lambda_decay"]
    num_agents = params["num_agents"]
    evac_times = {}
    dead = {}
    fallen_time_series = {}
    cl = {}

    def run_with_unique_filename(rep_idx, base_params):
        """Create a copy of params to avoid modifying the original."""
        local_params = base_params.copy()
        base_name = base_params.get("trajectory_file", "trajectory")
        local_params["trajectory_file"] = f"{base_name}_rep{rep_idx}.sqlite"
        return run_evacuation_simulation(params=local_params)

    res = Parallel(n_jobs=-1, verbose=1)(
        delayed(run_with_unique_filename)(rep_indx, base_params=params)
        for rep_indx in range(num_reps)
    )
    res = list(res)
    evac_times[lambda_decay] = [r[0] for r in res]  # Extract evacuation times
    dead[lambda_decay] = [r[1] for r in res]  # Extract number of fallen agents
    fallen_time_series[lambda_decay] = (
        [r[2] for r in res],
        [r[3] for r in res],
    )  # Extract time series
    cl[lambda_decay] = [r[4] for r in res]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "fig_results"
    os.makedirs(output_dir, exist_ok=True)

    save_path = f"{output_dir}/simulation_data_{num_agents}_{timestamp}.pkl"
    data_to_save = {
        "evac_times": evac_times,
        "dead": dead,
        "fallen_time_series": fallen_time_series,
        "cl": cl,
    }

    with open(save_path, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Simulation results saved to {save_path}")
    print(f"Trajectory file {params['trajectory_file']}")
