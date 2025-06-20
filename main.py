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
import json
import numpy as np
from joblib import Parallel, delayed
from shapely import Point
import logging

from utils import (
    calculate_probability,
    convert_seconds_to_hms,
    get_nearest_exit_id,
    get_trajectory_name,
    log_simulation_status,
    maybe_remove_agent,
    setup_geometry,
    setup_simulation,
    save_simulation_results,
)
import hashlib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_OUTPUT_DIR = "fig_results"


def generate_seeds(base_seed, num_reps):
    """
    Generate a list of reproducible, widely spacerd seeds using a base seed.

    Args:
        base_seed (int): The fixed base seed for reproducibility.
        num_reps (int): Number of repetitions or seeds needed.

    Returns:
        List[int]: A list of unique seeds for each repetition.
    """
    seeds = []
    for i in range(num_reps):
        # Use a hash to ensure well-distributed seed values
        seed_input = f"{base_seed}-{i}"
        hash_digest = hashlib.sha256(seed_input.encode()).hexdigest()
        # Convert hash to int and truncate to stay within RNG limits
        seed = int(hash_digest, 16) % (2**32)  # fits into 32-bit unsigned int
        seeds.append(seed)
    return seeds


def run_evacuation_simulation(params):
    """Run an evacuation simulation with agent stamina decay over time."""
    seed = params["seed"]
    rng = np.random.default_rng(seed)
    # Create simulation
    simulation, exit_ids, journey_ids = setup_simulation(params, rng)
    # Unpack parameters
    update_time = params["update_time"]
    lambda_decay = params["lambda_decay"]
    time_scale = params["time_scale"]
    determinism_strength_exits = params["determinism_strength_exits"]
    exit_probability = params["exit_probability"]
    exit_areas = params["exit_areas"]
    num_agents = params["num_agents"]
    exit_radius = params["wp_radius"]
    gamma = params["shielding_gamma"]
    alpha = params["shielding_alpha"]
    sigma = params["sigma"]
    # Constants
    MAX_SIMULATION_TIME = time_scale
    LAMBDA_VARIATION = params["LAMBDA_VARIATION"]  # variation in lambda values

    # Tracking data structures
    fallen_over_time = []
    time_series = []
    overall_fallen_positions = []
    fallen_status_agents = {agent.id: False for agent in simulation.agents()}
    v_distribution = {agent.id: agent.model.v0 for agent in simulation.agents()}
    last_update_time = -update_time
    wa = params["walkable_area"]
    min_x, min_y, max_x, max_y = wa.bounds
    # Assign individual decay rates to agents
    lambda_range = (lambda_decay - LAMBDA_VARIATION, lambda_decay + LAMBDA_VARIATION)
    agent_lambdas = {
        agent.id: np.random.uniform(*lambda_range) for agent in simulation.agents()
    }

    start_time = time.time()
    # print(f"Enter run_evacuation_simulation with {params['seed']}")
    while (
        simulation.agent_count() > 0
        and simulation.elapsed_time() <= MAX_SIMULATION_TIME
    ):
        simulation.iterate()
        elapsed_time = simulation.elapsed_time()

        # Only update at exact intervals
        if (elapsed_time // update_time) > (last_update_time // update_time):
            last_update_time = elapsed_time
            number_fallen_agents, number_active_agents, fallen_positions = (
                update_agent_statuses(
                    simulation=simulation,
                    fallen_status_agents=fallen_status_agents,
                    v_distribution=v_distribution,
                    agent_lambdas=agent_lambdas,
                    time_scale=time_scale,
                    elapsed_time=elapsed_time,
                    rng=rng,
                    min_x=min_x,
                    min_y=min_y,
                    sigma=sigma,
                    gamma=gamma,
                    alpha=alpha,
                    radius_around=params["radius_around"],
                    n_max=params["n_max"],
                )
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
                rng=rng,
            )

            # Record data
            fallen_over_time.append(number_fallen_agents)
            time_series.append(elapsed_time)
            overall_fallen_positions.extend(fallen_positions)

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

    logging.info(
        f"Simulation finished: λ={lambda_decay}, Execution time: {hours:2d} h {minutes:2d} min {seconds:.2f} s, fallen: {sum(fallen_over_time)}"
    )

    return (
        simulation.elapsed_time() / 60,
        simulation.agent_count(),
        time_series,
        fallen_over_time,
        overall_fallen_positions,
    )


def update_agent_statuses(
    simulation,
    elapsed_time,
    fallen_status_agents,
    v_distribution,
    agent_lambdas,
    time_scale,
    rng,
    min_x,
    min_y,
    sigma,
    gamma,
    alpha,
    radius_around,
    n_max,
):
    """Update agent stamina and handle fallen agents."""
    number_fallen_agents = 0
    number_active_agents = 0
    fallen_positions = []
    num_collapse_attempts = 0
    radius_around = radius_around  # Covers about 7 m²
    n_max = n_max  # Full shielding at ~1.7 persons/m²
    for agent in simulation.agents():
        agent_id = agent.id
        initial_v0 = v_distribution[agent_id]
        neighbors = list(
            simulation.agents_in_range(pos=agent.position, distance=radius_around)
        )
        shielding = min(1.0, len(neighbors) / n_max)
        # print(
        #     f"{simulation.elapsed_time()}: Agent: {agent.id} at {agent.position} has {len(neighbors)} neighbors. Density: {len(neighbors) / np.pi / radius_around**2:.2f}, shielding: {shielding:.2f}"
        # )

        # Calculate agent stamina
        survival_prob = calculate_probability(
            Point(agent.position),
            elapsed_time,
            agent_lambdas[agent_id],
            time_scale,
            walkable_area,
            shielding=shielding,
            gamma=gamma,
            alpha=alpha,
            sigma=sigma,
            rng=rng,
        )

        # small prob -> p_collapse big
        # Higher pcollapse → more likely to collapse
        # Lower pcollapse → less likely to collapse
        if initial_v0 == 0:
            p_collapse = 1.0
        else:
            p_collapse = 1.0 - survival_prob
        # Check if agent should fall
        rn_number = np.random.rand()
        if not fallen_status_agents[agent_id] and rn_number < p_collapse:
            number_fallen_agents += 1
            num_collapse_attempts += 1
            fallen_status_agents[agent_id] = True
            agent.model.v0 = 0
            v_distribution[agent_id] = 0
            fallen_positions.append(tuple(agent.position))

            # Count active agents
            # else:
            #    number_active_agents += 1
        elif not fallen_status_agents[agent_id]:
            number_active_agents += 1
        # print(
        #     f"{agent_id}: "
        #     f"prob = {prob:.2f}, "
        #     f"initial_v0 = {initial_v0:.2f}, "
        #     f"base_speed = {float(base_speed):.2f}, "
        #     f"p_collapse = {float(p_collapse):.2f}, "
        #     f"rn_number = {float(rn_number):.2f}, "
        #     f"fallen_status = {fallen_status_agents[agent_id]}, "
        #     f"position = ({agent.position[0]:.2f}, {agent.position[1]:.2f})"
        # )
    return number_fallen_agents, number_active_agents, fallen_positions


def remove_or_update_journey(
    simulation,
    fallen_status_agents,
    exit_areas,
    exit_ids,
    journey_ids,
    determinism_strength,
    exit_probability,
    exit_radius,
    rng,
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
                    rng=rng,
                )
                if agent_to_be_removed:
                    break

        if not agent_to_be_removed:
            new_journey_id, new_exit_id, *_ = get_nearest_exit_id(
                agent.position,
                exit_areas,
                exit_ids,
                journey_ids,
                rng=rng,
                determinism_strength=determinism_strength,
            )
            simulation.switch_agent_journey(agent.id, new_journey_id, new_exit_id)


def init_params(
    num_agents,
    lambda_decay,
    num_reps,
    alpha,
    sigma,
    config,
    gamma=0.8,
    seed=None,
):
    """Define parameters and return parm object."""
    # ================================= MODEL PARAMETERS =========
    time_scale = config["time_scale"]  # in seconds = 10 min of shooting
    update_time = config["update_time"]  # in seconds
    v0_max = config["v0_max"]  # m/s
    # Add some variability to avoid synchronized agent falls
    determinism_strength_exits = config["determinism_strength_exits"]
    exit_probability = config["exit_probability"]
    wp_radius = config["wp_radius"]  # Radius around exit to consider agent as exiting
    logging.info(
        f"\t\ttime_scale: {time_scale}, update_time: {update_time}, seed: {seed}, exit_probability: {exit_probability}, determinism_strength_exits: {determinism_strength_exits}"
    )
    # =============================================================
    if not seed:
        seed = random.randint(1, 10000)

    params = {
        # ================================= SIMULATION PARAMETERS ========
        "num_agents": num_agents,  # Number of agents in simulation
        "v0_max": v0_max,  # Maximum agent velocity (3 m/s)
        "seed": seed,
        "walkable_area": walkable_area,
        "spawning_area": spawning_area,
        "exit_areas": exit_areas,
        "wp_radius": wp_radius,
        # ============================= AGENT PARAMETERS ============
        "time_scale": time_scale,  # 600 seconds = 10 min of simulation time
        "update_time": update_time,  # How often to update agent status (10 seconds)
        "determinism_strength_exits": determinism_strength_exits,  # Controls randomness in exit selection (0.2)
        "exit_probability": exit_probability,  # Probability of agent exiting when at exit (0.2)
        "lambda_decay": lambda_decay,
        "trajectory_file": "",
        "num_reps": num_reps,
        "shielding_gamma": gamma,
        "shielding_alpha": alpha,  # 1.0 for physical shielding, 0.0 for targeted fire
        "sigma": sigma,  # for space_factor
        "radius_around": config[
            "radius_around"
        ],  # Radius around agent to consider neighbors
        "n_max": config["n_max"],  # Maximum number of neighbors for full shielding
        "LAMBDA_VARIATION": config["LAMBDA_VARIATION"],  # Variation in lambda values
    }
    params["trajectory_file"] = get_trajectory_name(params)
    return params


# ============================================================
def load_sweep_config(config_file):
    """Load simulation configuration from a JSON file."""
    with open(config_file, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    walkable_area, exit_areas, spawning_area = setup_geometry()

    # ========================= SWEEP PARAMETERS =========================
    # Load sweep parameters from config file
    config = load_sweep_config(DEFAULT_CONFIG_FILE)

    num_agents_list = config["num_agents_list"]
    lambda_decay_list = config["lambda_decay_list"]
    alpha_list = config["alpha_list"]
    num_reps = config["num_reps"]
    gamma = config["gamma"]
    sigma = config["sigma"]
    global_seed = config["global_seed"]
    # ================================================================
    # Output storage
    evac_times = {}
    dead = {}
    fallen_time_series = {}
    cl = {}

    all_tasks = []

    # Build all (num_agents, lambda_decay, rep_idx) combinations
    for num_agents_val in num_agents_list:
        rep_seeds = generate_seeds(base_seed=global_seed, num_reps=num_reps)
        for lambda_decay_val in lambda_decay_list:
            for alpha_val in alpha_list:
                for rep_idx in range(num_reps):
                    task = (
                        num_agents_val,
                        lambda_decay_val,
                        alpha_val,
                        sigma,
                        rep_idx,
                        rep_seeds[rep_idx],
                        config,
                    )
                    all_tasks.append(task)

    def run_single_simulation(
        num_agents_val, lambda_decay_val, alpha_val, sigma, rep_idx, seed_val, config
    ):
        """Run a single simulation with given parameters in Parallel."""
        print(
            f">>>> Running simulations for {rep_idx}:{seed_val} num_agents={num_agents_val}, lambda={lambda_decay_val}, sigma = {sigma}, gamma={gamma:.2f}, alpha={alpha_val:.2f}"
        )
        params = init_params(
            num_agents=num_agents_val,
            num_reps=num_reps,
            lambda_decay=lambda_decay_val,
            config=config,
            gamma=gamma,
            sigma=sigma,
            alpha=alpha_val,
            seed=global_seed,  # Important: still base_seed here
        ).copy()
        params["seed"] = seed_val
        base_name = params.get("trajectory_file", "trajectory")
        params["trajectory_file"] = (
            f"{base_name}_agents{num_agents_val}_decay{lambda_decay_val}_rep{rep_idx}.sqlite"
        )
        return (
            num_agents_val,
            lambda_decay_val,
            alpha_val,
            rep_idx,
            run_evacuation_simulation(params=params),
        )

    # Run all tasks fully parallel
    results = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(*task) for task in all_tasks
    )

    # Organize the results
    for num_agents_val, lambda_decay_val, alpha_val, rep_idx, result in results:
        key = (num_agents_val, lambda_decay_val, alpha_val)
        if key not in evac_times:
            evac_times[key] = []
            dead[key] = []
            fallen_time_series[key] = ([], [])
            cl[key] = []

        evac_times[key].append(result[0])
        dead[key].append(result[1])
        fallen_time_series[key][0].append(result[2])
        fallen_time_series[key][1].append(result[3])
        cl[key].append(result[4])

    results_file, summary_file = save_simulation_results(
        evac_times=evac_times,
        dead=dead,
        fallen_time_series=fallen_time_series,
        cl=cl,
        config=config,
        output_dir=DEFAULT_OUTPUT_DIR,
    )
