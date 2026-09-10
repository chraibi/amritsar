"""This script runs a simulation of a crowd evacuation scenario with agents that can fall down due to shooting.

The agents' speed is reduced based on the distance to the exit and the time elapsed.
The simulation is run for different values of λ (lambda) which controls the rate of speed reduction.
The simulation is run multiple times to get an average evacuation time and number of fallen agents.

The time series of fallen agents is also plotted.
"""

import argparse
from dataclasses import dataclass
import random
import time
import json
import numpy as np
from joblib import Parallel, delayed
from shapely import Point
import logging

from utils import (
    calculate_probability,
    collapse_hazard,
    collapse_probability,
    configure_logging,
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

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_OUTPUT_DIR = "fig_results"


@dataclass
class SimulationResult:
    """Outcome of a single simulation run."""

    elapsed_time_min: float  # simulated time at the end of the run, in minutes
    agents_remaining: int  # agents still inside (fallen or not exited) at the end
    time_series: list  # update times in seconds
    fallen_per_interval: list  # newly fallen agents at each update time
    fallen_positions: list  # (x, y) of every fallen agent

    @property
    def fallen_total(self):
        return sum(self.fallen_per_interval)


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
    simulation, exit_ids, journey_ids, trajectory_writer = setup_simulation(params, rng)
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
    # Assign individual decay rates to agents
    lambda_range = (lambda_decay - LAMBDA_VARIATION, lambda_decay + LAMBDA_VARIATION)
    agent_lambdas = {
        agent.id: rng.uniform(*lambda_range) for agent in simulation.agents()
    }

    start_time = time.time()
    logger.debug(f"Enter run_evacuation_simulation with seed {seed}")
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
                    sigma=sigma,
                    gamma=gamma,
                    alpha=alpha,
                    radius_around=params["radius_around"],
                    n_max=params["n_max"],
                    model_constants=params["model_constants"],
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

    trajectory_writer.close()  # flush buffered frames and release the sqlite file

    # Log execution time
    execution_time = time.time() - start_time
    hours, minutes, seconds = convert_seconds_to_hms(execution_time)

    logger.info(
        f"Simulation finished: λ={lambda_decay}, Execution time: {hours:2d} h {minutes:2d} min {seconds:.2f} s, fallen: {sum(fallen_over_time)}"
    )

    return SimulationResult(
        elapsed_time_min=simulation.elapsed_time() / 60,
        agents_remaining=simulation.agent_count(),
        time_series=time_series,
        fallen_per_interval=fallen_over_time,
        fallen_positions=overall_fallen_positions,
    )


def update_agent_statuses(
    simulation,
    elapsed_time,
    fallen_status_agents,
    v_distribution,
    agent_lambdas,
    time_scale,
    rng,
    sigma,
    gamma,
    alpha,
    radius_around,
    n_max,
    model_constants,
):
    """Update agent stamina and handle fallen agents."""
    number_fallen_agents = 0
    number_active_agents = 0
    fallen_positions = []
    for agent in simulation.agents():
        agent_id = agent.id
        initial_v0 = v_distribution[agent_id]
        neighbors = list(
            simulation.agents_in_range(pos=agent.position, distance=radius_around)
        )
        shielding = min(1.0, len(neighbors) / n_max)

        if initial_v0 == 0:
            p_collapse = 1.0
        elif model_constants["model"] == "hazard":
            p_collapse = collapse_hazard(
                Point(agent.position),
                elapsed_time,
                shielding,
                lambda_growth=agent_lambdas[agent_id],
                time_scale=time_scale,
                firing_line=model_constants["firing_line"],
                sigma=sigma,
                gamma=gamma,
                alpha=alpha,
                tau_line=model_constants["tau_line"],
                update_time=model_constants["update_time"],
                n_shooters=model_constants["n_shooters"],
            )
        else:  # legacy: exposure survival p(x, t), then the crowding term
            survival_prob = calculate_probability(
                Point(agent.position),
                elapsed_time,
                agent_lambdas[agent_id],
                time_scale,
                model_constants["firing_line"],
                sigma=sigma,
                rng=rng,
                p_min=model_constants["p_min"],
                p_max=model_constants["p_max"],
                n_shooters=model_constants["n_shooters"],
                survival_noise=model_constants["survival_noise"],
            )
            p_collapse = collapse_probability(
                survival_prob,
                shielding,
                gamma=gamma,
                alpha=alpha,
                crowding_model=model_constants["crowding_model"],
            )
        # Check if agent should fall
        rn_number = rng.random()
        if not fallen_status_agents[agent_id] and rn_number < p_collapse:
            number_fallen_agents += 1
            fallen_status_agents[agent_id] = True
            agent.model.v0 = 0
            v_distribution[agent_id] = 0
            fallen_positions.append(tuple(agent.position))
        elif not fallen_status_agents[agent_id]:
            number_active_agents += 1
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
            for exit_area, _exit_id in zip(exit_areas, exit_ids, strict=False):
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
    walkable_area,
    spawning_area,
    exit_areas,
    gamma=0.8,
    seed=None,
    rep_idx=0,
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
    logger.debug(
        f"time_scale: {time_scale}, update_time: {update_time}, seed: {seed}, exit_probability: {exit_probability}, determinism_strength_exits: {determinism_strength_exits}"
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
        "rep_idx": rep_idx,
        "shielding_gamma": gamma,
        "shielding_alpha": alpha,  # 1.0 for physical shielding, 0.0 for targeted fire
        "sigma": sigma,  # for space_factor
        "radius_around": config[
            "radius_around"
        ],  # Radius around agent to consider neighbors
        "n_max": config["n_max"],  # Maximum number of neighbors for full shielding
        "LAMBDA_VARIATION": config["LAMBDA_VARIATION"],  # Variation in lambda values
        # ============================= MODEL CONSTANTS =============
        "dt": config.get("dt", 0.01),  # Simulation time step (s)
        "agent_radius": config.get("agent_radius", 0.15),  # m
        "v0_std": config.get("v0_std", 0.05),  # Std of desired speed distribution (m/s)
        "distance_to_agents": config.get("distance_to_agents", 0.3),  # Initial spacing (m)
        "distance_to_polygon": config.get("distance_to_polygon", 0.5),  # Initial wall distance (m)
        "model_constants": {
            # "hazard": P = h r_space r_time c (default); "legacy": survival form of the submission
            "model": config.get("model", "hazard"),
            "tau_line": config.get("tau_line", 60.0),  # mean time to collapse on the firing line (s)
            "update_time": update_time,
            # legacy only: "risk" symmetric crowding or "survival" form of the submitted paper
            "crowding_model": config.get("crowding_model", "survival"),
            # Firing line endpoints (m); default follows the line drawn on Wagner's map
            "firing_line": tuple(map(tuple, config.get("firing_line", [[12, 11], [38, 90]]))),
            "n_shooters": config.get("n_shooters", 50),  # Shooter positions along the firing line
            "p_min": config.get("p_min", 0.05),  # Survival probability bounds per update
            "p_max": config.get("p_max", 0.95),
            "survival_noise": config.get("survival_noise", 0.05),  # Relative noise on survival probability
        },
    }
    params["trajectory_file"] = get_trajectory_name(params)
    return params


# ============================================================
def load_sweep_config(config_file):
    """Load simulation configuration from a JSON file."""
    with open(config_file) as f:
        return json.load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_FILE, help="Sweep configuration file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (DEBUG prints the per-interval simulation status)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    walkable_area, exit_areas, spawning_area = setup_geometry()

    # ========================= SWEEP PARAMETERS =========================
    # Load sweep parameters from config file
    config = load_sweep_config(args.config)

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
        configure_logging(args.log_level)  # worker processes start unconfigured
        logger.info(
            f"Running rep {rep_idx} (seed {seed_val}): num_agents={num_agents_val}, lambda={lambda_decay_val}, sigma={sigma}, gamma={gamma:.2f}, alpha={alpha_val:.2f}"
        )
        params = init_params(
            num_agents=num_agents_val,
            num_reps=num_reps,
            lambda_decay=lambda_decay_val,
            config=config,
            walkable_area=walkable_area,
            spawning_area=spawning_area,
            exit_areas=exit_areas,
            gamma=gamma,
            sigma=sigma,
            alpha=alpha_val,
            seed=seed_val,
            rep_idx=rep_idx,
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
    for num_agents_val, lambda_decay_val, alpha_val, _rep_idx, result in results:
        key = (num_agents_val, lambda_decay_val, alpha_val)
        if key not in evac_times:
            evac_times[key] = []
            dead[key] = []
            fallen_time_series[key] = ([], [])
            cl[key] = []

        evac_times[key].append(result.elapsed_time_min)
        dead[key].append(result.agents_remaining)
        fallen_time_series[key][0].append(result.time_series)
        fallen_time_series[key][1].append(result.fallen_per_interval)
        cl[key].append(result.fallen_positions)

    results_file, summary_file = save_simulation_results(
        evac_times=evac_times,
        dead=dead,
        fallen_time_series=fallen_time_series,
        cl=cl,
        config=config,
        output_dir=DEFAULT_OUTPUT_DIR,
    )
