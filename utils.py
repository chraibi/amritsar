"""Utility functions for running main.py."""

import pathlib
from typing import List, Tuple

import jupedsim as jps
import numpy as np
from numpy.random import normal
from shapely import LinearRing, Point, Polygon, intersection

import read_geometry as rr
import time
import json
import sys
import platform
import os
import pickle
import logging


def setup_geometry():
    """Parse geometry file and return walkable_area, exit_areas, spawning_area."""
    wkt = rr.parse_geo_file("./Jaleanwala_Bagh.xml")

    # %%
    # simulation might start with less than that, cause we will filter out some bad positions
    walkable_area0 = wkt[0]
    holes = walkable_area0.interiors[1:]
    holes.append(LinearRing([(84, 90), (84, 87), (90, 87), (90, 90), (84, 90)]))
    holes.append(LinearRing([(170, 80), (171, 80), (171, 81), (170, 81), (170, 80)]))
    holes.append(LinearRing([(100, 40), (101, 40), (101, 41), (100, 41), (100, 40)]))
    walkable_area = Polygon(shell=walkable_area0.exterior, holes=holes)
    exit_areas = [
        Polygon([(216, 124), (217.5, 124), (217.5, 123), (216, 123)]),
        Polygon([(67, 116), (68.5, 116), (68.5, 115), (67, 115)]),
        Polygon([(147, -7), (148.5, -7), (148.5, -6), (147, -6)]),
        Polygon([(92, 0), (93.5, 0), (93.5, 1), (92, 1)]),
        Polygon(
            [
                (213.326, 46.2927),
                (213.21, 49.7972),
                (212.21, 49.7972),
                (212.21, 46.2927),
            ]
        ),
        # Polygon(
        #    [(213.326, 41.2927), (213.21, 39.7972), (212.21, 39.7972), (212.21, 41.2927)]
        # ),
        # Polygon( [(213.326, 46.2927), (213.21, 49.7972), (212.21, 49.7972), (212.21, 46.2927)]),
    ]
    # small
    # spawning_area = Polygon([(60, 99), (172, 99), (172, 11), (60, 11)])
    # big
    spawning_area = Polygon([(40, 115), (202, 115), (202, 5), (40, 5)])
    return (walkable_area, exit_areas, spawning_area)


def setup_simulation(params, rng):
    """Create simulation, init agents with journeys and return simulation."""
    seed = params["seed"]

    num_agents = params["num_agents"]
    trajectory_file = params["trajectory_file"]
    exit_areas = params["exit_areas"]
    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=params["walkable_area"],
        dt=0.01,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=pathlib.Path(trajectory_file)
        ),
    )

    exit_ids = []
    wp_radius = params["wp_radius"]
    for exit_area in params["exit_areas"]:
        wp = exit_area.centroid
        exit_id = simulation.add_waypoint_stage((wp.x, wp.y), wp_radius)
        exit_ids.append(exit_id)

    journey_ids = [
        simulation.add_journey(jps.JourneyDescription([exit_id]))
        for exit_id in exit_ids
    ]
    pos_in_spawning_area = distribute_agents(
        num_agents=num_agents,
        seed=params["seed"],  # TODO seed but lets take same for all
        # spawning_area=params["walkable_area"],
        spawning_area=intersection(params["spawning_area"], params["walkable_area"]),
    )
    v_distribution = normal(params["v0_max"], 0.05, num_agents)
    for pos, v0 in zip(pos_in_spawning_area, v_distribution):
        journey_id, exit_id, _ = get_nearest_exit_id(
            pos,
            exit_areas,
            exit_ids,
            journey_ids,
            rng=rng,
            determinism_strength=params["determinism_strength_exits"],
        )
        simulation.add_agent(
            jps.CollisionFreeSpeedModelAgentParameters(
                journey_id=journey_id,
                stage_id=exit_id,
                position=pos,
                v0=v0,
                radius=0.15,
            )
        )

    return simulation, exit_ids, journey_ids


def convert_seconds_to_hms(seconds):
    """Convert seconds to hours, minutes, and remaining seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    return hours, minutes, remaining_seconds


def distribute_agents(num_agents, seed, spawning_area):
    """Distribute agents in spawning area."""
    pos_in_spawning_area = jps.distributions.distribute_by_number(
        polygon=spawning_area,
        number_of_agents=num_agents,
        distance_to_agents=0.3,
        distance_to_polygon=0.5,
        seed=seed,
    )
    return pos_in_spawning_area


def adjusted_probability(base_prob, shielding, gamma, alpha):
    """Shielding enhances the survival chances.

    alpha = 1.0 → physical shielding (more neighbors = safer)
    alpha = 0.0 → targeted fire (more neighbors = more dangerous)
    """
    crowd_exposure = 1.0 - shielding  # inverse of shielding
    hybrid_factor = alpha * shielding + (1 - alpha) * crowd_exposure

    adjusted_prob = base_prob * (1 + gamma * hybrid_factor)
    return min(adjusted_prob, 1.0)  # clamp to 1.0


def compute_max_risk(xmin, ymin, ymax, sigma, n_shooters):
    y_center = 0.5 * (ymin + ymax)
    shooter_ys = np.linspace(ymin, ymax, n_shooters)
    risk = sum(
        1 / (1 + ((0) ** 2 + (y_center - y) ** 2) / sigma**2) for y in shooter_ys
    )
    return risk


def calculate_probability(
    point,
    time_elapsed,
    lambda_decay,
    time_scale,
    walkable_area,
    shielding,
    gamma,
    alpha,
    rng,
    sigma,
    p_min=0.05,
    p_max=0.95,
    n_shooters=50,
):
    """Calculate the probability of survival for an agent using spatial exposure model."""

    # Spatial bounds
    min_x, min_y, max_x, max_y = walkable_area.bounds

    # Shooter line along x = min_x from min_y to max_y
    shooter_ys = np.linspace(min_y, max_y, n_shooters)

    # Compute exposure risk from all shooter positions
    risk = 0
    # calculate may risk of exposure based on distance to shooters
    for shooter_y in shooter_ys:
        dx = point.x - min_x
        dy = point.y - shooter_y
        risk += 1 / (1 + (dx**2 + dy**2) / sigma**2)

    # Normalize risk by maximum possible value (i.e. at min_x, shooter_y=center)
    max_risk = 29.558  # compute_max_risk(min_x, min_y, max_y, sigma, n_shooters)
    risk_norm = risk / max_risk

    # Convert to survival probability in [p_min, p_max]
    base_survival_prob = p_min + (1 - risk_norm) * (p_max - p_min)

    # Apply small noise
    noise = rng.uniform(0.95, 1.05)
    noisy_survival_prob = np.clip(base_survival_prob * noise, p_min, p_max)

    # Time factor
    normalized_time = time_elapsed / time_scale
    time_factor = np.exp(-lambda_decay * normalized_time)

    # Combine with time
    combined_prob = noisy_survival_prob * time_factor

    # Apply shielding
    probability_final = adjusted_probability(
        combined_prob, shielding, gamma=gamma, alpha=alpha
    )

    # print(
    #     f"{point.x:.2f}",
    #     f"{point.y:.2f}",
    #     f"{risk_norm:.3f}",
    #     f"{noisy_survival_prob:.3f}",
    #     f"{time_factor:.3f}",
    #     f"{probability_final:.3f}",
    # )

    return probability_final


def get_nearest_exit_id(
    position: Point,
    exit_areas: List[Polygon],
    exit_ids: List[int],
    journey_ids: List[int],
    rng,
    determinism_strength: float = 1.0,
) -> Tuple[int, int, float]:
    """
    Return a random exit ID and its distance, with bias toward the nearest exit.

    Args:
        position: The agent's current position.
        exit_areas: List of exit polygons.
        exit_ids: List of exit IDs corresponding to exit_areas.
        determinism_strength: Controls how strongly randomness affects exit selection.
        The higher the determinism factor, the more deterministic the choice becomes
        (favoring the nearest exit)

    Returns:
        Tuple[int, int, float]: Selected journey ID, exit ID and its distance.
    """
    distances = [Point(position).distance(exit_area) for exit_area in exit_areas]
    probabilities = 1 / (np.array(distances) + 1e-6) ** determinism_strength
    probabilities /= probabilities.sum()  # Normalize
    #    selected_exit_id = np.random.choice(exit_ids, p=probabilities)
    selected_exit_id = rng.choice(exit_ids, p=probabilities)
    selected_journey_id = journey_ids[exit_ids.index(selected_exit_id)]
    selected_distance = distances[exit_ids.index(selected_exit_id)]

    return selected_journey_id, selected_exit_id, selected_distance


def maybe_remove_agent(
    simulation, agent, exit_area, exit_probability, exit_radius, rng
):
    """Probabilistically remove agent if they are near an exit centroid."""
    # Set random seed if provided

    distance_to_exit = Point(agent.position).distance(exit_area.centroid)
    if distance_to_exit < exit_radius:
        if rng.random() < exit_probability:
            simulation.mark_agent_for_removal(agent.id)
            return True
    return False


def log_simulation_status(
    elapsed_time, num_fallen, active_agents, total_agents, current_count, fallen_status
):
    """Log the current simulation status."""
    exited = total_agents - current_count
    total_fallen = sum(fallen_status.values())

    print(
        f"[INFO] Time {elapsed_time:.2f}s: "
        f"Num fallen {num_fallen}. Active: {active_agents} "
        f"Exited: {exited}, Fallen total: {total_fallen}. "
        f"Still in simulation: {current_count}. "
        f"Fatality percentage: {total_fallen / total_agents * 100:.2f}%"
    )


def get_trajectory_name(params):
    """Create a descriptive trajectory name from simulation parameters."""
    os.makedirs("traj", exist_ok=True)
    name = (
        f"traj/agents{params['num_agents']}_"
        f"lambda{params['lambda_decay']:.2f}_"
        f"gamma{params['shielding_gamma']:.2f}_"
        f"alpha{params['shielding_alpha']:.2f}_"
        f"tscale{params['time_scale']}_"
        f"detexit{params['determinism_strength_exits']:.1f}_"
        f"probexit{params['exit_probability']:.1f}_"
        f"seed{params['seed']}"
    )
    return name


def save_simulation_results(
    evac_times, dead, fallen_time_series, cl, config, output_dir="fig_results"
):
    """
    Save simulation results along with configuration and metadata.

    Args:
        evac_times: Dictionary of evacuation times
        dead: Dictionary of dead agents
        fallen_time_series: Dictionary of fallen agent time series
        cl: Dictionary of fallen positions
        config: Configuration dictionary used for the simulation
        output_dir: Output directory for results
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_subdir = f"{output_dir}/{timestamp}"
    results_file = f"{output_subdir}/sweep_simulation_data_{timestamp}.pkl"
    os.makedirs(output_subdir, exist_ok=True)

    metadata = {
        "timestamp": timestamp,
        "python_version": sys.version,
        "platform": platform.platform(),
        "total_parameter_combinations": len(evac_times),
        "filename": results_file,
        "simulation_description": "Amritsar Massacre ABM Simulation - Parameter Sweep Results",
    }

    summary_stats = calculate_summary_statistics(evac_times, dead, fallen_time_series)
    data_to_save = {
        # Metadata and configuration
        "metadata": metadata,
        "config": config,
        "summary_statistics": summary_stats,
        # Raw simulation results
        "evac_times": evac_times,
        "dead": dead,
        "fallen_time_series": fallen_time_series,
        "results": cl,
        # Data structure documentation
        "data_structure_info": {
            "evac_times": "Dictionary with keys (num_agents, lambda_decay, alpha) containing lists of evacuation times",
            "dead": "Dictionary with keys (num_agents, lambda_decay, alpha) containing lists of dead agent counts",
            "fallen_time_series": "Dictionary with keys (num_agents, lambda_decay, alpha) containing (time_series, fallen_counts) tuples",
            "fallen_positions": "Dictionary with keys (num_agents, lambda_decay, alpha) containing lists of fallen agent positions",
        },
    }

    with open(results_file, "wb") as f:
        pickle.dump(data_to_save, f)

    summary_file = f"{output_subdir}/simulation_summary_{timestamp}.json"
    save_human_readable_summary(data_to_save, summary_file)

    logging.info(f"Simulation results saved to: {results_file}")
    logging.info(f"Summary saved to: {summary_file}")

    return results_file, summary_file


def calculate_summary_statistics(evac_times, dead, fallen_time_series):
    """Calculate summary statistics for the simulation results."""
    import numpy as np

    summary = {
        "parameter_combinations": {},
        "overall_statistics": {
            "total_simulations_run": 0,
            "avg_evacuation_time": 0,
            "avg_casualties": 0,
            "parameter_ranges": {},
        },
    }

    all_evac_times = []
    all_casualties = []

    for key, evac_list in evac_times.items():
        num_agents, lambda_decay, alpha = key
        dead_list = dead[key]

        # Calculate statistics for this parameter combination
        param_stats = {
            "num_agents": num_agents,
            "lambda_decay": lambda_decay,
            "alpha": alpha,
            "num_repetitions": len(evac_list),
            "evacuation_time": {
                "mean": np.mean(evac_list),
                "std": np.std(evac_list),
                "min": np.min(evac_list),
                "max": np.max(evac_list),
            },
            "casualties": {
                "mean": np.mean(dead_list),
                "std": np.std(dead_list),
                "min": np.min(dead_list),
                "max": np.max(dead_list),
            },
        }

        summary["parameter_combinations"][str(key)] = param_stats
        all_evac_times.extend(evac_list)
        all_casualties.extend(dead_list)
        summary["overall_statistics"]["total_simulations_run"] += len(evac_list)

    # Overall statistics
    if all_evac_times:
        summary["overall_statistics"]["avg_evacuation_time"] = np.mean(all_evac_times)
        summary["overall_statistics"]["avg_casualties"] = np.mean(all_casualties)

    # Parameter ranges
    if evac_times:
        all_keys = list(evac_times.keys())
        num_agents_vals = [k[0] for k in all_keys]
        lambda_vals = [k[1] for k in all_keys]
        alpha_vals = [k[2] for k in all_keys]

        summary["overall_statistics"]["parameter_ranges"] = {
            "num_agents": {"min": min(num_agents_vals), "max": max(num_agents_vals)},
            "lambda_decay": {"min": min(lambda_vals), "max": max(lambda_vals)},
            "alpha": {"min": min(alpha_vals), "max": max(alpha_vals)},
        }

    return summary


def save_human_readable_summary(data, filename):
    """Save a human-readable JSON summary of the simulation."""
    # Create a version that's JSON-serializable
    json_safe_data = {
        "metadata": data["metadata"],
        "config": data["config"],
        "summary_statistics": data["summary_statistics"],
        "data_structure_info": data["data_structure_info"],
    }

    with open(filename, "w") as f:
        json.dump(json_safe_data, f, indent=2, default=str)


def load_simulation_results(filepath):
    """
    Load simulation results from a saved file.

    Args:
        filepath: Path to the saved .pkl file

    Returns:
        Dictionary containing all simulation data
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    logging.info(f"Loaded simulation data from: {filepath}")
    logging.info(f"Simulation timestamp: {data['metadata']['timestamp']}")
    logging.info(
        f"Total parameter combinations: {data['metadata']['total_parameter_combinations']}"
    )
    logging.info(f"Configuration used: {len(data['config'])} parameters")

    return data
