"""Utility functions for running main.py."""

import functools
import pathlib

import jupedsim as jps
import numpy as np
from shapely import LinearRing, Point, Polygon, intersection

import read_geometry as rr
import time
import json
import sys
import platform
import os
import pickle
import logging

logger = logging.getLogger(__name__)


def configure_logging(level="INFO"):
    """Configure root logging; safe to call again in joblib worker processes."""
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def setup_geometry(extra_exits=()):
    """Parse geometry file and return walkable_area, exit_areas, spawning_area.

    extra_exits: optional list of (x, y) centres of additional openings, each
    modelled as a 1.5 m x 1 m box like the five openings from the map.
    """
    geometry_file = pathlib.Path(__file__).with_name("Jaleanwala_Bagh.xml")
    wkt = rr.parse_geo_file(str(geometry_file))

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
    ]
    for x, y in extra_exits:
        exit_areas.append(
            Polygon([(x - 0.75, y), (x + 0.75, y), (x + 0.75, y - 1), (x - 0.75, y - 1)])
        )
    spawning_area = Polygon([(40, 115), (202, 115), (202, 5), (40, 5)])
    return (walkable_area, exit_areas, spawning_area)


def setup_simulation(params, rng):
    """Create simulation and agents; return simulation, exit ids, journey ids, initial
    target exit per agent, and the trajectory writer.

    The caller must close the trajectory writer at the end of the run.
    """
    num_agents = params["num_agents"]
    trajectory_file = params["trajectory_file"]
    exit_areas = params["exit_areas"]
    trajectory_writer = None
    if trajectory_file:
        trajectory_writer = jps.SqliteTrajectoryWriter(
            output_file=pathlib.Path(trajectory_file),
            every_nth_frame=params["trajectory_every_nth_frame"],
        )
    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=params["walkable_area"],
        dt=params["dt"],
        trajectory_writer=trajectory_writer,
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
        seed=params["seed"],
        spawning_area=intersection(params["spawning_area"], params["walkable_area"]),
        distance_to_agents=params["distance_to_agents"],
        distance_to_polygon=params["distance_to_polygon"],
    )
    v_distribution = rng.normal(params["v0_max"], params["v0_std"], num_agents)
    agent_targets = {}
    for pos, v0 in zip(pos_in_spawning_area, v_distribution, strict=False):
        journey_id, exit_id, _ = get_nearest_exit_id(
            pos,
            exit_areas,
            exit_ids,
            journey_ids,
            rng=rng,
            exit_choice_exponent=params["exit_choice_exponent"],
        )
        agent_id = simulation.add_agent(
            jps.CollisionFreeSpeedModelAgentParameters(
                journey_id=journey_id,
                stage_id=exit_id,
                position=pos,
                v0=v0,
                radius=params["agent_radius"],
            )
        )
        agent_targets[agent_id] = exit_id

    return simulation, exit_ids, journey_ids, agent_targets, trajectory_writer


def convert_seconds_to_hms(seconds):
    """Convert seconds to hours, minutes, and remaining seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    return hours, minutes, remaining_seconds


def distribute_agents(
    num_agents, seed, spawning_area, distance_to_agents=0.3, distance_to_polygon=0.5
):
    """Distribute agents in spawning area."""
    pos_in_spawning_area = jps.distributions.distribute_by_number(
        polygon=spawning_area,
        number_of_agents=num_agents,
        distance_to_agents=distance_to_agents,
        distance_to_polygon=distance_to_polygon,
        seed=seed,
    )
    return pos_in_spawning_area


def exposure_factor(point, firing_line, sigma, n_shooters):
    """Normalised spatial exposure r_space(x) = R(x) / R_max in [0, 1]."""
    shooters = shooter_positions(firing_line, n_shooters)
    risk = exposure_risk(point.x, point.y, shooters, sigma)
    return min(risk / compute_max_risk(firing_line, sigma, n_shooters), 1.0)


def crowding_factor(shielding, gamma, alpha):
    """c(s, alpha) = 1 - gamma (2 alpha - 1)(2 s - 1), in [1 - gamma, 1 + gamma].

    alpha = 1: dense agents are protected, isolated ones exposed;
    alpha = 0: dense clusters are targeted; alpha = 0.5: no effect.
    """
    return 1.0 - gamma * (2.0 * alpha - 1.0) * (2.0 * shielding - 1.0)


def collapse_hazard(
    point,
    time_elapsed,
    shielding,
    lambda_growth,
    time_scale,
    firing_line,
    sigma,
    gamma,
    alpha,
    tau_line,
    update_time,
    n_shooters=50,
):
    """Collapse probability per update (hazard model).

    P = h * r_space(x) * r_time(t) * c(s, alpha), capped at 1, with
    h = update_time / tau_line the per-update collapse probability on the firing
    line (tau_line: mean time to collapse there), r_time = 1 + lambda t / T the
    growth of risk with exposure time, and c the crowding factor.
    """
    h = update_time / tau_line
    r_space = exposure_factor(point, firing_line, sigma, n_shooters)
    r_time = 1.0 + lambda_growth * time_elapsed / time_scale
    hazard = h * r_space * r_time * crowding_factor(shielding, gamma, alpha)
    return float(min(hazard, 1.0))


def sample_hits(weights, expected_hits, rng):
    """Rounds-limited collapse: draw the number of hits this interval and pick who is hit.

    weights: exposure * crowding factor of every active agent (>= 0). The number of
    hits is Poisson with the given mean (rounds per interval times hits per round),
    capped by the number of agents with positive weight; the hit agents are drawn
    without replacement with probability proportional to their weight. Returns the
    indices of the hit agents and the number of hits that found no target.
    """
    weights = np.asarray(weights, dtype=float)
    hits = rng.poisson(expected_hits)
    candidates = np.flatnonzero(weights > 0)
    if hits == 0 or candidates.size == 0:
        return np.empty(0, dtype=int), hits
    n = min(hits, candidates.size)
    p = weights[candidates] / weights[candidates].sum()
    chosen = rng.choice(candidates, size=n, replace=False, p=p)
    return chosen, hits - n


def collapse_probability(survival, shielding, gamma, alpha, crowding_model="risk"):
    """Collapse probability from the exposure survival p(x,t) and the local crowding.

    shielding s in [0, 1] is the local density level; alpha in [0, 1] selects the
    regime: alpha = 1 crowds protect (dense = safer), alpha = 0 crowds are targeted
    (dense = more dangerous), alpha = 0.5 crowding has no effect.

    crowding_model = "risk" (default):
        P = (1 - p) * (1 - gamma * (2 alpha - 1) * (2 s - 1))
        Symmetric: the risk 1 - p is scaled by a factor in [1 - gamma, 1 + gamma],
        so both regimes can raise or lower the risk relative to an agent at s = 0.5.
    crowding_model = "survival" (form used in the submitted manuscript):
        P = 1 - p * (1 + gamma * [alpha s + (1 - alpha)(1 - s)])
        The crowding term only ever raises survival; at alpha = 0 dense agents get
        the plain exposure risk 1 - p and never more.
    """
    if crowding_model == "risk":
        factor = crowding_factor(shielding, gamma, alpha)
        return float(np.clip((1.0 - survival) * factor, 0.0, 1.0))
    if crowding_model == "survival":
        hybrid_factor = alpha * shielding + (1.0 - alpha) * (1.0 - shielding)
        return 1.0 - min(survival * (1.0 + gamma * hybrid_factor), 1.0)
    raise ValueError(f"Unknown crowding_model {crowding_model!r}; use 'risk' or 'survival'")


def shooter_positions(firing_line, n_shooters):
    """Return n_shooters points evenly spaced along the firing line segment.

    firing_line: ((x0, y0), (x1, y1)) endpoints in simulation coordinates.
    """
    (x0, y0), (x1, y1) = firing_line
    t = np.linspace(0.0, 1.0, n_shooters)
    return np.column_stack((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))


def exposure_risk(x, y, shooters, sigma):
    """Sum of Lorentzian kernels from all shooter positions (Eq. rawrisk)."""
    dx = x - shooters[:, 0]
    dy = y - shooters[:, 1]
    return float(np.sum(1.0 / (1.0 + (dx**2 + dy**2) / sigma**2)))


@functools.cache
def compute_max_risk(firing_line, sigma, n_shooters):
    """Largest exposure risk, attained on the firing line near its midpoint.

    Cached: firing_line must be a tuple of two (x, y) tuples.
    """
    shooters = shooter_positions(firing_line, n_shooters)
    return max(exposure_risk(x, y, shooters, sigma) for x, y in shooters)


def calculate_probability(
    point,
    time_elapsed,
    lambda_decay,
    time_scale,
    firing_line,
    rng,
    sigma,
    p_min=0.05,
    p_max=0.95,
    n_shooters=50,
    survival_noise=0.05,
):
    """Legacy exposure survival p(x, t) = r_space(x) * r_time(t) of the submitted manuscript.

    firing_line: ((x0, y0), (x1, y1)) segment along which the shooters stand.
    Combine with collapse_probability() to obtain the collapse probability.
    """
    risk_norm = exposure_factor(point, firing_line, sigma, n_shooters)

    # Convert to survival probability in [p_min, p_max]
    base_survival_prob = p_min + (1 - risk_norm) * (p_max - p_min)

    # Apply small noise
    noise = rng.uniform(1 - survival_noise, 1 + survival_noise)
    noisy_survival_prob = np.clip(base_survival_prob * noise, p_min, p_max)

    # Time factor
    normalized_time = time_elapsed / time_scale
    time_factor = np.exp(-lambda_decay * normalized_time)

    return noisy_survival_prob * time_factor


def get_nearest_exit_id(
    position: Point,
    exit_areas: list[Polygon],
    exit_ids: list[int],
    journey_ids: list[int],
    rng,
    exit_choice_exponent: float = 1.0,
) -> tuple[int, int, float]:
    """
    Return a random exit ID and its distance, with bias toward the nearest exit.

    Args:
        position: The agent's current position.
        exit_areas: List of exit polygons.
        exit_ids: List of exit IDs corresponding to exit_areas.
        exit_choice_exponent: Exponent beta of the inverse-distance weighting.
        The higher the determinism factor, the more deterministic the choice becomes
        (favoring the nearest exit)

    Returns:
        Tuple[int, int, float]: Selected journey ID, exit ID and its distance.
    """
    distances = [Point(position).distance(exit_area) for exit_area in exit_areas]
    probabilities = 1 / (np.array(distances) + 1e-6) ** exit_choice_exponent
    probabilities /= probabilities.sum()  # Normalize
    selected_exit_id = rng.choice(exit_ids, p=probabilities)
    selected_journey_id = journey_ids[exit_ids.index(selected_exit_id)]
    selected_distance = distances[exit_ids.index(selected_exit_id)]

    return selected_journey_id, selected_exit_id, selected_distance


def exit_capacity_per_update(flow_rate, exit_width, update_time):
    """Agents that can pass one opening per update: J * w * dt (persons)."""
    return flow_rate * exit_width * update_time


def select_exiting_agents(candidates, credit, capacity):
    """Pick the agents allowed through an opening in this update.

    candidates: list of (distance to the opening, agent id) for agents inside the
    exit zone. credit: unused capacity carried over from earlier updates. The
    closest agents go first; the carry-over is capped at one update's capacity so
    an empty opening does not bank a burst. Returns (agent ids, new credit).
    """
    credit = min(credit + capacity, 2 * capacity)
    chosen = [agent_id for _, agent_id in sorted(candidates)[: int(credit)]]
    credit -= len(chosen)
    return chosen, min(credit, capacity)


def log_simulation_status(
    elapsed_time, num_fallen, active_agents, total_agents, current_count, fallen_status
):
    """Log the current simulation status."""
    exited = total_agents - current_count
    total_fallen = sum(fallen_status.values())

    logger.debug(
        f"Time {elapsed_time:.2f}s: "
        f"Num fallen {num_fallen}. Active: {active_agents} "
        f"Exited: {exited}, Fallen total: {total_fallen}. "
        f"Still in simulation: {current_count}. "
        f"Fatality percentage: {total_fallen / total_agents * 100:.2f}%"
    )


def get_trajectory_name(params, trajectory_dir="traj"):
    """Create a descriptive trajectory name from simulation parameters."""
    os.makedirs(trajectory_dir, exist_ok=True)
    name = (
        f"{trajectory_dir}/agents{params['num_agents']}_"
        f"lambda{params['lambda_decay']:.2f}_"
        f"gamma{params['shielding_gamma']:.2f}_"
        f"alpha{params['shielding_alpha']:.2f}_"
        f"kappa{params['kappa']:.2f}_"
        f"tscale{params['time_scale']}_"
        f"seed{params['seed']}_"
        f"rep{params['rep_idx']}.sqlite"
    )
    return name


def save_simulation_results(
    evac_times,
    dead,
    fallen_time_series,
    cl,
    config,
    output_dir="fig_results",
    exited_per_exit=None,
    run_name=None,
    hits_without_target=None,
):
    """
    Save simulation results along with configuration and metadata.

    With run_name, files go to <output_dir>/<run_name>/sweep_simulation_data_<run_name>.pkl
    (deterministic paths for the reproduction pipeline); otherwise a timestamp is used.

    Args:
        evac_times: Dictionary of evacuation times
        dead: Dictionary of dead agents
        fallen_time_series: Dictionary of fallen agent time series
        cl: Dictionary of fallen positions
        config: Configuration dictionary used for the simulation
        output_dir: Output directory for results
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = run_name or timestamp
    output_subdir = f"{output_dir}/{tag}"
    results_file = f"{output_subdir}/sweep_simulation_data_{tag}.pkl"
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
        "exited_per_exit": exited_per_exit,
        "hits_without_target": hits_without_target,
        # Data structure documentation
        "data_structure_info": {
            "evac_times": "Dictionary with keys (num_agents, lambda_decay, alpha, kappa) containing lists of evacuation times",
            "dead": "Dictionary with keys (num_agents, lambda_decay, alpha, kappa) containing lists of agents still inside at the end (collapsed or not exited); collapsed counts are the sums of fallen_time_series",
            "fallen_time_series": "Dictionary with keys (num_agents, lambda_decay, alpha, kappa) containing (time_series, fallen_counts) tuples",
            "fallen_positions": "Dictionary with keys (num_agents, lambda_decay, alpha, kappa) containing lists of fallen agent positions",
            "exited_per_exit": "Dictionary with the same keys containing, per run, the number of agents that left through each opening (order of exit_areas)",
        },
    }

    with open(results_file, "wb") as f:
        pickle.dump(data_to_save, f)

    summary_file = f"{output_subdir}/simulation_summary_{tag}.json"
    save_human_readable_summary(data_to_save, summary_file)

    logger.info(f"Simulation results saved to: {results_file}")
    logger.info(f"Summary saved to: {summary_file}")

    return results_file, summary_file


def calculate_summary_statistics(evac_times, dead, fallen_time_series):
    """Calculate summary statistics for the simulation results."""
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
        num_agents, lambda_decay, alpha, kappa = key
        dead_list = [sum(f) for f in fallen_time_series[key][1]]  # collapsed agents per run

        # Calculate statistics for this parameter combination
        param_stats = {
            "num_agents": num_agents,
            "lambda_decay": lambda_decay,
            "alpha": alpha,
            "kappa": kappa,
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
        kappa_vals = [k[3] for k in all_keys]

        summary["overall_statistics"]["parameter_ranges"] = {
            "num_agents": {"min": min(num_agents_vals), "max": max(num_agents_vals)},
            "lambda_decay": {"min": min(lambda_vals), "max": max(lambda_vals)},
            "alpha": {"min": min(alpha_vals), "max": max(alpha_vals)},
            "kappa": {"min": min(kappa_vals), "max": max(kappa_vals)},
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

    logger.info(f"Loaded simulation data from: {filepath}")
    logger.info(f"Simulation timestamp: {data['metadata']['timestamp']}")
    logger.info(
        f"Total parameter combinations: {data['metadata']['total_parameter_combinations']}"
    )
    logger.info(f"Configuration used: {len(data['config'])} parameters")

    return data
