"""Utility functions for running main.py."""

import os
import pathlib
from typing import List, Tuple

import jupedsim as jps
import numpy as np
from numpy.random import normal
from shapely import LinearRing, Point, Polygon, intersection

import read_geometry as rr


def setup_geometry():
    """Parse geometry file and return walkable_area, exit_areas, spawning_area."""
    wkt = rr.parse_geo_file("./Jaleanwala_Bagh.xml")
    output_dir = "fig_results"
    os.makedirs(output_dir, exist_ok=True)

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


def setup_simulation(params):
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
        seed=seed,
        spawning_area=intersection(params["spawning_area"], params["walkable_area"]),
    )
    v_distribution = normal(params["v0_max"], 0.05, num_agents)
    for pos, v0 in zip(pos_in_spawning_area, v_distribution):
        journey_id, exit_id, _ = get_nearest_exit_id(
            pos,
            exit_areas,
            exit_ids,
            journey_ids,
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


def calculate_probability(
    point, time_elapsed, lambda_decay, time_scale, walkable_area, seed=None
):
    """Calculate the probability of an agent falling down based on the distance to the exit and the time elapsed."""
    min_x, _, max_x, _ = walkable_area.bounds
    distance_to_left = point.x - min_x
    # todo: add some min distance then people may be initially a bit further from the danger line
    # max_distance = max_x - min_x
    # distance_factor = distance_to_left / max_distance
    normalized_time = time_elapsed / time_scale
    time_factor = np.exp(-lambda_decay * normalized_time)
    # distance_factor = 1 - np.exp(-2 * (distance_to_left / max_distance))
    d_crit = 10
    k = 10
    distance_factor = 1 / (1 + np.exp(-(distance_to_left - d_crit) / k))
    if seed:
        np.random.seed(seed)
    noise = np.random.uniform(0.95, 1.05)  # ±5% noise
    probability = distance_factor * time_factor * noise
    return probability


def get_nearest_exit_id(
    position: Point,
    exit_areas: List[Polygon],
    exit_ids: List[int],
    journey_ids: List[int],
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
    selected_exit_id = np.random.choice(exit_ids, p=probabilities)
    selected_journey_id = journey_ids[exit_ids.index(selected_exit_id)]
    selected_distance = distances[exit_ids.index(selected_exit_id)]

    return selected_journey_id, selected_exit_id, selected_distance


def maybe_remove_agent(
    simulation, agent, exit_area, exit_probability, exit_radius, seed=None
):
    """Probabilistically remove agent if they are near an exit centroid."""
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    distance_to_exit = Point(agent.position).distance(exit_area.centroid)
    if distance_to_exit < exit_radius:
        if np.random.rand() < exit_probability:
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
        f"Still in simulation: {current_count}"
    )


def get_trajectory_name(params):
    """Create a descriptive trajectory name from simulation parameters."""
    name = (
        f"traj/agents{params['num_agents']}_"
        f"lambda{params['lambda_decay']:.2f}_"
        f"tscale{params['time_scale']}_"
        f"detexit{params['determinism_strength_exits']:.1f}_"
        f"probexit{params['exit_probability']:.1f}_"
        f"seed{params['seed']}"
    )
    return name
