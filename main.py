"""This script runs a simulation of a crowd evacuation scenario with agents that can fall down due to shooting.

The agents' speed is reduced based on the distance to the exit and the time elapsed.
The simulation is run for different values of λ (lambda) which controls the rate of speed reduction.
The simulation is run multiple times to get an average evacuation time and number of fallen agents.

The time series of fallen agents is also plotted.
"""

import jupedsim as jps
import pedpy
import read_geometry as rr
from shapely import Polygon, Point, LinearRing, intersection
import pathlib
from numpy.random import normal
import random
import numpy as np
from typing import List, Tuple
import os
from collections import defaultdict
from joblib import Parallel, delayed
import time
import pickle

# %%
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
        [(213.326, 46.2927), (213.21, 49.7972), (212.21, 49.7972), (212.21, 46.2927)]
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


def calculate_probability(point, time_elapsed, lambda_decay, time_scale, seed=None):
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


def setup_simulation(params):
    """Create simulation, init agents with journeys and return simulation."""
    seed = params["seed"]
    num_agents = params["num_agents"]
    trajectory_file = params["trajectory_file"]

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
        f"agents{params['num_agents']}_"
        f"lambda{params['lambda_decay']:.2f}_"
        f"tscale{params['time_scale']}_"
        f"detexit{params['determinism_strength_exits']:.1f}_"
        f"probexit{params['exit_probability']:.1f}_"
        f"seed{params['seed']}"
        ".sqlite"
    )
    return name


def init_params(seed=None):
    "Define parameters and return parm object."
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
    params = init_params(seed=111)
    num_reps = params["num_reps"]
    lambda_decay = params["lambda_decay"]
    num_agents = params["num_agents"]
    evac_times = {}
    dead = {}
    fallen_time_series = {}
    cl = {}
    res = Parallel(n_jobs=-1)(
        delayed(run_evacuation_simulation)(params=params) for _ in range(num_reps)
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
