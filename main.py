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
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from joblib import Parallel, delayed
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d

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


# %%
def plot_simulation_configuration(
    walkable_area, spawning_area, starting_positions, exit_areas
):
    axes = pedpy.plot_walkable_area(walkable_area=pedpy.WalkableArea(walkable_area))
    axes.fill(
        *intersection(spawning_area, walkable_area).exterior.xy,
        color="lightgrey",
        alpha=0.2,
    )
    for exit_area in exit_areas:
        axes.fill(*exit_area.exterior.xy, color="indianred")
    axes.scatter(*zip(*starting_positions), s=1, alpha=0.7)
    axes.set_xlabel("x/m")
    axes.set_ylabel("y/m")
    axes.set_aspect("equal")


pos_in_spawning_area = distribute_agents(
    num_agents=100, seed=1, spawning_area=intersection(spawning_area, walkable_area)
)
plot_simulation_configuration(
    walkable_area, spawning_area, pos_in_spawning_area, exit_areas
)


def calculate_probability(point, time_elapsed, lambda_decay, time_scale):
    """Calculate the probability of an agent falling down based on the distance to the exit and the time elapsed."""
    min_x, _, max_x, _ = walkable_area.bounds
    distance_to_left = point.x - min_x
    # todo: add some min distance then people may be initially a bit further from the danger line
    max_distance = max_x - min_x
    distance_factor = distance_to_left / max_distance
    normalized_time = time_elapsed / time_scale
    time_factor = np.exp(-lambda_decay * normalized_time)
    distance_factor = 1 - np.exp(-2 * (distance_to_left / max_distance))
    noise = np.random.uniform(0.95, 1.05)  # ±5% noise
    probability = distance_factor * time_factor * noise
    return probability


def get_nearest_exit_id(
    position: Point,
    exit_areas: List[Polygon],
    exit_ids: List[int],
    journey_ids: List[int],
    randomness_strength: float = 1.0,
) -> Tuple[int, int, float]:
    """
    Return a random exit ID and its distance, with bias toward the nearest exit.

    Args:
        position: The agent's current position.
        exit_areas: List of exit polygons.
        exit_ids: List of exit IDs corresponding to exit_areas.
        randomness_strength: Controls how strongly randomness affects exit selection.
                             Higher values make selection more random.

    Returns:
        Tuple[int, int, float]: Selected journey ID, exit ID and its distance.
    """
    distances = [Point(position).distance(exit_area) for exit_area in exit_areas]
    min_distance = min(distances)
    if randomness_strength < 0.01:
        # Always select the nearest exit
        min_distance = min(distances)
        selected_exit_id = exit_ids[distances.index(min_distance)]
        selected_journey_id = journey_ids[distances.index(min_distance)]
        selected_distance = min_distance
    else:
        # Use distance-based probabilities
        probabilities = 1 / (np.array(distances) + 1e-6) ** randomness_strength
        probabilities /= probabilities.sum()  # Normalize
        selected_exit_id = np.random.choice(exit_ids, p=probabilities)
        selected_journey_id = journey_ids[exit_ids.index(selected_exit_id)]
        selected_distance = distances[exit_ids.index(selected_exit_id)]

    return selected_journey_id, selected_exit_id, selected_distance


def apply_exit_flow_control(
    agent,
    factor,
    exit_areas: List[Polygon],
    exit_ids: List[int],
    journey_ids: List[int],
    damping_radius=5,
    randomness_strength_exits=0,
):
    """Return damped exit v0 and flag if near exit."""
    position = Point(agent.position)
    _, _, nearest_exit_distance = get_nearest_exit_id(
        position, exit_areas, exit_ids, journey_ids=journey_ids
    )
    if nearest_exit_distance < damping_radius:
        damping_factor = max(factor, 1 - nearest_exit_distance / damping_radius)
        return agent.model.v0 * damping_factor, True
    else:
        return agent.model.v0, False


def run_simulation(
    time_scale,
    lambda_decay,
    update_time,
    threshold,
    v0_max,
    recovery_factor,
    damping_factor,
    randomness_strength_exits,
    seed,
    walkable_area,
    spawning_area,
    exit_areas,
    num_agents,
):
    """Run simulation logic."""
    trajectory_file = (
        f"traj/trajectory_Nagents{num_agents}_Seed{seed}_Lambda{lambda_decay}.sqlite"
    )
    pos_in_spawning_area = distribute_agents(
        num_agents=num_agents,
        seed=seed,
        spawning_area=intersection(spawning_area, walkable_area),
    )

    print(
        f"[INFO] Starting simulation with λ={lambda_decay}, {num_agents} agents, seed={seed}"
    )
    start_time = time.time()
    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=walkable_area,
        dt=0.01,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=pathlib.Path(trajectory_file)
        ),
    )

    exit_ids = []
    for exit_area in exit_areas:
        exit_id = simulation.add_exit_stage(exit_area)
        exit_ids.append(exit_id)

    journey_ids = [
        simulation.add_journey(jps.JourneyDescription([exit_id]))
        for exit_id in exit_ids
    ]

    num_agents = len(pos_in_spawning_area)
    v_distribution = normal(v0_max, 0.05, num_agents)
    for pos, v0 in zip(pos_in_spawning_area, v_distribution):
        journey_id, exit_id, _ = get_nearest_exit_id(
            pos,
            exit_areas,
            exit_ids,
            journey_ids,
            randomness_strength=randomness_strength_exits,
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

    # **Tracking fallen agents over time**
    fallen_over_time = []
    time_series = []
    dont_stop = True
    # Create a list to track agents near exits

    fallen_status = {agent.id: False for agent in simulation.agents()}
    near_exit_flag = {agent.id: False for agent in simulation.agents()}
    v_distribution = {agent.id: agent.model.v0 for agent in simulation.agents()}

    last_update_time = -update_time  # Track last update
    causality_locations = defaultdict(int)
    while simulation.agent_count() > 0 and dont_stop:
        simulation.iterate()
        elapsed_time = simulation.elapsed_time()
        # Ensure we only update at exact `update_time` intervals
        if (elapsed_time // update_time) > (last_update_time // update_time):
            last_update_time = elapsed_time  # Update the last processed time
            dont_stop = False
            num_fallen = 0
            active_agents = 0
            for agent in simulation.agents():
                agent_id = agent.id
                initial_v0 = v_distribution[agent_id]
                prob = calculate_probability(
                    Point(agent.position),
                    simulation.elapsed_time(),
                    lambda_decay,
                    time_scale,
                )

                base_speed = initial_v0 * prob
                if base_speed < threshold and not fallen_status[agent.id]:
                    num_fallen += 1
                    fallen_status[agent.id] = True
                    agent.model.v0 = 0
                    v_distribution[agent_id] = 0
                    grid_x, grid_y = (
                        int(agent.position[0] // 5),
                        int(agent.position[1] // 5),
                    )  # 5-unit grid
                    causality_locations[(grid_x, grid_y)] += 1

                # Apply exit damping ONLY to non-fallen agents
                elif not fallen_status[agent.id]:
                    v0_at_exit, is_near_exit = apply_exit_flow_control(
                        agent,
                        factor=damping_factor,
                        exit_areas=exit_areas,
                        exit_ids=exit_ids,
                        journey_ids=journey_ids,
                        damping_radius=20,
                        randomness_strength_exits=0,
                    )
                    if is_near_exit:
                        v_distribution[agent_id] = v0_at_exit
                        agent.model.v0 = max(v0_at_exit, threshold)
                    else:
                        agent.model.v0 = base_speed
                    # Track active agents
                    active_agents += 1

                if agent.model.v0 > threshold or near_exit_flag[agent_id]:
                    dont_stop = True
                # change randomly journey
                new_journey_id, new_exit_id, _ = get_nearest_exit_id(
                    agent.position,
                    exit_areas,
                    exit_ids,
                    journey_ids,
                    randomness_strength=randomness_strength_exits * 2,
                )
                # Change Journeys: Randomly based on distance
                simulation.switch_agent_journey(agent.id, new_journey_id, new_exit_id)

            # Record fallen agent count at this time step
            fallen_over_time.append(num_fallen)
            time_series.append(simulation.elapsed_time())
            exited = num_agents - simulation.agent_count()

            print(
                f"[INFO] Time {simulation.elapsed_time():.2f}s: Num fallen {num_fallen}. Active: {active_agents} Exited: {exited}, Fallen status: {sum(fallen_status.values())}. Still in {simulation.agent_count()}"
            )

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


# ================================= MODEL PARAMETERS =========
num_agents = 500  # 10000, 20000
time_scale = 600  # in seconds = 10 min of shooting
update_time = 10  # in seconds
v0_max = 3  # m/s
# Add some variability to avoid synchronized agent falls
speed_threshold = v0_max * 0.1 + np.random.uniform(-0.1, 0.1)
recovery_factor = 1.0
damping_factor = 0.8
randomness_strength_exits = 1.0
lambda_decays = [0.5]  # , 0.5, 1]
num_reps = 3
# ============================================================
evac_times = {}

dead = {}
fallen_time_series = {}
cl = {}
for lambda_decay in lambda_decays:
    res = Parallel(n_jobs=-1)(
        delayed(run_simulation)(
            time_scale=time_scale,
            lambda_decay=lambda_decay,
            update_time=update_time,
            threshold=speed_threshold,
            v0_max=v0_max,
            randomness_strength_exits=randomness_strength_exits,
            recovery_factor=recovery_factor,
            damping_factor=damping_factor,
            seed=random.randint(1, 10000),
            walkable_area=walkable_area,
            spawning_area=spawning_area,
            exit_areas=exit_areas,
            num_agents=num_agents,
        )
        for _ in range(num_reps)
    )
    res = list(res)
    evac_times[lambda_decay] = [r[0] for r in res]  # Extract evacuation times
    dead[lambda_decay] = [r[1] for r in res]  # Extract number of fallen agents
    fallen_time_series[lambda_decay] = (
        [r[2] for r in res],
        [r[3] for r in res],
    )  # Extract time series
    cl[lambda_decay] = [r[4] for r in res]


# %%
mean_evac_times = {scenario: np.mean(times) for scenario, times in evac_times.items()}
std_dev_evac_times = {scenario: np.std(times) for scenario, times in evac_times.items()}

mean_dead = {scenario: np.mean(dead) for scenario, dead in dead.items()}
std_dead = {scenario: np.std(dead) for scenario, dead in dead.items()}

means = [mean_evac_times[scenario] for scenario in lambda_decays]
std_devs = [std_dev_evac_times[scenario] for scenario in lambda_decays]

means1 = [mean_dead[scenario] for scenario in lambda_decays]
std_devs1 = [std_dead[scenario] for scenario in lambda_decays]

fig1, ax1 = plt.subplots(nrows=1, ncols=1)
fig2, ax2 = plt.subplots(nrows=1, ncols=1)

ax1.errorbar(lambda_decays, means, yerr=std_devs, fmt="o-", ecolor="blue")
ax1.set_xlabel(r"$\lambda$")
ax1.set_ylabel("max. simulation itme [min]")

ax2.errorbar(lambda_decays, means1, yerr=std_devs1, fmt="o-", ecolor="red")
ax2.set_xlabel(r"$\lambda$")
ax2.set_ylabel("Number of agents lying on the ground")

ax2.set_xticks(lambda_decays)
ax1.set_xticks(lambda_decays)
ax2.grid(alpha=0.1)
ax1.grid(alpha=0.1)

# plt.tight_layout()

fig1.savefig(f"{output_dir}/result1_{num_agents}.pdf")
fig2.savefig(f"{output_dir}/result2_{num_agents}.pdf")

fig3, ax3 = plt.subplots(figsize=(8, 5))


def interpolate_series(time_series_list, fallen_series_list, common_time_points):
    """Interpolates all time series to have the same time points."""
    interpolated_values = []
    for times, values in zip(time_series_list, fallen_series_list):
        interp_func = interp1d(
            times, values, kind="linear", bounds_error=False, fill_value=0
        )
        interpolated_values.append(interp_func(common_time_points))
    return np.array(interpolated_values)


fig4, ax4 = plt.subplots(figsize=(10, 6))

# Get a colormap with distinct colors

colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_decays)))
color = "gray"
for i, lambda_decay in enumerate(lambda_decays):
    print(f"Plot with Lambda {lambda_decay}")
    time_series, fallen_series = fallen_time_series[lambda_decay]
    for time_serie, fallen_serie in zip(time_series, fallen_series):
        ax3.plot(time_serie, fallen_serie, color=color, alpha=0.3, linewidth=0.8)
        cumulative_fallen = np.cumsum(fallen_serie)
        ax3.plot(
            time_serie,
            cumulative_fallen,
            color=color,
            alpha=0.3,
            linewidth=0.8,
            linestyle="--",
        )

    representative_time = time_series[0]
    representative_fallen = fallen_series[0]
    representative_cumulative_fallen = np.cumsum(fallen_series[0])
    ax3.plot(
        representative_time,
        representative_fallen,
        label=rf"$\lambda = {lambda_decay}$",
        color=color,
        linewidth=2,
    )

    ax3.plot(
        representative_time,
        representative_cumulative_fallen,
        label=rf"Cumulative $\lambda = {lambda_decay}$",
        color=color,
        linestyle="--",
        linewidth=2,
    )

ax3.set_xlabel("Time [seconds]")
ax3.set_ylabel("New Fallen Agents per Time Step")
ax3.set_title(rf"Fallen Agents $\approx$ {int(np.sum(fallen_series[0]))}")
ax3.grid(alpha=0.3)
ax3.legend()
plt.tight_layout()
fig3.savefig(f"{output_dir}/fallen_agents_time_series_{num_agents}.pdf")

for lambda_decay in lambda_decays:
    min_x, min_y, max_x, max_y = walkable_area.bounds

    grid_size_x = int(max_x // 5)  # Scale down grid for visualization
    grid_size_y = int(max_y // 5)

    # Initialize the casualty grid
    causality_locations = cl[lambda_decay]
    for ic, causality_location in enumerate(causality_locations):
        fig4, ax4 = plt.subplots(figsize=(8, 10))
        causality_grid = np.zeros((grid_size_x, grid_size_y))
        for (x, y), count in causality_location.items():
            grid_x = int(x // 5)  # Scale coordinates for grid
            grid_y = int(y // 5)
            if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
                causality_grid[x, y] += count

        # Plot heatmap
        im = ax4.imshow(
            causality_grid.T, cmap="jet", origin="lower", interpolation="lanczos"
        )
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig4.colorbar(im, cax=cax, label="Number of Fallen Agents")
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x)}")
        )  # Format as int
        vmin, vmax = np.min(causality_grid), np.max(causality_grid)
        cbar.set_ticks(np.linspace(vmin, vmax, num=2))  # Set 5 evenly spaced ticks

        ax4.set_xlabel("X [m]")
        ax4.set_ylabel("Y [m]")
        ax4.set_title(f"Locations of Fallen Agents (λ={lambda_decay})")
        # Save heatmap
        fig4.savefig(
            f"{output_dir}/Casualty_Locations_{num_agents}_lambda_{lambda_decay}_{ic:03d}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig4)
