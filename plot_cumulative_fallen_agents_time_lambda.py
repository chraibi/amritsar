import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from plot_utils import load_results, walkable_area

loaded_data, stem, output_dir = load_results()

evac_times = loaded_data["evac_times"]
dead = loaded_data["dead"]
fallen_time_series = loaded_data["fallen_time_series"]
print("Simulation data successfully loaded.")

walkable_area = walkable_area()

# Extract scenarios
scenarios = list(fallen_time_series.keys())
print(scenarios)
unique_num_agents = sorted(set(num_agents for num_agents, _, _ in scenarios))
lambda_decays = sorted(set(lambda_decay for _, lambda_decay, _ in scenarios))
alphas = sorted(set(alpha for _, _, alpha in scenarios))
print(f"Unique number of agents: {unique_num_agents}")
print(f"Unique lambda decays: {lambda_decays}")
print(f"Unique alphas: {alphas}")

# --------- Time Series Plot per num_agents with enhanced visualization ----------
max_time = 600  # seconds
for num_agents in unique_num_agents:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pick all lambda values corresponding to this num_agents
    relevant_lambdas = [
        lambda_decay for (n, lambda_decay, _) in scenarios if n == num_agents
    ]
    relevant_alphas = [
        alpha_value for (n, _, alpha_value) in scenarios if n == num_agents
    ]
    relevant_alphas = set(relevant_alphas)
    relevant_lambdas = set(relevant_lambdas)

    fix_alpha = list(relevant_alphas)[1]
    print(f"{num_agents}")
    print(f"{relevant_alphas = }")
    print(f"{relevant_lambdas = }")
    # Dictionary to store processed data for calculating statistics
    processed_data = {}

    # First pass - collect data for each lambda value
    for lambda_decay in relevant_lambdas:
        processed_data[lambda_decay] = []
        time_series_list, fallen_series_list = fallen_time_series[
            (num_agents, lambda_decay, fix_alpha)
        ]

        for time_series, fallen_series in zip(time_series_list, fallen_series_list):
            cumulative_fallen = np.cumsum(fallen_series)

            # Extend all time series to max_time for proper comparison
            if time_series[-1] < max_time:
                extra_times = np.arange(time_series[-1] + 1, max_time + 1)
                extended_time = np.concatenate([time_series, extra_times])
                last_value = cumulative_fallen[-1]
                extended_cumulative = np.concatenate(
                    [cumulative_fallen, np.full(len(extra_times), last_value)]
                )
            else:
                # Truncate to max_time if needed
                # Convert time_series to numpy array if it's not already
                time_series_array = np.array(time_series)
                mask = time_series_array <= max_time
                extended_time = time_series_array[mask]
                extended_cumulative = cumulative_fallen[mask]

                # Make sure we have a value exactly at max_time
                if extended_time[-1] < max_time:
                    extended_time = np.append(extended_time, max_time)
                    extended_cumulative = np.append(
                        extended_cumulative, extended_cumulative[-1]
                    )

            # Resample to common time points for statistics
            common_times = np.linspace(0, max_time, 1000)
            resampled_values = np.interp(
                common_times, extended_time, extended_cumulative
            )
            processed_data[lambda_decay].append(resampled_values)

    # Second pass - plot individual runs and statistics
    for lambda_decay in relevant_lambdas:
        # Choose color based on lambda value
        if lambda_decay == 0.2:
            color = "teal"  # Teal color for lambda=0.2
        elif lambda_decay == 0.3:
            color = "gold"  # Gold color for lambda=0.5
        else:
            color = plt.cm.viridis(lambda_decay / max(lambda_decays))

        # Get all resampled time series for this lambda
        all_series = np.array(processed_data[lambda_decay])

        # Calculate mean and std at each time point
        mean_values = np.mean(all_series, axis=0)
        std_values = np.std(all_series, axis=0)

        # Calculate final statistics for legend
        final_mean = mean_values[-1]
        final_std = std_values[-1]

        # Plot individual simulation runs with transparency
        time_series_list, fallen_series_list = fallen_time_series[
            (num_agents, lambda_decay, fix_alpha)
        ]
        for i, (time_series, fallen_series) in enumerate(
            zip(time_series_list, fallen_series_list)
        ):
            cumulative_fallen = np.cumsum(fallen_series)

            if time_series[-1] < max_time:
                extra_times = np.arange(time_series[-1] + 1, max_time + 1)
                extended_time = np.concatenate([time_series, extra_times])
                last_value = cumulative_fallen[-1]
                extended_cumulative = np.concatenate(
                    [cumulative_fallen, np.full(len(extra_times), last_value)]
                )
            else:
                # Convert time_series to numpy array if it's not already
                time_series_array = np.array(time_series)
                mask = time_series_array <= max_time
                extended_time = time_series_array[mask]
                extended_cumulative = cumulative_fallen[mask]

            ax.plot(
                extended_time,
                extended_cumulative,
                color=color,
                alpha=0.3,  # More transparency for individual runs
                linewidth=0.8,
                linestyle="-",
            )

        # Common times for smooth mean and std lines
        common_times = np.linspace(0, max_time, 1000)

        # Plot mean with bold line
        ax.plot(
            common_times,
            mean_values,
            color=color,
            linewidth=3,  # Bold line for mean
            linestyle="-",
            label=rf"$\alpha = {fix_alpha:.1f},\ \lambda = {lambda_decay:.1f}$  (mean: {final_mean:.0f} $\pm$ {final_std:.0f})",
        )

        # Add shaded area for standard deviation
        ax.fill_between(
            common_times,
            mean_values - std_values,
            mean_values + std_values,
            color=color,
            alpha=0.2,
        )

    # Enhance the plot appearance
    ax.set_xlabel("Time [s]", fontsize=18)
    ax.set_ylabel("Cumulative Fallen Agents", fontsize=18)
    #    ax.set_title(f"Fallen Agents Over Time (N = {num_agents})", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="major", labelsize=16)  # Larger x-axis numbers
    ax.tick_params(axis="y", which="major", labelsize=16)  # Standard y-axis numbers

    # Improve legend
    legend = ax.legend(fontsize=14, loc="upper left", framealpha=0.9)
    legend.get_frame().set_linewidth(1)

    # Save the figure
    output_file = (
        Path(output_dir) / f"{stem}_enhanced_fallen_time_series_N{num_agents}.pdf"
    )
    fig.savefig(output_file, dpi=300, bbox_inches="tight")

    # Also save as PNG for easier viewing
    png_output = (
        Path(output_dir) / f"{stem}_enhanced_fallen_time_series_N{num_agents}.png"
    )
    fig.savefig(png_output, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f">> Saved: {output_file} and {png_output}")

print("Enhanced visualization complete.")
