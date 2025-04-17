"""Converts the SQLite output file from Jupyter-Pedsim to a formatted .txt file for JPSVis."""
# Usage: python sqlite_to_jpsvis.py <sqlite_file>

import sys
import pathlib
import subprocess
from jupedsim.internal.notebook_utils import read_sqlite_file
import numpy as np
import pandas as pd


def compute_speed(traj, df, fps):
    """
    Calculates the speed from the trajectory points with proper padding.
    """
    size = traj.shape[0]
    speed = np.zeros(size)

    if size < df:
        print(
            f"Warning: Trajectory length {size} is shorter than frame difference {df}"
        )
        return np.ones(size)

    # Calculate speeds for the main portion of the trajectory
    delta = traj[df:, :] - traj[:-df, :]
    delta_square = np.square(delta)
    delta_x_square = delta_square[:, 0]
    delta_y_square = delta_square[:, 1]
    s = np.sqrt(delta_x_square + delta_y_square)

    # Place the calculated speeds in the correct position
    speed[df // 2 : -df // 2] = s / df * fps

    # Handle the edges with forward/backward differences
    # Start (use forward difference)
    for i in range(df // 2):
        delta = traj[df : df + 1, :] - traj[i : i + 1, :]
        speed[i] = np.sqrt(np.sum(np.square(delta))) / df * fps

    # End (use backward difference)
    for i in range(size - df // 2, size):
        delta = traj[i : i + 1, :] - traj[size - df - 1 : size - df, :]
        speed[i] = np.sqrt(np.sum(np.square(delta))) / df * fps

    return speed


def export_trajectory_to_txt(
    trajectory_data,
    output_file="output.txt",
    geometry_file="geometry.xml",
    df=10,
    v0=1.2,
    radius=0.18,
):
    """
    Exports trajectory data from a SQLite file to a formatted .txt file, including speed and color.
    """
    df_data = trajectory_data.data
    fps = trajectory_data.frame_rate
    print(f"{fps = }")
    # Extract trajectories for speed calculations
    # trajectories = df_data.groupby("id").apply(
    #     lambda group: group.sort_values(by="frame")[["x", "y"]].values
    # )

    speeds = []
    frame_indices = []

    for traj_id, group in df_data.groupby("id"):
        # Sort by frame within each trajectory
        group = group.sort_values(by="frame")
        traj = group[["x", "y"]].values

        # Calculate speed for this trajectory
        speed = compute_speed(traj, df, fps)

        # Store speed and corresponding frame indices
        speeds.extend(speed)
        frame_indices.extend(group.index)

    speed_series = pd.Series(speeds, index=frame_indices)
    df_data.loc[frame_indices, "speed"] = speed_series
    df_data["angle"] = np.degrees(np.arctan2(df_data["oy"], df_data["ox"]))
    # Calculate color based on speed
    df_data["color"] = (df_data["speed"] / v0 * 255).clip(0, 255).astype(int)

    # Write the formatted data to the output file
    with open(output_file, "w") as f:
        # Write the header
        f.write(f"#framerate: {fps}\n")
        f.write("#unit: m\n")
        f.write(f"#geometry: {geometry_file}\n")
        f.write("#ID\tFR\tX\tY\tZ\tA\tB\tANGLE\tCOLOR\n")

        # Write each row of trajectory data
        for _, row in df_data.iterrows():
            f.write(
                f"{row['id']}\t{row['frame']}\t{row['x']:.4f}\t{row['y']:.4f}\t0\t"
                f"{radius:.4f}\t{radius:.4f}\t{row['angle']:.4f}\t{row['color']}\n"
            )


if len(sys.argv) < 2:
    sys.exit("Usage: python sqlite_to_jpsvis.py <output_path>")


output_path = sys.argv[1]
print(output_path)
trajectory_data, walkable_area = read_sqlite_file(output_path)
output_file = pathlib.Path(output_path).stem + ".txt"
geometry_file = "Jaleanwala_Bagh.xml"
v0_mean = 1.2
export_trajectory_to_txt(
    trajectory_data,
    output_file=output_file,
    geometry_file=geometry_file,
    df=10,
    v0=v0_mean,
)

# polygon_to_xml(walkable_area=walkable_area, output_file=geometry_file)
print(">>> ", output_file)
# print(">>> ", geometry_file)
command = ["/Applications/jpsvis.app/Contents/MacOS/jpsvis", output_file]
result = subprocess.run(command, capture_output=True, text=True)
