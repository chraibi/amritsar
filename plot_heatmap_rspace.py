from shapely import wkt
from shapely.geometry import Polygon, LinearRing, Point
import numpy as np

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# Plot heatmap with detailed walkable area
import matplotlib.pyplot as plt


seed = 10
rng = np.random.default_rng(seed)


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
    p_min=0.05,
    p_max=0.95,
    sigma=40.0,
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

    return combined_prob


# Parse the walkable area from the provided WKT polygon string
polygon_wkt = (
    "POLYGON ((146.87 -7.66, 146.87 0, 76.87 0.0000000000000086, 75.37 0.0000000000000088, 0 0, "
    "31.8898 95.1903, 58.7163 96.6689, 62.3453 116.662, 67.3453 116.662, 68.8453 116.662, "
    "100.575 116.662, 148.323 133.381, 152.963 133.381, 152.963 125.331, 217.053 125.331, "
    "218.553 125.331, 213.326 41.2927, 213.21 39.7972, 211.829 19.845, 196.802 21.9229, "
    "190.58 -7.66, 148.37 -7.66, 146.87 -7.66), "
    "(149.842 33.3846, 149.842 29.3085, 155.859 29.3085, 155.959 33.3395, 149.842 33.3846), "
    "(93.1082 106.314, 93.1082 103.984, 97.5472 103.984, 97.5472 106.203, 93.1082 106.314), "
    "(99.1009 101.653, 99.1009 98.657, 105.204 98.657, 105.315 101.653, 99.1009 101.653), "
    "(107.646 96.2155, 107.646 93.1082, 113.861 93.2192, 113.861 96.3265, 107.646 96.2155), "
    "(156.248 41.1484, 156.248 38.4311, 160.906 38.4311, 160.857 41.2481, 156.248 41.1484), "
    "(162.819 41.2704, 162.93 38.4013, 167.178 38.4565, 167.068 41.2704, 162.819 41.2704), "
    "(168.613 39.8358, 168.668 37.0771, 173.358 37.0771, 173.247 39.891, 168.613 39.8358), "
    "(173.137 34.815, 173.137 31.9459, 178.71 31.9459, 178.599 34.8701, 173.137 34.815), "
    "(84 90, 84 87, 90 87, 90 90, 84 90), "
    "(170 80, 171 80, 171 81, 170 81, 170 80), "
    "(100 40, 101 40, 101 41, 100 41, 100 40))"
)
walkable_area = wkt.loads(polygon_wkt)
min_x, min_y, max_x, max_y = walkable_area.bounds

nx = 500
ny = 500
# Recreate grid
x = np.linspace(min_x, max_x, nx)
y = np.linspace(min_y, max_y, ny)
X, Y = np.meshgrid(x, y)

for t in [0, 200, 400, 600]:
    Z = np.zeros_like(X)
    lambda_decay = 0.5
    for i in range(ny):
        for j in range(nx):
            pt = Point(X[i, j], Y[i, j])
            if walkable_area.contains(pt):
                Z[i, j] = calculate_probability(
                    point=pt,
                    time_elapsed=t,
                    lambda_decay=lambda_decay,
                    time_scale=600,
                    walkable_area=walkable_area,
                    shielding=1,
                    gamma=0.8,
                    alpha=1,
                    rng=rng,
                )
            else:
                Z[i, j] = np.nan  # outside walkable area
    fig, ax = plt.subplots(figsize=(10, 10), dpi=600)

    color_map = "viridis"
    color_map = "inferno"
    # color_map = "magma"
    # color_map = "plasma"
    # color_map = "cividis"
    im = ax.imshow(
        Z,
        origin="lower",
        extent=(min_x, max_x, min_y, max_y),
        cmap=color_map,
        vmin=0.05,
        vmax=0.95,
    )
    fs = 20
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=fs)
    cbar.set_label("Survival Probability", fontsize=fs)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x)}")
    )  # Format as int
    cbar.set_ticks(np.linspace(0.05, 0.95, num=2))  # Set 5 evenly spaced ticks
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    ax.set_title(f"time = {t} s", fontsize=fs)

    ax.set_xlabel("X [m]", fontsize=fs)
    ax.set_ylabel("Y [m]", fontsize=fs)

    ax.set_xticklabels(ax.get_xticks(), fontsize=fs)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fs)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    # Plot outline of walkable area and holes
    x_outer, y_outer = walkable_area.exterior.xy
    ax.plot(x_outer, y_outer, color="black", linewidth=1)
    for interior in walkable_area.interiors:
        x_hole, y_hole = interior.xy
        ax.plot(x_hole, y_hole, color="black", linewidth=1)

    fig.tight_layout()
    fig.savefig(f"rspace_at_time_{t}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"rspace_at_time_{t}.pdf")
