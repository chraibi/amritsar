import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib.gridspec import GridSpec


# Function to read data from files using pandas
def read_data_files(pattern="*.txt"):
    # Initialize an empty list to store DataFrames
    dfs = []

    # Glob all matching files
    files = glob.glob(pattern)

    for file in files:
        try:
            # Read CSV with no header, assign column names
            df = pd.read_csv(file, header=None, names=["lambda", "fallen", "std", "N"])

            # Add source file information
            df["source_file"] = os.path.basename(file)

            # Calculate additional metrics
            df["fallen_percent"] = (df["fallen"] / df["N"]) * 100
            df["std_percent"] = (df["std"] / df["N"]) * 100

            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    # Combine all DataFrames
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(
            columns=[
                "lambda",
                "fallen",
                "std",
                "N",
                "source_file",
                "fallen_percent",
                "std_percent",
            ]
        )


# Function to create multiple plots
def create_analysis_plots(data):
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})

    # Create a figure with subplots
    # fig = plt.figure(figsize=(18, 12))
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 1, figure=fig)

    # 1. Number of Fallen vs N with error bars
    ax2 = fig.add_subplot(gs[0, 0])
    grouped_n = data.groupby("N").agg({"fallen": "mean", "std": "mean"}).reset_index()
    grouped_n = grouped_n.sort_values("N")
    # ax1.errorbar(
    #     grouped_n["N"],
    #     grouped_n["fallen"],
    #     yerr=grouped_n["std"],
    #     fmt="o-",
    #     capsize=5,
    #     linewidth=2,
    #     markersize=8,
    #     color="blue",
    # )
    # ax1.set_xlabel("N (Population Size)")
    # ax1.set_ylabel("Number of Fallen")
    # ax1.set_title("Number of Fallen vs Population Size")

    # 2. Percentage of Fallen vs N
    # ax2 = fig.add_subplot(gs[0, 1])
    grouped_n_pct = (
        data.groupby("N")
        .agg({"fallen_percent": "mean", "std_percent": "mean"})
        .reset_index()
    )
    grouped_n_pct = grouped_n_pct.sort_values("N")
    ax2.errorbar(
        grouped_n_pct["N"],
        grouped_n_pct["fallen_percent"],
        yerr=grouped_n_pct["std_percent"],
        fmt="o-",
        capsize=5,
        linewidth=2,
        markersize=8,
        color="green",
    )
    ax2.set_xticks([5000, 10000, 15000])

    ax2.set_xlabel("N", fontsize=14)
    ax2.set_ylabel("Percentage Fallen (%)", fontsize=14)
    # ax2.set_title("Percentage of Fallen vs Population Size")

    # 3. Number of Fallen vs Lambda
    # ax3 = fig.add_subplot(gs[0, 2])
    grouped_lambda = (
        data.groupby("lambda").agg({"fallen": "mean", "std": "mean"}).reset_index()
    )
    grouped_lambda = grouped_lambda.sort_values("lambda")
    # ax3.errorbar(
    #     grouped_lambda["lambda"],
    #     grouped_lambda["fallen"],
    #     yerr=grouped_lambda["std"],
    #     fmt="o-",
    #     capsize=5,
    #     linewidth=2,
    #     markersize=8,
    #     color="red",
    # )
    # ax3.set_xlabel("Lambda Value")
    # ax3.set_ylabel("Number of Fallen")
    # ax3.set_title("Number of Fallen vs Lambda")

    # # 4. Heatmap of Lambda vs N (with fallen as values)
    # ax4 = fig.add_subplot(gs[1, 0:2])
    # # Create pivot table for heatmap
    if len(data["lambda"].unique()) > 1 and len(data["N"].unique()) > 1:
        pivot_data = data.pivot_table(
            index="lambda", columns="N", values="fallen", aggfunc="mean"
        )
    #     sns.heatmap(
    #         pivot_data,
    #         annot=True,
    #         cmap="viridis",
    #         ax=ax4,
    #         fmt=".1f",
    #         cbar_kws={"label": "Number of Fallen"},
    #     )
    #     ax4.set_title("Heatmap of Lambda vs N (Number of Fallen)")
    #     ax4.set_xlabel("N (Population Size)")
    #     ax4.set_ylabel("Lambda Value")
    # else:
    #     ax4.text(
    #         0.5,
    #         0.5,
    #         "Insufficient data for heatmap\n(need multiple lambda and N values)",
    #         horizontalalignment="center",
    #         verticalalignment="center",
    #     )

    # # 5. Box plot of fallen by N
    # ax5 = fig.add_subplot(gs[1, 2])
    # if len(data["N"].unique()) > 1:
    #     sns.boxplot(x="N", y="fallen", data=data, ax=ax5, palette="Blues")
    #     ax5.set_title("Distribution of Fallen by N")
    #     ax5.set_xlabel("N (Population Size)")
    #     ax5.set_ylabel("Number of Fallen")
    # else:
    #     ax5.text(
    #         0.5,
    #         0.5,
    #         "Insufficient data for boxplot\n(need multiple values per N)",
    #         horizontalalignment="center",
    #         verticalalignment="center",
    #     )

    plt.tight_layout()
    plt.savefig("comprehensive_analysis.png", dpi=300)

    # Additional individual plots

    # Fallen percentage vs Lambda
    plt.figure(figsize=(10, 6))

    grouped_lambda_pct = (
        data.groupby("lambda")
        .agg({"fallen_percent": "mean", "std_percent": "mean"})
        .reset_index()
    )
    plt.errorbar(
        grouped_lambda_pct["lambda"],
        grouped_lambda_pct["fallen_percent"],
        yerr=grouped_lambda_pct["std_percent"],
        fmt="o-",
        capsize=5,
        linewidth=2,
        markersize=5,
        color="purple",
    )
    plt.xlabel("Lambda Value", fontsize=14)
    plt.ylabel("Percentage Fallen (%)", fontsize=14)
    # plt.title("Percentage of Fallen vs Lambda", fontsize=16)
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(
        "fallen_percent_vs_lambda.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    # Return the grouped data for reference
    return {
        "by_n": grouped_n,
        "by_lambda": grouped_lambda,
        "by_n_percent": grouped_n_pct,
        "by_lambda_percent": grouped_lambda_pct,
    }


# Function to generate statistical summary
def generate_statistical_summary(data):
    # Calculate correlations
    corr = data[["lambda", "N", "fallen", "fallen_percent"]].corr()

    # Group by lambda
    lambda_summary = data.groupby("lambda").agg(
        {"fallen": ["mean", "std", "min", "max", "count"], "N": ["mean"]}
    )

    # Group by N
    n_summary = data.groupby("N").agg(
        {"fallen": ["mean", "std", "min", "max", "count"], "lambda": ["mean"]}
    )

    # Calculate efficiency (fallen per lambda)
    data["efficiency"] = data["fallen"] / data["lambda"]

    return {
        "correlation": corr,
        "lambda_summary": lambda_summary,
        "n_summary": n_summary,
        "efficiency_stats": data["efficiency"].describe(),
    }


# Main execution
if __name__ == "__main__":
    # Change the pattern to match your files
    data = read_data_files("fig_results/fallen_agents_stats*.txt")

    if data.empty:
        print("No data found in the matching files!")
    else:
        print(
            f"Found {len(data)} data points across {data['source_file'].nunique()} files"
        )

        # Display basic statistics
        print("\nBasic statistics:")
        print(data.describe())

        # Generate enhanced plots
        grouped_data = create_analysis_plots(data)

        # Generate statistical summary
        stats = generate_statistical_summary(data)

        # Display correlation matrix
        print("\nCorrelation Matrix:")
        print(stats["correlation"].round(3))

        # Display efficiency statistics
        print("\nEfficiency Statistics (Fallen per Lambda):")
        print(stats["efficiency_stats"])

        print("\nAnalysis complete! All plots have been saved.")
