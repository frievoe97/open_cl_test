"""
This module analyzes a CSV file containing execution times and associated image files,
computes image properties (such as dimensions and total pixels), and generates a series
of plots visualizing the execution time performance across different modes and categories.
Plots include boxplots, bar charts, line plots, and heatmaps. All plots are saved in an
output directory with high resolution.

The script performs the following tasks:
- Reads a CSV file containing data about execution times and image file paths.
- Loads each image to compute its dimensions and total number of pixels.
- Augments the CSV data with the computed total pixels.
- Generates multiple plots to compare execution times:
  - Boxplots for overall modes and OpenCL-specific modes.
  - Bar charts for average execution times per mode.
  - Line plots correlating image size (total pixels) with execution time.
  - Grouped bar charts and heatmaps comparing execution times across different image categories.
- Saves all generated plots in the specified output directory.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Path to the CSV file containing execution time and image file data.
csv_path = "../backup/intel/results.csv"

# Directory where plots will be saved.
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)


def save_plot(fig, filename):
    """
    Save a matplotlib figure to the specified output directory with high resolution.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object to save.
        filename (str): The name of the file (including extension) to save the figure as.
    """
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {path}")


def analyze_csv():
    """
    Load the CSV data, compute additional metrics, and generate several plots.

    The function performs the following steps:
      1. Load the CSV file into a pandas DataFrame.
      2. For each unique image file listed in the 'InputFile' column:
         - Load the image using PIL.
         - Calculate its dimensions and total number of pixels.
      3. Map the total pixel count for each image back into the DataFrame.
      4. Generate various plots (boxplots, bar charts, line plots, heatmaps) to 
         visualize execution times across different 'Mode' values and image categories.
      5. Save each generated plot into the output directory.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"The file {csv_path} was not found.")
        return

    print("Data loaded successfully.")

    # Dictionary to store computed image sizes keyed by image file path.
    image_sizes = {}

    # Iterate through each unique image file to compute dimensions and total pixels.
    for input_file in df["InputFile"].unique():
        try:
            with Image.open(input_file) as img:
                width, height = img.size
                total_pixels = width * height
                image_sizes[input_file] = {
                    "dimensions": (width, height),
                    "total_pixels": total_pixels
                }
        except FileNotFoundError:
            print(f"Image file not found: {input_file}")
        except Exception as e:
            print(f"Error loading image {input_file}: {e}")

    # Add a new column "TotalPixels" to the DataFrame based on computed image sizes.
    df["TotalPixels"] = df["InputFile"].map(
        lambda x: image_sizes[x]["total_pixels"] if x in image_sizes else None
    )

    # Configure plot aesthetics for better readability.
    plt.rcParams.update({
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14
    })

    # -------------------------------------------------------------------------
    # Boxplot: Execution Time Distribution by Mode
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    df.boxplot(
        column="ExecutionTimeSeconds",
        by="Mode",
        ax=ax1,
        grid=False,
        patch_artist=True,
        boxprops=dict(facecolor=plt.cm.viridis(0.2))
    )
    ax1.set_xlabel("Mode")
    ax1.set_ylabel("Execution Time (s)")
    ax1.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.suptitle("")
    save_plot(fig1, "boxplot_execution_time_per_mode.png")

    # -------------------------------------------------------------------------
    # Boxplot: Execution Time Distribution for OpenCL Modes
    opencl_modes = df[df["Mode"].str.startswith("opencl")]
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    opencl_modes.boxplot(
        column="ExecutionTimeSeconds",
        by="Mode",
        ax=ax2,
        grid=False,
        patch_artist=True,
        boxprops=dict(facecolor=plt.cm.viridis(0.5))
    )
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("Execution Time (s)")
    ax2.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.suptitle("")
    save_plot(fig2, "boxplot_execution_time_opencl_modes.png")

    # -------------------------------------------------------------------------
    # Bar Chart: Average Execution Time per Mode
    avg_time_per_mode = df.groupby("Mode")["ExecutionTimeSeconds"].mean().sort_values()

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    avg_time_per_mode.plot(
        kind="bar",
        color=[plt.cm.viridis(i / len(avg_time_per_mode)) for i in range(len(avg_time_per_mode))],
        edgecolor="black",
        ax=ax3,
        zorder=3
    )
    ax3.set_xlabel("Mode")
    ax3.set_ylabel("Execution Time (s)")
    ax3.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig3, "barchart_avg_execution_time_per_mode.png")

    # -------------------------------------------------------------------------
    # Bar Chart: Average Execution Time per Mode (Log Scale)
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    avg_time_per_mode.plot(
        kind="bar",
        color=[plt.cm.viridis(i / len(avg_time_per_mode)) for i in range(len(avg_time_per_mode))],
        edgecolor="black",
        ax=ax4,
        zorder=3
    )
    ax4.set_yscale("log")
    ax4.set_xlabel("Mode")
    ax4.set_ylabel("Execution Time (s)")
    ax4.grid(axis='y', which='both', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig4, "barchart_avg_execution_time_per_mode_logscale.png")

    # -------------------------------------------------------------------------
    # Bar Chart: Average Execution Time for OpenCL Modes
    avg_time_opencl = opencl_modes.groupby("Mode")["ExecutionTimeSeconds"].mean().sort_values()
    fig5, ax5 = plt.subplots(figsize=(8, 3))
    avg_time_opencl.plot(
        kind="bar",
        color=[plt.cm.viridis(i / len(avg_time_opencl)) for i in range(len(avg_time_opencl))],
        edgecolor="black",
        ax=ax5,
        zorder=3
    )
    ax5.set_xlabel("Mode")
    ax5.set_ylabel("Execution Time (s)")
    ax5.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig5, "barchart_avg_execution_time_opencl_modes.png")

    # -------------------------------------------------------------------------
    # Line Plot: Execution Time vs. Total Pixels by Mode
    fig6, ax6 = plt.subplots(figsize=(8, 3))
    for i, (mode, group) in enumerate(df.groupby("Mode")):
        group_sorted = group.sort_values("TotalPixels")
        x = group_sorted["TotalPixels"]
        y = group_sorted["ExecutionTimeSeconds"]
        color = plt.cm.viridis(i / len(df["Mode"].unique()))
        ax6.plot(x, y, label=mode, marker="o", color=color)
    ax6.set_xlabel("Total Pixels")
    ax6.set_ylabel("Execution Time (s)")
    ax6.legend(title="Mode", loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=4, frameon=True)
    ax6.grid(color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig6, "lineplot_execution_time_vs_total_pixels.png")

    # -------------------------------------------------------------------------
    # Line Plot: Execution Time vs. Total Pixels by Mode (Log Scale)
    fig7, ax7 = plt.subplots(figsize=(8, 3))
    for i, (mode, group) in enumerate(df.groupby("Mode")):
        group_sorted = group.sort_values("TotalPixels")
        x = group_sorted["TotalPixels"]
        y = group_sorted["ExecutionTimeSeconds"]
        color = plt.cm.viridis(i / len(df["Mode"].unique()))
        ax7.plot(x, y, label=mode, marker="o", color=color)
    ax7.set_yscale("log")
    ax7.set_xlabel("Total Pixels")
    ax7.set_ylabel("Execution Time (s)")
    ax7.legend(title="Mode", loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=4, frameon=True)
    ax7.grid(color='lightgray', linestyle='-', linewidth=0.5, which='both', zorder=-1)
    save_plot(fig7, "lineplot_execution_time_vs_total_pixels_logscale.png")

    # -------------------------------------------------------------------------
    # Line Plot: Execution Time vs. Total Pixels for OpenCL Modes Only
    fig8, ax8 = plt.subplots(figsize=(8, 3))
    for i, (mode, group) in enumerate(opencl_modes.groupby("Mode")):
        group_sorted = group.sort_values("TotalPixels")
        x = group_sorted["TotalPixels"]
        y = group_sorted["ExecutionTimeSeconds"]
        color = plt.cm.viridis(i / len(opencl_modes["Mode"].unique()))
        ax8.plot(x, y, label=mode, marker="o", color=color)
    ax8.set_xlabel("Total Pixels")
    ax8.set_ylabel("Execution Time (s)")
    ax8.legend(title="Mode", loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=4, frameon=True)
    ax8.grid(color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig8, "lineplot_execution_time_vs_total_pixels_opencl_modes.png")

    # -------------------------------------------------------------------------
    # Grouped Bar Chart: Average Execution Time by Image Category and Mode
    df["Category"] = df["InputFile"].str.extract(r'images/([a-zA-Z]+)')[0]
    grouped = df.groupby(["Category", "Mode"])["ExecutionTimeSeconds"].mean().unstack()
    fig9, ax9 = plt.subplots(figsize=(8, 3))
    grouped.plot(
        kind="bar",
        stacked=False,
        ax=ax9,
        colormap="viridis",
        width=0.8,
        zorder=3,
        edgecolor="black"
    )
    ax9.set_xlabel("Category")
    ax9.set_ylabel("Execution Time (s)")
    ax9.legend(title="Mode")
    ax9.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig9, "grouped_barchart_execution_time_by_category.png")

    # -------------------------------------------------------------------------
    # Grouped bar chart: Average execution time by image category for OpenCL modes.
    opencl_modes = df[df["Mode"].str.startswith("opencl")]
    grouped_opencl = opencl_modes.groupby(["Category", "Mode"])["ExecutionTimeSeconds"].mean().unstack()
    fig10, ax10 = plt.subplots(figsize=(8, 3))
    grouped_opencl.plot(
        kind="bar", 
        stacked=False, 
        ax=ax10, 
        colormap="viridis", 
        width=0.8, 
        zorder=3, 
        edgecolor="black"
    )
    ax10.set_xlabel("Category")
    ax10.set_ylabel("Execution Time (s)")
    ax10.legend(
        title="Mode", 
        loc="upper right", 
    )
    ax10.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig10, "grouped_barchart_execution_time_opencl_by_category.png")

    # -------------------------------------------------------------------------
    # Heatmap: Execution Time by Category and Mode
    fig11, ax11 = plt.subplots(figsize=(8, 6))
    heatmap_data = grouped.fillna(0).T

    # Define the desired order of categories.
    desired_order = ["animal", "nature", "dice", "human"]
    heatmap_data = heatmap_data[desired_order]

    cax1 = ax11.matshow(heatmap_data, cmap="viridis", aspect="auto", zorder=-1)
    fig11.colorbar(cax1)
    ax11.set_xticks(range(len(heatmap_data.columns)))
    ax11.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
    ax11.set_yticks(range(len(heatmap_data.index)))
    ax11.set_yticklabels(heatmap_data.index)
    save_plot(fig11, "heatmap_execution_time_by_category_and_mode.png")

    # -------------------------------------------------------------------------
    # Heatmap: Execution Time by Category for OpenCL Modes
    heatmap_opencl_data = grouped_opencl.fillna(0).T
    heatmap_opencl_data = heatmap_opencl_data[desired_order]

    fig12, ax12 = plt.subplots(figsize=(8, 6))
    cax2 = ax12.matshow(heatmap_opencl_data, cmap="viridis", aspect="auto", zorder=-1)
    fig12.colorbar(cax2)
    ax12.set_xticks(range(len(heatmap_opencl_data.columns)))
    ax12.set_xticklabels(heatmap_opencl_data.columns, rotation=45, ha="right")
    ax12.set_yticks(range(len(heatmap_opencl_data.index)))
    ax12.set_yticklabels(heatmap_opencl_data.index)
    save_plot(fig12, "heatmap_execution_time_opencl_by_category_and_mode.png")


if __name__ == "__main__":
    """
    Entry point for the script. Executes the CSV analysis and plotting functions.
    """
    analyze_csv()
