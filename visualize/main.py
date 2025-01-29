"""
This module analyzes a CSV file containing execution times and associated image files,
computes image properties (such as dimensions and total pixels), and generates a series
of plots visualizing the execution time performance across different modes and categories.
Plots include boxplots, bar charts, line plots, and heatmaps. All plots are saved in an
output directory with high resolution.

The script performs the following tasks:
- Reads in a CSV file containing data about execution times and image file paths.
- Loads each image to compute its dimensions and total number of pixels.
- Augments the CSV data with the computed total pixels.
- Generates multiple plots to compare execution times:
  - Boxplots for overall modes and OpenCL-specific modes.
  - Bar charts for average execution times per mode.
  - Line plots correlating image size (total pixels) with execution time.
  - Grouped bar charts and heatmaps comparing execution times across different image categories.
- Saves all generated plots in the specified output directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Path to the CSV file containing execution time and image file data.
# csv_path = "../output/results.csv"
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
    df["TotalPixels"] = df["InputFile"].map(lambda x: image_sizes[x]["total_pixels"] if x in image_sizes else None)

    # -------------------------------------------------------------------------
    # Create a boxplot showing the distribution of execution times by mode.
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    df.boxplot(
        column="ExecutionTimeSeconds", 
        by="Mode", 
        ax=ax1, 
        grid=False, 
        patch_artist=True, 
        boxprops=dict(facecolor=plt.cm.viridis(0.2))
    )
    ax1.set_title("Execution Time Distribution by Mode")
    ax1.set_xlabel("Mode")
    ax1.set_ylabel("Execution Time (s)")
    ax1.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.suptitle("")
    save_plot(fig1, "boxplot_execution_time_per_mode.png")

    # -------------------------------------------------------------------------
    # Create a boxplot for execution times for modes starting with "opencl".
    opencl_modes = df[df["Mode"].str.startswith("opencl")]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    opencl_modes.boxplot(
        column="ExecutionTimeSeconds", 
        by="Mode", 
        ax=ax2, 
        grid=False, 
        patch_artist=True, 
        boxprops=dict(facecolor=plt.cm.viridis(0.5))
    )
    ax2.set_title("Execution Time Distribution for OpenCL Modes")
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("Execution Time (s)")
    ax2.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.suptitle("")
    save_plot(fig2, "boxplot_execution_time_opencl_modes.png")

    # -------------------------------------------------------------------------
    # Create a bar chart of average execution time for each mode.
    avg_time_per_mode = df.groupby("Mode")["ExecutionTimeSeconds"].mean().sort_values()

    # Normaler Balkendiagramm-Plot
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    avg_time_per_mode.plot(
        kind="bar",
        color=[plt.cm.viridis(i / len(avg_time_per_mode)) for i in range(len(avg_time_per_mode))],
        edgecolor="black",
        ax=ax1,
        zorder=3
    )
    ax1.set_title("Average Execution Time by Mode")
    ax1.set_xlabel("Mode")
    ax1.set_ylabel("Execution Time (s)")
    ax1.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
        
    # Speichern des normalen Plots
    save_plot(fig1, "barchart_avg_execution_time_per_mode.png")

    # -------------------------------------------------------------------------
    # Zweiter Plot mit logarithmischer y-Achse
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    avg_time_per_mode.plot(
        kind="bar",
        color=[plt.cm.viridis(i / len(avg_time_per_mode)) for i in range(len(avg_time_per_mode))],
        edgecolor="black",
        ax=ax2,
        zorder=3
    )
    ax2.set_yscale("log")  # Setzt die y-Achse auf logarithmische Skalierung
    ax2.set_title("Average Execution Time by Mode (Log Scale)")
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("Execution Time (s, log scale)")
    ax2.grid(axis='y', which='both', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)

    # Speichern des logarithmischen Plots
    save_plot(fig2, "barchart_avg_execution_time_per_mode_logscale.png")    


    # -------------------------------------------------------------------------
    # Create a bar chart for the average execution time for OpenCL modes.
    opencl_modes = df[df["Mode"].str.startswith("opencl")]
    avg_time_opencl = opencl_modes.groupby("Mode")["ExecutionTimeSeconds"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    avg_time_opencl.plot(
        kind="bar",
        color=[plt.cm.viridis(i / len(avg_time_opencl)) for i in range(len(avg_time_opencl))],
        edgecolor="black",
        ax=ax2,
        zorder=3
    )
    ax2.set_title("Average Execution Time for OpenCL Modes")
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("Execution Time (s)")
    ax2.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig2, "barchart_avg_execution_time_opencl_modes.png")

    # -------------------------------------------------------------------------
    # Line plot: Execution time vs. Total Pixels, differentiated by Mode.
    fig1, ax1 = plt.subplots(figsize=(10, 4))

    for i, (mode, group) in enumerate(df.groupby("Mode")):
        group = group.sort_values("TotalPixels")
        x = group["TotalPixels"]
        y = group["ExecutionTimeSeconds"]

        color = plt.cm.viridis(i / len(df["Mode"].unique()))
        ax1.plot(x, y, label=mode, marker="o", color=color)

    ax1.set_title("Execution Time vs. Total Pixels by Mode")
    ax1.set_xlabel("Total Pixels")
    ax1.set_ylabel("Execution Time (s)")
    ax1.legend(title="Mode")
    ax1.grid(color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)

    save_plot(fig1, "lineplot_execution_time_vs_total_pixels.png")

    # -------------------------------------------------------------------------
    # Line plot with log-scale y-axis
    fig2, ax2 = plt.subplots(figsize=(10, 4))

    for i, (mode, group) in enumerate(df.groupby("Mode")):
        group = group.sort_values("TotalPixels")
        x = group["TotalPixels"]
        y = group["ExecutionTimeSeconds"]

        color = plt.cm.viridis(i / len(df["Mode"].unique()))
        ax2.plot(x, y, label=mode, marker="o", color=color)

    ax2.set_yscale("log")
    ax2.set_title("Execution Time vs. Total Pixels by Mode (Log Scale)")
    ax2.set_xlabel("Total Pixels")
    ax2.set_ylabel("Execution Time (s, log scale)")
    ax2.legend(title="Mode")
    ax2.grid(color='lightgray', linestyle='-', linewidth=0.5, which='both', zorder=-1)

    save_plot(fig2, "lineplot_execution_time_vs_total_pixels_logscale.png")


    # -------------------------------------------------------------------------
    # Line plot: Execution time vs. Total Pixels for OpenCL modes only.
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    opencl_modes = df[df["Mode"].str.startswith("opencl")]
    for i, (mode, group) in enumerate(opencl_modes.groupby("Mode")):
        group = group.sort_values("TotalPixels")
        x = group["TotalPixels"]
        y = group["ExecutionTimeSeconds"]
        color = plt.cm.viridis(i / len(opencl_modes["Mode"].unique()))
        ax2.plot(x, y, label=mode, marker="o", color=color)
    ax2.set_title("Execution Time vs. Total Pixels by OpenCL Mode")
    ax2.set_xlabel("Total Pixels")
    ax2.set_ylabel("Execution Time (s)")
    ax2.legend(title="Mode")
    ax2.grid(color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig2, "lineplot_execution_time_vs_total_pixels_opencl_modes.png")

    # -------------------------------------------------------------------------
    # Grouped bar chart: Average execution time by image category and mode.
    # Extract the image category from the input file path using a regular expression.
    df["Category"] = df["InputFile"].str.extract(r'images/([a-zA-Z]+)')[0]
    grouped = df.groupby(["Category", "Mode"])["ExecutionTimeSeconds"].mean().unstack()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    grouped.plot(
        kind="bar", 
        stacked=False, 
        ax=ax1, 
        colormap="viridis", 
        width=0.8, 
        zorder=3, 
        edgecolor="black"
    )
    ax1.set_title("Execution Time by Category and Mode")
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Execution Time (s)")
    ax1.legend(title="Mode")
    ax1.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig1, "grouped_barchart_execution_time_by_category.png")

    # -------------------------------------------------------------------------
    # Grouped bar chart: Average execution time by image category for OpenCL modes.
    opencl_modes = df[df["Mode"].str.startswith("opencl")]
    grouped_opencl = opencl_modes.groupby(["Category", "Mode"])["ExecutionTimeSeconds"].mean().unstack()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    grouped_opencl.plot(
        kind="bar", 
        stacked=False, 
        ax=ax2, 
        colormap="viridis", 
        width=0.8, 
        zorder=3, 
        edgecolor="black"
    )
    ax2.set_title("Execution Time by Category and OpenCL Mode")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Execution Time (s)")
    ax2.legend(
        title="Mode", 
        loc="upper right", 
    )
    ax2.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig2, "grouped_barchart_execution_time_opencl_by_category.png")

    # -------------------------------------------------------------------------
    # Heatmap: Execution time by category and mode.
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    heatmap_data = grouped.fillna(0).T

    # Reorder categories as desired.
    desired_order = ["animal", "nature", "dice", "human"]
    heatmap_data = heatmap_data[desired_order]

    cax1 = ax1.matshow(heatmap_data, cmap="viridis", aspect="auto", zorder=-1)
    fig1.colorbar(cax1)
    ax1.set_xticks(range(len(heatmap_data.columns)))
    ax1.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
    ax1.set_yticks(range(len(heatmap_data.index)))
    ax1.set_yticklabels(heatmap_data.index)
    ax1.set_title("Heatmap of Execution Time by Category and Mode")
    save_plot(fig1, "heatmap_execution_time_by_category_and_mode.png")

    # -------------------------------------------------------------------------
    # Heatmap: Execution time by category for OpenCL modes.
    opencl_modes = df[df["Mode"].str.startswith("opencl")]
    grouped_opencl = opencl_modes.groupby(["Category", "Mode"])["ExecutionTimeSeconds"].mean().unstack()
    heatmap_opencl_data = grouped_opencl.fillna(0).T

    # Apply the desired category order.
    heatmap_opencl_data = heatmap_opencl_data[desired_order]

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    cax2 = ax2.matshow(heatmap_opencl_data, cmap="viridis", aspect="auto", zorder=-1)
    fig2.colorbar(cax2)
    ax2.set_xticks(range(len(heatmap_opencl_data.columns)))
    ax2.set_xticklabels(heatmap_opencl_data.columns, rotation=45, ha="right")
    ax2.set_yticks(range(len(heatmap_opencl_data.index)))
    ax2.set_yticklabels(heatmap_opencl_data.index)
    ax2.set_title("Heatmap of Execution Time by Category and OpenCL Mode")
    save_plot(fig2, "heatmap_execution_time_opencl_by_category_and_mode.png")

if __name__ == "__main__":
    # Run the CSV analysis and plotting when the script is executed.
    analyze_csv()
