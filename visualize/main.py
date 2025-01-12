import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Path to the CSV file
csv_path = "../output/results.csv"
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

def save_plot(fig, filename):
    """Save a plot to the output directory with high dpi."""
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {path}")

def analyze_csv():
    """Load the CSV file, calculate image sizes, and generate plots."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"The file {csv_path} was not found.")
        return

    print("Data loaded successfully.")

    # Dictionary to store image dimensions and total pixel count
    image_sizes = {}

    # Iterate over unique input files
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

    # Add total pixels to the DataFrame
    df["TotalPixels"] = df["InputFile"].map(lambda x: image_sizes[x]["total_pixels"] if x in image_sizes else None)

    # Plot 1: Boxplot for average, min, max execution time per mode
    fig, ax = plt.subplots(figsize=(10, 4))
    df.boxplot(column="ExecutionTimeSeconds", by="Mode", ax=ax, grid=False, patch_artist=True, 
            boxprops=dict(facecolor=plt.cm.viridis(0.2)))
    ax.set_title("Execution Time Distribution by Mode")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Execution Time (s)")
    ax.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Texte rotieren
    plt.suptitle("")
    save_plot(fig, "boxplot_execution_time_per_mode.png")


    # Plot 2: Bar chart for average execution time by mode
    avg_time_per_mode = df.groupby("Mode")["ExecutionTimeSeconds"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    avg_time_per_mode.plot(kind="bar", color=[plt.cm.viridis(i / len(avg_time_per_mode)) for i in range(len(avg_time_per_mode))], 
                           edgecolor="black", ax=ax, zorder=3)
    ax.set_title("Average Execution Time by Mode")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Execution Time (s)")
    ax.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig, "barchart_avg_execution_time_per_mode.png")

    # Plot 3: Smoothed line plot for execution time vs. total pixels by mode
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (mode, group) in enumerate(df.groupby("Mode")):
        group = group.sort_values("TotalPixels")
        x = group["TotalPixels"]
        y = group["ExecutionTimeSeconds"]
        color = plt.cm.viridis(i / len(df["Mode"].unique()))  # Farbauswahl basierend auf Modus-Index
        ax.plot(x, y, label=mode, marker="o", color=color)  # Linie mit 'viridis'-Farbe
    ax.set_title("Execution Time vs. Total Pixels by Mode")
    ax.set_xlabel("Total Pixels")
    ax.set_ylabel("Execution Time (s)")
    ax.legend(title="Mode")
    ax.grid(color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig, "lineplot_execution_time_vs_total_pixels.png")


    # Plot 4: Grouped bar chart for execution time by image type
    df["Category"] = df["InputFile"].str.extract(r'images/([a-zA-Z]+)')[0]
    grouped = df.groupby(["Category", "Mode"])["ExecutionTimeSeconds"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 4))
    grouped.plot(kind="bar", stacked=False, ax=ax, colormap="viridis", width=0.8, zorder=3, edgecolor="black")
    ax.set_title("Execution Time by Category and Mode")
    ax.set_xlabel("Category")
    ax.set_ylabel("Execution Time (s)")
    ax.legend(title="Mode")
    ax.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=-1)
    save_plot(fig, "grouped_barchart_execution_time_by_category.png")

    # Plot 5: Heatmap for execution time (Category vs. Mode)
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap_data = grouped.fillna(0).T

    # Reorder columns in the desired order
    desired_order = ["animal", "nature", "dice", "human"]
    heatmap_data = heatmap_data[desired_order]

    print(heatmap_data)
    cax = ax.matshow(heatmap_data, cmap="viridis", aspect="auto", zorder=-1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set_title("Heatmap of Execution Time by Category and Mode")
    save_plot(fig, "heatmap_execution_time_by_category_and_mode.png")

if __name__ == "__main__":
    analyze_csv()
