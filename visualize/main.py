import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file
csv_path = "../output/results.csv"

def main():
    """
    Main function to read a CSV file, process the data, and create a plot
    showing the average execution time for each mode. The plot is saved
    as an image file in the /output directory.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"The file {csv_path} was not found.")
        return

    # Display a preview of the data
    print("Data preview:")
    print(df.head())

    # Aggregate data: Calculate average execution time per mode
    mode_times = df.groupby("Mode")["ExecutionTimeSeconds"].mean().sort_values()

    # Create a modern, clean plot
    plt.figure(figsize=(10, 6))
    mode_times.plot(kind="bar", color=plt.cm.viridis(range(len(mode_times))), edgecolor="black")

    plt.title("Average Execution Time per Mode", fontsize=16)
    plt.xlabel("Mode", fontsize=12)
    plt.ylabel("Average Time (Seconds)", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot to a file
    output_path = "./plots/average_execution_time.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved as {output_path}")

if __name__ == "__main__":
    main()
