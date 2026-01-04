import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import sys
import os
import argparse

def plot_from_csv(csv_path):
    """Reads experiment data from a CSV and generates a performance plot."""
    if not os.path.exists(csv_path):
        print(f"Error: File not found at '{csv_path}'")
        return

    # Read data using pandas
    data = pd.read_csv(csv_path)

    updates = data['update_step']
    win_rates = data['win_rate']
    avg_rewards = data['avg_reward']

    # Setup plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.suptitle('Training Performance Analysis', fontsize=16)

    # Axis for Win Rate
    ax1.set_xlabel('Training Updates')
    ax1.set_ylabel('Win Rate (%)', color='tab:blue')
    ax1.plot(updates, win_rates, 'o-', label='Win Rate', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 101)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Axis for Average Reward
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Reward (All Games)', color='tab:red')
    ax2.plot(updates, avg_rewards, 's--', label='Average Reward', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Auto-scale reward axis to ensure max value is visible
    data_min = avg_rewards.min()
    data_max = avg_rewards.max()
    margin = (data_max - data_min) * 0.1 if (data_max - data_min) > 0 else 0.5
    ax2.set_ylim(data_min - margin, data_max + margin)

    # --- Add horizontal lines for max values ---
    max_win_rate = win_rates.max()
    max_avg_reward = avg_rewards.max()
    ax1.axhline(y=max_win_rate, color='tab:cyan', linestyle=':', linewidth=2, label=f'Max Win Rate ({max_win_rate:.1f}%)')
    ax2.axhline(y=max_avg_reward, color='tab:orange', linestyle=':', linewidth=2, label=f'Max Avg Reward ({max_avg_reward:.2f})')

    # Add annotations for baseline models
    if len(updates) >= 2 and updates.iloc[1] == 0:
        ax1.text(updates.iloc[0], win_rates.iloc[0], ' Scratch', va='bottom', ha='center', color='blue')
        ax1.text(updates.iloc[1], win_rates.iloc[1], ' Teacher', va='top', ha='center', color='blue')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower right')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle

    # Save the plot
    output_filename = os.path.splitext(csv_path)[0] + '.png'
    plt.savefig(output_filename)
    print(f"Plot saved to '{output_filename}'")
    plt.show()

def main():
    """Parses command-line arguments and calls the appropriate plotting function."""
    parser = argparse.ArgumentParser(description="Generate a performance plot from experiment data.")
    parser.add_argument('csv_file', nargs='?', default=None, help="Path to the experiment results CSV file.")
    args = parser.parse_args()

    csv_path = args.csv_file
    if not csv_path:
        print("Usage: python plot_results.py <path_to_csv_file>")
        # Attempt to find and plot the default experiment CSV
        default_csv = "experiment_results.csv"
        if os.path.exists(default_csv):
            print(f"\nNo file specified. Plotting the default experiment file: '{default_csv}'")
            csv_path = default_csv

    if csv_path:
        plot_from_csv(csv_path)

if __name__ == "__main__":
    main()