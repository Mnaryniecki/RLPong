import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import os
import argparse

def plot_multiple_csvs(csv_paths, show_reward=True):
    # Setup plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.suptitle('Training Performance Analysis', fontsize=16)

    ax2 = None
    if show_reward:
        # Axis for Average Reward (Right Y-Axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Average Reward', color='black')

    # Axis for Win Rate
    ax1.set_xlabel('Training Updates')
    ax1.set_ylabel('Win Rate (%)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 101)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']

    for i, csv_path in enumerate(csv_paths):
        if not os.path.exists(csv_path):
            print(f"Warning: File not found at '{csv_path}'")
            continue

        data = pd.read_csv(csv_path)
        updates = data['update_step']
        win_rates = data['win_rate']
        avg_rewards = data['avg_reward']

        label_base = os.path.splitext(os.path.basename(csv_path))[0].replace("results_", "").replace("_", " ").capitalize()
        color = colors[i % len(colors)]

        # Plot Win Rate (Solid Line)
        ax1.plot(updates, win_rates, linestyle='-', marker='o', markersize=4, label=f'{label_base} Win Rate', color=color, alpha=0.8)

        if show_reward:
            # Plot Avg Reward (Dashed Line)
            ax2.plot(updates, avg_rewards, linestyle='--', marker='x', markersize=4, label=f'{label_base} Reward', color=color, alpha=0.5)

    if show_reward:
        # Auto-scale reward axis
        ax2.set_ylim(-5.5, 5.5)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    if show_reward:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='lower right')
    else:
        ax1.legend(lines, labels, loc='lower right')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle

    # Save the plot
    output_filename = 'comparison_plot_stochastic_vs_greedy2.png'
    plt.savefig(output_filename)
    print(f"Plot saved to '{output_filename}'")
    plt.show()

def main():
    """Parses command-line arguments and calls the appropriate plotting function."""
    parser = argparse.ArgumentParser(description="Generate a performance plot from experiment data.")
    parser.add_argument('csv_files', nargs='*', help="Paths to the experiment results CSV files.")
    parser.add_argument('--no-reward', action='store_true', help="Do not plot Average Reward.")
    args = parser.parse_args()

    csv_paths = args.csv_files
    if not csv_paths:

        print("Usage: python plot_results.py <path_to_csv_file> [additional_files...]")
        return

    plot_multiple_csvs(csv_paths, show_reward=not args.no_reward)

if __name__ == "__main__":
    main()