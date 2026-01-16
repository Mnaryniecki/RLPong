import torch
import os
import csv
from datetime import datetime

from agent import PongNet, STATE_DIM, NUM_ACTIONS
from env import PongEnv
from eval import evaluate_parallel
from train_rl import run_training_updates, LEARNING_RATE, NUM_ENVS

# --- Experiment Configuration ---
TOTAL_TRAINING_UPDATES = 1024  # Total number of training batches to run
UPDATES_PER_EVAL = 16        # How often to evaluate the model (should match save frequency)
EVAL_ENVS = 128               # Number of parallel environments for evaluation
EVAL_EPISODES = 16            # Number of games each environment plays during evaluation


def run_single_experiment(experiment_name, use_teacher, save_prefix, csv_filename):
    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print(f"{'='*60}\n")

    # --- Clean up artifacts from previous runs for a clean experiment ---
    # We don't delete files here to avoid deleting results from the previous experiment in the loop

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['update_step', 'win_rate', 'avg_reward'])
        csvfile.flush()  # Ensure header is written immediately

        # --- Stage 1: Evaluate Empty Model (Scratch) ---
        print("--- Evaluating Empty Model (Scratch) ---")
        win_rate, avg_reward = evaluate_parallel(weights_file='scratch', num_envs=EVAL_ENVS, episodes=EVAL_EPISODES, verbose=False)
        csv_writer.writerow([-UPDATES_PER_EVAL, win_rate, avg_reward])
        csvfile.flush()

        # --- Stage 2: Evaluate Teacher Model ---
        if use_teacher and os.path.exists("pong_pretrained_teacher.pth"):
            print("\n--- Evaluating Pre-trained Teacher Model ---")
            win_rate, avg_reward = evaluate_parallel(weights_file="pong_pretrained_teacher.pth", num_envs=EVAL_ENVS, episodes=EVAL_EPISODES, verbose=False)
            csv_writer.writerow([0, win_rate, avg_reward])
            csvfile.flush()

        # --- Stage 3: Incremental RL Training and Evaluation ---
        print("\n--- Starting Incremental RL Training ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = PongNet(STATE_DIM, NUM_ACTIONS).to(device)

        # Start fine-tuning from the teacher model if it exists
        if use_teacher and os.path.exists("pong_pretrained_teacher.pth"):
            print("Loading teacher weights to begin fine-tuning...")
            policy.load_state_dict(torch.load("pong_pretrained_teacher.pth", map_location=device, weights_only=True))

        optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
        envs = [PongEnv() for _ in range(NUM_ENVS)]
        best_avg_reward = -float('inf')

        for i in range(0, TOTAL_TRAINING_UPDATES, UPDATES_PER_EVAL):
            print(f"\n--- Training updates {i + 1} to {i + UPDATES_PER_EVAL} ---")
            best_avg_reward = run_training_updates(policy, optimizer, envs, device, i + 1, UPDATES_PER_EVAL, best_avg_reward, save_prefix=save_prefix)
            
            print(f"\n--- Evaluating model after {i + UPDATES_PER_EVAL} updates ---")
            win_rate, avg_reward = evaluate_parallel(weights_file=f"{save_prefix}_rl.pth", num_envs=EVAL_ENVS, episodes=EVAL_EPISODES, verbose=False)
            
            csv_writer.writerow([i + UPDATES_PER_EVAL, win_rate, avg_reward])
            csvfile.flush()

    print(f"\n--- {experiment_name} Complete. Results saved to {csv_filename} ---")

def main():
    # Experiment 1: Train from Scratch
    run_single_experiment(
        experiment_name="Training From Scratch",
        use_teacher=False,
        save_prefix="scratch",
        csv_filename="results_scratch.csv"
    )

    # Experiment 2: Train from Teacher
    run_single_experiment(
        experiment_name="Training From Teacher",
        use_teacher=True,
        save_prefix="teacher",
        csv_filename="results_teacher.csv"
    )


if __name__ == "__main__":
    main()