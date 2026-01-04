import torch
import os
import csv
from datetime import datetime

from agent import PongNet, STATE_DIM, NUM_ACTIONS
from env import PongEnv
from eval import evaluate_parallel
from train_rl import run_training_updates, LEARNING_RATE, NUM_ENVS

# --- Experiment Configuration ---
TOTAL_TRAINING_UPDATES = 200  # Total number of training batches to run
UPDATES_PER_EVAL = 10        # How often to evaluate the model (should match save frequency)
EVAL_ENVS = 128               # Number of parallel environments for evaluation
EVAL_EPISODES = 10            # Number of games each environment plays during evaluation


def main():
    """Orchestrates the training and evaluation experiment."""
    # --- Clean up artifacts from previous runs for a clean experiment ---
    print("--- Cleaning up previous experiment artifacts ---")
    if os.path.exists("pong_rl.pth"):
        os.remove("pong_rl.pth")
    if os.path.exists("pong_best.pth"):
        os.remove("pong_best.pth")
    print("Cleanup complete.")
    
    # --- Setup CSV Logging ---
    csv_filename = "experiment_results.csv"
    print(f"--- Logging results to {csv_filename} ---")

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
        if os.path.exists("pong_pretrained_teacher.pth"):
            print("\n--- Evaluating Pre-trained Teacher Model ---")
            win_rate, avg_reward = evaluate_parallel(weights_file="pong_pretrained_teacher.pth", num_envs=EVAL_ENVS, episodes=EVAL_EPISODES, verbose=False)
            csv_writer.writerow([0, win_rate, avg_reward])
            csvfile.flush()

        # --- Stage 3: Incremental RL Training and Evaluation ---
        print("\n--- Starting Incremental RL Training ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = PongNet(STATE_DIM, NUM_ACTIONS).to(device)
        
        # Start fine-tuning from the teacher model if it exists
        if os.path.exists("pong_pretrained_teacher.pth"):
            print("Loading teacher weights to begin fine-tuning...")
            policy.load_state_dict(torch.load("pong_pretrained_teacher.pth", map_location=device, weights_only=True))

        optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
        envs = [PongEnv() for _ in range(NUM_ENVS)]
        best_avg_reward = -float('inf')

        for i in range(0, TOTAL_TRAINING_UPDATES, UPDATES_PER_EVAL):
            print(f"\n--- Training updates {i + 1} to {i + UPDATES_PER_EVAL} ---")
            best_avg_reward = run_training_updates(policy, optimizer, envs, device, i + 1, UPDATES_PER_EVAL, best_avg_reward)
            
            print(f"\n--- Evaluating model after {i + UPDATES_PER_EVAL} updates ---")
            win_rate, avg_reward = evaluate_parallel(weights_file="pong_rl.pth", num_envs=EVAL_ENVS, episodes=EVAL_EPISODES, verbose=False)
            
            csv_writer.writerow([i + UPDATES_PER_EVAL, win_rate, avg_reward])
            csvfile.flush()

    print("\n--- Experiment Complete ---")
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    main()