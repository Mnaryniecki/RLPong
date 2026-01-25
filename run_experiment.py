import torch
import os
import csv

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

    # --- Resume Logic ---
    start_step = 0
    best_avg_reward = -float('inf')
    mode = 'w'

    # Check if we can resume from an existing file
    if os.path.exists(csv_filename) and os.path.exists(f"{save_prefix}_rl.pth"):
        try:
            with open(csv_filename, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    if 'update_step' in last_row:
                        start_step = int(last_row['update_step'])
                        # Recover best reward to prevent overwriting best model with a worse one
                        best_avg_reward = max(float(r['avg_reward']) for r in rows)

                        if start_step < TOTAL_TRAINING_UPDATES:
                            print(f"Resuming from step {start_step}. Best reward so far: {best_avg_reward:.2f}")
                            mode = 'a'
                        else:
                            print("Experiment already completed.")
                            return
        except Exception as e:
            print(f"Could not resume (Error: {e}). Starting from scratch.")

    # Only clean up artifacts if we are starting fresh
    if mode == 'w':
        if os.path.exists(f"{save_prefix}_rl.pth"):
            os.remove(f"{save_prefix}_rl.pth")
        if os.path.exists(f"{save_prefix}_best.pth"):
            os.remove(f"{save_prefix}_best.pth")

    with open(csv_filename, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        if mode == 'w':
            csv_writer.writerow(['update_step', 'win_rate', 'avg_reward'])
            csvfile.flush()

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

        # Load weights based on mode
        if mode == 'a':
            print(f"Loading checkpoint weights: {save_prefix}_rl.pth")
            policy.load_state_dict(torch.load(f"{save_prefix}_rl.pth", map_location=device, weights_only=True))
        elif use_teacher and os.path.exists("pong_pretrained_teacher.pth"):
            print("Loading teacher weights to begin fine-tuning...")
            policy.load_state_dict(torch.load("pong_pretrained_teacher.pth", map_location=device, weights_only=True))

        optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
        envs = [PongEnv() for _ in range(NUM_ENVS)]
        
        # Loop starts from start_step (0 if fresh, last_step if resuming)
        for i in range(start_step, TOTAL_TRAINING_UPDATES, UPDATES_PER_EVAL):
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
        experiment_name="Training From Scratch Stochastic Evaluation",
        use_teacher=False,
        save_prefix="scratch_stochastic",
        csv_filename="results_scratch_stochastic.csv"
    )

    # Experiment 2: Train from Teacher
    run_single_experiment(
        experiment_name="Training From Teacher Stochastic Evaluation",
        use_teacher=True,
        save_prefix="teacher_stochastic",
        csv_filename="results_teacher_stochastic.csv"
    )


if __name__ == "__main__":
    main()