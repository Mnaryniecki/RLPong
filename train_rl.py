import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from collections import deque

from env import PongEnv
from agent import PongNet, STATE_DIM, NUM_ACTIONS

# --- Hyperparameters ---
LEARNING_RATE = 1e-4      # Lower learning rate for more stable fine-tuning
GAMMA = 0.99              # Discount factor for future rewards
ENTROPY_COEFF = 0.01      # Coefficient for the entropy bonus to encourage exploration
NUM_ENVS = 256          # Number of parallel games (Batch Size)
BATCH_UPDATES = 1000    # Number of gradient updates to perform
TRAINING_MODE = 'continue' # Options: 'continue' (loads pong_rl.pth), 'teacher' (loads pong_pretrained_teacher.pth), 'scratch'
# Total Games Played = NUM_ENVS * BATCH_UPDATES (e.g., 32 * 1000 = 32,000 games)

def train():
    # Initialize multiple environments
    envs = [PongEnv() for _ in range(NUM_ENVS)]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with {NUM_ENVS} parallel environments.")

    # Initialize Policy Network
    policy = PongNet(STATE_DIM, NUM_ACTIONS).to(device)

    # --- Weight Loading with Waterfall Logic ---
    weights_loaded = False

    # Default to 'continue' logic if the mode is invalid
    effective_mode = TRAINING_MODE if TRAINING_MODE in ['continue', 'teacher', 'scratch'] else 'continue'
    if effective_mode != TRAINING_MODE:
        print(f"\nWarning: Invalid TRAINING_MODE '{TRAINING_MODE}'. Defaulting to 'continue'.")

    if effective_mode == 'continue':
        if os.path.exists("pong_rl.pth"):
            print("\nLoading weights from last session (pong_rl.pth) to continue training...")
            policy.load_state_dict(torch.load("pong_rl.pth", map_location=device, weights_only=True))
            weights_loaded = True
        else:
            print("\nCould not find 'pong_rl.pth'. Falling back to teacher model...")

    if effective_mode in ['continue', 'teacher'] and not weights_loaded:
        if os.path.exists("pong_pretrained_teacher.pth"):
            print("\nLoading teacher weights (pong_pretrained_teacher.pth) to start fine-tuning...")
            policy.load_state_dict(torch.load("pong_pretrained_teacher.pth", map_location=device, weights_only=True))
            weights_loaded = True
        elif effective_mode == 'teacher': # Only print if teacher was the primary goal
            print("\nCould not find 'pong_pretrained_teacher.pth'. Falling back to scratch...")

    if not weights_loaded:
        print("\nStarting training from scratch with random weights...")

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    # Metrics for logging
    reward_history = deque(maxlen=100)
    win_history = deque(maxlen=100)
    best_avg_reward = -float('inf')

    for update in range(1, BATCH_UPDATES + 1):
        # Reset all environments for the new batch
        states = [env.reset() for env in envs]

        # Storage for the current batch of episodes
        # We need a list of lists because each env has its own trajectory
        batch_log_probs = [[] for _ in range(NUM_ENVS)]
        batch_rewards = [[] for _ in range(NUM_ENVS)]
        
        # Track which envs are still running in this batch
        active_envs = [True] * NUM_ENVS

        while any(active_envs):
            # 1. Batch Inference (GPU Speedup happens here)
            # We convert ALL states to a tensor (even done ones, to keep shape constant)
            state_t = torch.tensor(states, dtype=torch.float32, device=device)

            # Get Action Probabilities for all 32 envs at once
            logits = policy(state_t) # Shape: [NUM_ENVS, 3]
            dist = Categorical(logits=logits)
            actions = dist.sample()  # Shape: [NUM_ENVS]

            # We need log probabilities for the gradient
            step_log_probs = dist.log_prob(actions)

            # Move actions to CPU for the environment
            actions_np = actions.cpu().numpy()

            # 2. Step Environments
            for i in range(NUM_ENVS):
                if not active_envs[i]:
                    continue

                # Store log prob for this specific environment
                batch_log_probs[i].append(step_log_probs[i])

                # Step the specific environment
                next_state, reward, done, info = envs[i].step(actions_np[i])

                batch_rewards[i].append(reward)
                states[i] = next_state

                if done:
                    active_envs[i] = False
                    # Record metrics
                    reward_history.append(sum(batch_rewards[i]))
                    win_history.append(1 if info["winner"] == "right" else 0)

        # --- Batch Finished: Update Policy ---
        optimizer.zero_grad()
        total_loss = 0

        # Calculate loss for each of the 32 trajectories
        for i in range(NUM_ENVS):
            rewards = batch_rewards[i]
            log_probs = batch_log_probs[i]

            # Calculate Discounted Returns (G_t)
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)
            
            returns = torch.tensor(returns, device=device)

            # Normalize returns (Crucial for parallel training stability)
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            else:
                returns = returns - returns.mean()

            # Loss for this single trajectory
            log_probs_stack = torch.stack(log_probs)
            policy_loss = (-log_probs_stack * returns).sum()
            total_loss += policy_loss
        
        # --- Calculate Final Loss and Update ---
        # We add an entropy bonus to encourage exploration and prevent premature convergence.
        # The distribution 'dist' is from the last step, but it's a reasonable proxy for the batch's average entropy.
        entropy_bonus = dist.entropy().mean()
        
        # Average the policy loss and subtract the entropy bonus (we want to maximize entropy, so we subtract it from the loss)
        final_loss = (total_loss / NUM_ENVS) - (ENTROPY_COEFF * entropy_bonus)
        
        final_loss.backward()
        optimizer.step()

        # Real-time logging
        total_games = update * NUM_ENVS
        avg_reward = sum(reward_history) / len(reward_history)
        win_rate = sum(win_history) / len(win_history) * 100

        avg_loss = final_loss.item()
        print(f"Game {total_games:6d} | Avg Reward: {avg_reward:5.2f} | Win Rate: {win_rate:3.0f}% | Loss: {avg_loss:.2f} | Entropy: {entropy_bonus.item():.2f}", end='\r', flush=True)

        if update % 10 == 0: # Save every 10 updates (2560 games)
            print()  # Newline to preserve the log
            # Save the RL weights
            torch.save(policy.state_dict(), "pong_rl.pth")

            # Check for new best model and save it separately
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                print(f"New best model found! Avg Reward: {avg_reward:.2f}. Saving to pong_best.pth...")
                torch.save(policy.state_dict(), "pong_best.pth")

    print("Training Complete. Saved to pong_rl.pth")


if __name__ == "__main__":
    train()
