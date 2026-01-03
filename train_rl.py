import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from collections import deque

from env import PongEnv
from agent import PongNet, STATE_DIM, NUM_ACTIONS

# --- Hyperparameters ---
LEARNING_RATE = 0.0005  # Smaller LR for fine-tuning
GAMMA = 0.99  # Discount factor for future rewards
NUM_ENVS = 32           # Number of parallel games (Batch Size)
BATCH_UPDATES = 1000    # Number of gradient updates to perform
# Total Games Played = NUM_ENVS * BATCH_UPDATES (e.g., 32 * 1000 = 32,000 games)

def train():
    # Initialize multiple environments
    envs = [PongEnv() for _ in range(NUM_ENVS)]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with {NUM_ENVS} parallel environments.")

    # Initialize Policy Network
    policy = PongNet(STATE_DIM, NUM_ACTIONS).to(device)

    # Load Teacher weights if available (Jumpstart training)
    if os.path.exists("pong_pretrained_teacher.pth"):
        policy.load_state_dict(torch.load("pong_pretrained_teacher.pth", map_location=device, weights_only=True))
        print("Loaded teacher weights. Fine-tuning with RL...")
    else:
        print("No teacher weights found. Training from scratch...")

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    # Metrics for logging
    reward_history = deque(maxlen=100)
    win_history = deque(maxlen=100)

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

        # Average the gradients across all environments
        (total_loss / NUM_ENVS).backward()
        optimizer.step()

        # Real-time logging
        total_games = update * NUM_ENVS
        avg_reward = sum(reward_history) / len(reward_history)
        win_rate = sum(win_history) / len(win_history) * 100
        
        # Use total_loss.item() / NUM_ENVS to see average loss per game
        avg_loss = total_loss.item() / NUM_ENVS
        print(f"Game {total_games:6d} | Avg Reward: {avg_reward:5.2f} | Win Rate: {win_rate:3.0f}% | Loss: {avg_loss:.2f}", end='\r', flush=True)

        if update % 10 == 0: # Save every 10 updates (320 games)
            print()  # Newline to preserve the log
            # Save the RL weights
            torch.save(policy.state_dict(), "pong_rl.pth")

    print("Training Complete. Saved to pong_rl.pth")


if __name__ == "__main__":
    train()
