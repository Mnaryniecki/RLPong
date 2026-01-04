from agent import PongAgent
from env import PongEnv
import time

def evaluate_parallel(num_envs=8, episodes=20):
    agent = PongAgent(weights_file="pong_best.pth")
    right_wins = 0
    left_wins = 0
    timeouts = 0

    total_games = num_envs * episodes

    # Create environments once outside the loop for efficiency
    envs = [PongEnv() for _ in range(num_envs)]

    for ep in range(episodes):
        states = [env.reset() for env in envs]
        dones = [False] * num_envs

        while not all(dones):
            # Get a batch of actions from the agent (runs on GPU)
            actions_tensor = agent.act_batch(states, stochastic=False) # Use greedy for eval
            # Move all actions to CPU at once for the environments
            actions_np = actions_tensor.cpu().numpy()

            for i, env in enumerate(envs):
                if dones[i]:
                    continue
                next_state, reward, done, info = env.step(actions_np[i])
                states[i] = next_state
                dones[i] = done

                if done:
                    if info["winner"] == "right":
                        right_wins += 1
                    elif info["winner"] == "left":
                        left_wins += 1
                    elif info["winner"] == "timeout":
                        timeouts += 1

        # Real-time progress update
        current_games = (ep + 1) * num_envs
        current_win_rate = (right_wins / current_games) * 100 if current_games > 0 else 0
        print(f"Round {ep+1}/{episodes} | Games: {current_games}/{total_games} | Agent Win Rate: {current_win_rate:.1f}%  ", end='\r')

    print("\n" + "="*35)
    print("Evaluation results:")
    print("="*35)

    right_win_rate = (right_wins / total_games) * 100 if total_games > 0 else 0
    left_win_rate = (left_wins / total_games) * 100 if total_games > 0 else 0
    timeout_rate = (timeouts / total_games) * 100 if total_games > 0 else 0

    print(f"Right wins (Agent): {right_wins:>5}/{total_games} ({right_win_rate:5.1f}%)")
    print(f"Left wins (Enemy):  {left_wins:>5}/{total_games} ({left_win_rate:5.1f}%)")
    print(f"Timeouts:           {timeouts:>5}/{total_games} ({timeout_rate:5.1f}%)")

if __name__ == "__main__":
    start = time.perf_counter()
    num_envs = 256
    episodes = 100
    evaluate_parallel(num_envs, episodes)
    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Time per environment: {elapsed/(num_envs*episodes):.2f} seconds")