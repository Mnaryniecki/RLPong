from agent import PongAgent
from env import PongEnv
import time

def evaluate_parallel(num_envs=8, episodes=20):
    agent = PongAgent()
    right_wins = 0
    left_wins = 0
    timeouts = 0

    total_games = num_envs * episodes

    for ep in range(episodes):
        envs = [PongEnv() for _ in range(num_envs)]
        states = [env.reset() for env in envs]
        dones = [False] * num_envs

        while not all(dones):
            actions = agent.act_batch(states)  # list of ints
            for i, env in enumerate(envs):
                if dones[i]:
                    continue
                state = states[i]
                action = actions[i]
                next_state, reward, done, info = env.step(action)
                states[i] = next_state
                dones[i] = done

                if done:
                    if info["winner"] == "right":
                        right_wins += 1
                    elif info["winner"] == "left":
                        left_wins += 1
                    elif info["winner"] == "timeout":
                        timeouts += 1

        print(f"Finished round {ep+1}/{episodes}")

    print("Evaluation results:")
    print(f"Total games: {total_games}")
    print(f"Right wins: {right_wins}/{total_games}")
    print(f"Left wins: {left_wins}/{total_games}")
    print(f"Timeouts: {timeouts}")

if __name__ == "__main__":
    start = time.perf_counter()
    num_envs = 20
    episodes = 10
    evaluate_parallel(num_envs, episodes)
    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Time per environment: {elapsed/(num_envs*episodes):.2f} seconds")