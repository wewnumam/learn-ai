import gymnasium as gym
import matplotlib.pyplot as plt
import csv
import os

ENV_NAME = "MountainCar-v0"
EPISODES = 50
MAX_STEPS = 1000

env = gym.make(ENV_NAME)

episode_rewards = []

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mountaincar_log.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "total_reward"])

    for episode in range(EPISODES):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        writer.writerow([episode, total_reward])

        print(f"[MountainCar] Episode {episode}, Reward: {total_reward}")

env.close()

# Visualization
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("MountainCar-v0 Reward per Episode")
plt.grid()
plt.show()
