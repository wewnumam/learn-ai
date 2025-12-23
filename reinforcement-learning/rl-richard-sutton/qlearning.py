# Q-learning with logging and visualization (CartPole)
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 200

num_actions = env.action_space.n

def discretize_state(state):
    return tuple(np.round(state, 1))

Q = {}
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    S = discretize_state(state)
    done = False
    total_reward = 0

    if S not in Q:
        Q[S] = np.zeros(num_actions)

    step = 0
    while not done:
        if np.random.rand() < epsilon:
            A = env.action_space.sample()
        else:
            A = np.argmax(Q[S])

        next_state, R, terminated, truncated, _ = env.step(A)
        done = terminated or truncated
        total_reward += R

        S_prime = discretize_state(next_state)
        if S_prime not in Q:
            Q[S_prime] = np.zeros(num_actions)

        Q[S][A] += alpha * (R + gamma * np.max(Q[S_prime]) - Q[S][A])

        S = S_prime
        step += 1

    episode_rewards.append(total_reward)

    # Log per episode
    print(f"Episode {episode+1:3d} | Steps: {step:3d} | Total Reward: {total_reward:.0f}")

env.close()

# Visualization
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Performance (CartPole)")
plt.show()
