import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ENVIRONMENT
# ==========================================
env = gym.make("CartPole-v1")

# ==========================================
# DISCRETIZATION
# ==========================================
# CartPole state:
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

NUM_BINS = (6, 6, 12, 12)
LOWER_BOUNDS = [
    -2.4,    # cart position
    -3.0,    # cart velocity
    -0.418,  # pole angle (~24 deg)
    -3.5     # pole angular velocity
]
UPPER_BOUNDS = [
    2.4,
    3.0,
    0.418,
    3.5
]

def discretize_state(state):
    ratios = [
        (state[i] - LOWER_BOUNDS[i]) / (UPPER_BOUNDS[i] - LOWER_BOUNDS[i])
        for i in range(len(state))
    ]
    bins = [
        int(np.clip(ratios[i] * NUM_BINS[i], 0, NUM_BINS[i] - 1))
        for i in range(len(state))
    ]
    return tuple(bins)

num_states = np.prod(NUM_BINS)
num_actions = env.action_space.n

# ==========================================
# HYPERPARAMETER
# ==========================================
alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 1000

# ==========================================
# Q-TABLE
# ==========================================
Q = np.zeros(NUM_BINS + (num_actions,))

# ==========================================
# EPSILON GREEDY
# ==========================================
def epsilon_greedy_policy(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(Q[state])

# ==========================================
# LOGGING
# ==========================================
episode_rewards = []
episode_lengths = []

# ==========================================
# TRAINING SARSA
# ==========================================
for episode in range(num_episodes):
    state_continuous, _ = env.reset()
    state = discretize_state(state_continuous)

    action = epsilon_greedy_policy(state, Q, epsilon)

    done = False
    total_reward = 0
    steps = 0

    while not done:
        next_state_continuous, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = discretize_state(next_state_continuous)
        next_action = epsilon_greedy_policy(next_state, Q, epsilon)

        # SARSA UPDATE
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][next_action] - Q[state][action]
        )

        state = next_state
        action = next_action

        total_reward += reward
        steps += 1

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

    # LOG
    if (episode + 1) % 100 == 0:
        print(
            f"Episode {episode+1}/{num_episodes} | "
            f"Reward: {total_reward:.1f} | "
            f"Steps: {steps}"
        )

env.close()


# Learning Curve – Reward per Episode
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("SARSA on CartPole – Reward per Episode")
plt.grid(True)
plt.show()

# Panjang Episode
plt.figure()
plt.plot(episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Episode Length over Time")
plt.grid(True)
plt.show()

# Visualisasi Q-table (Slice Representatif)
# Ambil slice tengah dari state
mid_state = (
    NUM_BINS[0] // 2,
    NUM_BINS[1] // 2,
    NUM_BINS[2] // 2,
    NUM_BINS[3] // 2
)

plt.figure()
plt.bar([0, 1], Q[mid_state])
plt.xticks([0, 1], ["Left", "Right"])
plt.ylabel("Q-value")
plt.title("Q-value at Representative State")
plt.show()