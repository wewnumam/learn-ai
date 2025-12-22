import os
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

# =====================================================
# Path setup (save videos next to this file)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "cartpole-best")

# =====================================================
# Configuration
# =====================================================
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 5_000

# =====================================================
# Globals for recording logic
# =====================================================
best_reward_so_far = -float("inf")
record_next_episode = False  # whether the upcoming episode should be recorded

def best_reward_trigger(episode_id: int) -> bool:
    """
    Record ONLY if previous episode achieved a new best reward.
    """
    global record_next_episode
    return record_next_episode

# =====================================================
# Environment
# =====================================================
env = gym.make(ENV_NAME, render_mode="rgb_array")

env = RecordVideo(
    env,
    video_folder=VIDEO_DIR,
    name_prefix="best",
    episode_trigger=best_reward_trigger
)

env = RecordEpisodeStatistics(env)

print("Training started")
print("Recording only when best reward improves")
print(f"Videos saved to: {VIDEO_DIR}\n")

# =====================================================
# Training loop
# =====================================================
for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Replace with your agent
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    # -------------------------------------------------
    # Check for new best reward
    # -------------------------------------------------
    if episode_reward > best_reward_so_far:
        print(
            f"🏆 New BEST at episode {episode}: "
            f"{episode_reward:.1f} (prev {best_reward_so_far:.1f})"
        )
        best_reward_so_far = episode_reward
        record_next_episode = True   # record NEXT episode
    else:
        record_next_episode = False

    # Optional logging
    if "episode" in info:
        ep = info["episode"]
        print(
            f"Episode {episode:5d} | "
            f"reward={ep['r']:6.1f} | "
            f"length={ep['l']:4d}"
        )

env.close()

print("\nTraining finished")
print(f"Best reward achieved: {best_reward_so_far}")