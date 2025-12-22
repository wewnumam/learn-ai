import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

# 1. Initialize the TensorBoard Writer
# This will create a folder named 'runs' in your directory
writer = SummaryWriter("runs/lunar_lander_experiment_1")

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset(seed=42)

episode_reward = 0
episode_count = 0

for step_idx in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward

    # Log specific step details if you want (can be large file size)
    # writer.add_scalar("Reward/Step", reward, step_idx)

    if terminated or truncated:
        # 2. Log the primary metric: Episode Reward
        writer.add_scalar("Reward/Episode", episode_reward, episode_count)
        
        print(f"Episode {episode_count} logged to TensorBoard.")
        
        episode_reward = 0
        episode_count += 1
        observation, info = env.reset()

env.close()
writer.close()