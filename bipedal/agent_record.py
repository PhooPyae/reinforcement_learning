import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

num_eval_episodes = 4

env = gym.make("-v1", render_mode="rgb_array")  # replace with your environment
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval",
                  episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env, deque_size=num_eval_episodes)

obs, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # replace with actual agent
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated
    env.render()
print('-' * 100)
env.close()
