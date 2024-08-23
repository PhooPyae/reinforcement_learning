import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import logging
from combine_video import combine_videos
import json
import numpy as np

logging.basicConfig(level=logging.WARNING)

action_map = ['Left', 'Down', 'Right', 'Up']
show_action = False

if __name__ == '__main__':
    env = gym.make("FrozenLake-v1", is_slippery=False)
    episodes = 1000
    output = []
    reward_over_episode = []
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = env.action_space.sample()
            if show_action:
                print(f'Action {action_map[action]}')
            obs_, reward, done, info, _ = env.step(action)
            total_reward += reward
        print(f'episode {episode}, total reward {total_reward:.1f}')
        reward_over_episode.append(total_reward)
    print(f'Average Reward: {np.mean(reward_over_episode)}')
