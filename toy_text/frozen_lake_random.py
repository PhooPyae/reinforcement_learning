import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import logging
from combine_video import combine_videos
import json
from frozen_lake_q_learning import CustomRewardFrozenLake

logging.basicConfig(level=logging.WARNING)

action_map = ['Left', 'Down', 'Right', 'Up']
show_action = False

if __name__ == '__main__':
    env = gym.make("FrozenLake-v1",is_slippery=False)
    env = CustomRewardFrozenLake(env)
    episodes = 100
    output = []
    episode_rewards = []

    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        trajectory = []
        while not done:
            action = env.action_space.sample()
            if show_action:
                print(f'Action {action_map[action]}')
            obs_, reward, done, info, _ = env.step(action)

            total_reward += reward
        print(f'episode {episode}, total reward {total_reward:.1f}')
        episode_rewards.append(total_reward)
        
with open('episode_rewards_random.json', 'w', encoding='utf-8') as f:
    json.dump(episode_rewards, f, ensure_ascii=False, indent=4)
