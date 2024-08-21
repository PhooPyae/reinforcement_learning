import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch
from agent import REINFORCE
import numpy as np

from gymnasium.wrappers import RecordVideo

env = gym.make("BipedalWalker-v3",  render_mode="rgb_array")
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval",
                  episode_trigger=lambda x: True)
# wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
hidden_state = 128

states, actions, rewards = [], [], []
rewards_over_seeds = []
total_num_episodes = 1  # Total number of episodes
device = torch.device("cpu")

n_episodes = 1
agent = REINFORCE(state_space, hidden_state, action_space, device)
model = torch.load('bipedalwalker_policy.pth', map_location=torch.device('cpu'))
# print(model)
agent.net.load_state_dict(model)

actions = [0, 1, 2, 3]

# for episode in range(total_num_episodes):
obs, info = env.reset()
done = False

while not done:
    state = torch.tensor(obs, dtype=torch.float32).to(device)
    action = agent.choose_action(state)
    print(action)
    obs, reward, terminated, truncated, info = env.step(action.cpu().detach())
    agent.rewards.append(reward)

    done = terminated or truncated


if done:
    print(f"{reward=}")

env.close()
