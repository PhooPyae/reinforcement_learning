import gymnasium as gym
import torch
import numpy as np

from config import Config as config
from agent import Agent

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(config.seed)
env = gym.make('LunarLander-v2')

def evaluate(agent):
  with torch.no_grad():
    loss = 0
    training_step = 0

    for i in range(3):
        state, _ = env.reset()
        done = False
        step = 0
        cumulative_reward = 0

        while not done and step < 1000:
            action = agent.choose_action(state)
            next_state, reward, done, info, _ = env.step(action)
            state = next_state
            cumulative_reward += reward
            step += 1

    return cumulative_reward/3

if __name__ == '__main__':
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    scores = []
    entropy = []

    for i in range(config.n_games):
        state, _ = env.reset()
        done = False
        env_step = 0
        score = 0
        losses = []

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info, _ = env.step(action)
            reward = reward / 7.0

            loss = agent.learn(state, next_state, reward)

            env_step += 1
            state = next_state
            score += reward

            entropy.append(agent.entropy.cpu().detach().numpy())
            losses.append(loss)

        scores.append(score)

        avg_score = np.mean(scores[-100:])
        avg_loss = np.mean(losses[-100:])
        avg_entropy = np.mean(entropy[-100:])

        eval_reward = evaluate(agent)

        if i % 10 == 0:
            print(f'Episode {i}, Loss : {avg_loss:.4f}, Reward : {avg_score:.4f}')