import gymnasium as gym
import torch
import numpy as np

from config import Config as config
from agent import Agent

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    reward_history = []
    loss = 0
    training_step = 0
    torch.manual_seed(config.seed)
    
    env = gym.make('LunarLander-v2')
    logger.debug(f'Observation Space {env.observation_space.shape}, Action Sapce {env.action_space.n}')
    
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    scores = []
    for i in range(config.n_games):
        state, _ = env.reset()
        done = False
        env_step = 0
        score = 0
        
        while not done and env_step < 10:
            action = agent.choose_action(state)
            next_state, reward, dones, info, _ = env.step(action)
            state = next_state
            env_step += 1
            score += reward
            
            agent.rewards.append(reward)

        total_reward = np.mean(agent.rewards)
        loss = agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
                    
        print(f'Episode {i}, Loss : {loss:.4f}, Reward : {avg_score:.4f}')
        