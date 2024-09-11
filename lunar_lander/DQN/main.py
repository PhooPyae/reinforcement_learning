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
    torch.manual_seed(config.seed)
    
    env = gym.make('LunarLander-v2')
    logger.debug(f'Observation Space {env.observation_space.shape}, Action Sapce {env.action_space.n}')
    
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    
    for i in range(config.n_games):
        state, _ = env.reset()
        done = False
        step = 0
        cumulative_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info, _ = env.step(action)
            
            agent.memory.add(state, action, reward, next_state, done)
            state = next_state
            cumulative_reward += reward
            
            if len(agent.memory.buffer) == config.capacity:
                loss = agent.learn()
                agent.memory.buffer.clear()
                
            if step % 3000 == 0:
                agent.q_target_network.load_state_dict(agent.q_network.state_dict())
        
        reward_history.append(cumulative_reward)
        
        if i % 10 == 0:
            logger.info(f'Episode {i} : Loss {loss:.4f} : Reward: {np.mean(reward_history[-10:]):.4f}')

            