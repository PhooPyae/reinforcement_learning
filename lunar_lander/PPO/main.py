import gymnasium as gym
import torch
import numpy as np

import config as Config
from agent import Agent
from utils import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    config = Config.config
    reward_history = []
    torch.manual_seed(config["seed"])

    env = gym.make('LunarLander-v2')
    envs = gym.vector.make("LunarLander-v2", num_envs=8)

    logger.debug(f'Observation Space {env.observation_space.shape}, Action Sapce {env.action_space.n}')
    
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    scores = []
    states, _ = envs.reset()
    done = False
    env_step = 0
    N = 20

    while env_step < config["num_envs"]:
        actions, probs, values = agent.choose_action(states)
        next_states, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)
        
        agent.memory.store_memory(states, actions, probs, values, rewards, dones)
        
        states = next_states
        env_step += 1

        if env_step % N == 0:
            loss = agent.learn()
            eval_reward = evaluate(env, agent)

            print(f'Environment Step: {env_step} : Loss {loss:.4f} : Reward {eval_reward:.4f}')
        