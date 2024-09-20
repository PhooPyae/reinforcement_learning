import gymnasium as gym
import torch
import numpy as np
import wandb

import config as Config
from agent import Agent
from utils import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config.config

run = wandb.init(
    project='lunar_lander',
    config=config
)

if __name__ == '__main__':
    reward_history = []
    torch.manual_seed(config["seed"])

    env = gym.make('LunarLander-v2')
    envs = gym.make_vec("LunarLander-v2", num_envs=8)

    logger.debug(f'Observation Space {env.observation_space.shape}, Action Sapce {env.action_space.n}')
    
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    logger.info(f'Using {agent.device}')
    
    scores = []
    states, _ = envs.reset()
    done = False
    env_step = 0
    N = 128
    reward_history = []

    while env_step < config["num_steps"]:
        actions, probs, entropy, values = agent.choose_action(states)
        next_states, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)
        
        rewards = rewards / 100.
        agent.memory.store_memory(states, actions, probs, values, rewards, dones)
        
        states = next_states
        env_step += 1
        reward_history.append(np.sum(rewards))

        if env_step % N == 0:
            actor_loss, critic_loss, total_loss = agent.learn()
            eval_reward = evaluate(env, agent)

            avg_reward = np.mean(reward_history[-100:])

            logger.info(f'Environment Step: {env_step} : Average Reward {avg_reward:.2f} : Eval Reward {eval_reward:.2f} : Actor Loss {actor_loss:.2f} : Critic Loss {critic_loss:.2f} : Total Loss {total_loss:.2f}')
            wandb.log({"env_steps": env_step, "reward": avg_reward, "eval_reward": eval_reward, "actor_loss": actor_loss, "critic_loss": critic_loss, "total_loss": total_loss, "entropy" : np.mean(entropy)})