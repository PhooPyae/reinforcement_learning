import torch
import copy
import wandb
import numpy as np
import random

from env_wrapper import SingleAgentSelfPlayEnv
from agent import Agent
from utils import evaluate_agent
import config as Config
from torch.utils.tensorboard import SummaryWriter
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config.config

wandb_project_name = 'connect_four'
run_name = f"ppo__{config['seed']}__{int(time.time())}"
    
wandb.init(
            project=wandb_project_name,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
)

random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

if __name__ == '__main__':
    writer = SummaryWriter()
    env_test = SingleAgentSelfPlayEnv(agent_starts=True)
    agent_policy = Agent(state_space=env_test.observation_space, action_space=env_test.action_space)
    opponent_policy = copy.deepcopy(agent_policy)
    env_test.set_policies(agent_policy=agent_policy, opponent_policy=opponent_policy)

    num_episodes = config["num_episodes"]
    max_steps_per_episode = config["max_steps_per_episode"]
    global_step = 0
    opponent_update_interval = config["opponent_update_interval"]  # Update every 10 episodes
    reward_history = []

    for episode in range(num_episodes):
        obs = env_test.reset()
        done = False
        total_reward = 0
        step = 0
        while not done and step < max_steps_per_episode:
            state = obs['observation'].reshape(-1)
            action_mask = obs['action_mask']

            action, prob, entropy, value = agent_policy.choose_action(state, action_mask)
            # logger.info(f'action {action}')
            next_obs, reward, done, info = env_test.step(action)

            agent_policy.memory.store_memory(state, action, prob,  value.item(), reward, done)

            obs = next_obs
            total_reward += reward
            step += 1
            global_step += 1
        
        reward_history.append(total_reward)

        actor_loss, critic_loss, total_loss = agent_policy.learn()
        writer.add_scalar('Loss/policy_loss', actor_loss, global_step)
        writer.add_scalar('Loss/value_loss', critic_loss, global_step)
        writer.add_scalar('Loss/total_loss', total_loss, global_step)
        writer.add_scalar('Loss/entropy', entropy, global_step)
        writer.add_scalar('Loss/value', value, global_step)
        writer.add_scalar('Loss/prob', prob, global_step)
        writer.add_scalar("charts/episode_reward", total_reward, global_step)
        writer.add_scalar("charts/episodic_length", step, global_step)


        logger.info(f"global_step={global_step}, episodic_return={total_reward}")
        
        if episode % opponent_update_interval == 0:
            opponent_policy = copy.deepcopy(agent_policy)
            env_test.set_policies(agent_policy=agent_policy, opponent_policy=opponent_policy)
            average_reward = np.mean(reward_history[-100:])
            logger.info(f'Episode {episode}: Total Reward: {average_reward}: Actor loss: {actor_loss:.2f}, Critic loss: {critic_loss:.2f}, Total loss: {total_loss:.2f}')
            writer.add_scalar("evaluate/average_reward", average_reward, global_step)

            # #player 1
            win_rate, draw_rate, loss_rate = evaluate_agent(agent_policy=agent_policy, opponent_policy=opponent_policy, agent_starts=True, num_episodes=10)
            writer.add_scalar("evaluate/player1_win_rate", win_rate, global_step)
            writer.add_scalar("evaluate/player1_draw_rate", draw_rate, global_step)
            writer.add_scalar("evaluate/player1_loss_rate", loss_rate, global_step)
            # #player 2
            win_rate, draw_rate, loss_rate = evaluate_agent(agent_policy=agent_policy, opponent_policy=opponent_policy, agent_starts=False, num_episodes=10)
            writer.add_scalar("evaluate/player2_win_rate", win_rate, global_step)
            writer.add_scalar("evaluate/player2_draw_rate", draw_rate, global_step)
            writer.add_scalar("evaluate/player2_loss_rate", loss_rate, global_step)
            
            logger.info(f'========')
            torch.save(agent_policy.actor_critic_network.state_dict(), 'agent_policy_network.pth')
            