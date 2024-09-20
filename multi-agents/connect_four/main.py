import torch
import copy

from env_wrapper import SingleAgentSelfPlayEnv
from agent import Agent
from utils import evaluate_agent
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter()
    env_test = SingleAgentSelfPlayEnv(agent_starts=True)
    agent_policy = Agent(state_space=env_test.observation_space, action_space=env_test.action_space)
    opponent_policy = copy.deepcopy(agent_policy)
    env_test.set_policies(agent_policy=agent_policy, opponent_policy=opponent_policy)

    num_episodes = 1000
    max_steps_per_episode = 100
    global_step = 0
    opponent_update_interval = 100  # Update every 10 episodes

    for episode in range(num_episodes):
        obs = env_test.reset()
        done = False
        total_reward = 0
        step = 0
        while not done and step < max_steps_per_episode:
            state = obs['observation'].reshape(-1)
            action_mask = obs['action_mask']

            action, prob, entropy, value = agent_policy.choose_action(state, action_mask)
            # print(f'action {action}')
            next_obs, reward, done, info = env_test.step(action)

            agent_policy.memory.store_memory(state, action, prob,  value.item(), reward, done)

            obs = next_obs
            total_reward += reward
            step += 1
            global_step += 1

        actor_loss, critic_loss, total_loss = agent_policy.learn()
        writer.add_scalar('Loss/policy_loss', actor_loss, global_step)
        writer.add_scalar('Loss/value_loss', critic_loss, global_step)
        writer.add_scalar('Reward/episode_reward', total_reward, episode)
        writer.add_scalar('Loss/total_loss', total_loss, global_step)
        writer.add_scalar('Loss/entropy', entropy, global_step)
        writer.add_scalar('Loss/value', value, global_step)
        writer.add_scalar('Loss/prob', prob, global_step)


        
        if episode % opponent_update_interval == 0:
            opponent_policy = copy.deepcopy(agent_policy)
            env_test.set_policies(agent_policy=agent_policy, opponent_policy=opponent_policy)
            print(f'Episode {episode}: Total Reward: {total_reward}: Actor loss: {actor_loss:.2f}, Critic loss: {critic_loss:.2f}, Total loss: {total_loss:.2f}')

            # #player 1
            evaluate_agent(agent_policy=agent_policy, opponent_policy=opponent_policy, agent_starts=True, num_episodes=10)
            # #player 2
            evaluate_agent(agent_policy=agent_policy, opponent_policy=opponent_policy, agent_starts=False, num_episodes=10)

            print(f'========')
            torch.save(agent_policy.actor_critic_network.state_dict(), 'agent_policy_network.pth')
            