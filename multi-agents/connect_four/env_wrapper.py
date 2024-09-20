import numpy as np
import torch
import gym
from gym import spaces
from pettingzoo.classic import connect_four_v3

class SingleAgentSelfPlayEnv(gym.Env):
    def __init__(self, agent_starts=True):
        super(SingleAgentSelfPlayEnv, self).__init__()

        # Initialize the PettingZoo environment
        self.env = connect_four_v3.env()
        self.env.reset()

        # Define the observation and action spaces based on the first agent
        self.observation_space = np.prod(self.env.observation_space(self.env.possible_agents[0])['observation'].shape)
        self.action_space = self.env.action_space(self.env.possible_agents[0]).n
        self.action_mask = self.env.observation_space(self.env.possible_agents[0])['action_mask']

        # Current agent's turn
        self.agent_starts = agent_starts
        self.current_agent = self.env.agent_selection
        self.done = False

        # Placeholder for the shared policy
        self.agent_policy = None
        self.opponent_policy = None
      
    def set_policies(self, agent_policy, opponent_policy):
        self.agent_policy = agent_policy
        self.opponent_policy = opponent_policy

    def reset(self):
        self.env.reset()
        self.current_agent = self.env.agent_selection
        self.done = False

        if not self.agent_starts:
          opponent_env_obs = self.env.observe(self.current_agent)
          opponent_obs = self.flatten_observation(opponent_env_obs)

          opponent_masked_action = opponent_env_obs['action_mask']

          opponent_action = self.select_opponent_action(opponent_obs, opponent_masked_action)
          self.env.step(opponent_action)
          self.current_agent = self.env.agent_selection

        obs = self.env.observe(self.current_agent)
        return obs

    def step(self, action):
        # Apply the action for the current agent (agent's turn)
        self.env.step(action)
        self.done = self.env.terminations[self.current_agent] or self.env.truncations[self.current_agent]
        reward = self.env.rewards[self.current_agent]
        info = self.env.infos[self.current_agent]

        if not self.done:
            # Switch to the next agent (opponent)
            self.current_agent = self.env.agent_selection

            # Get observation for the opponent
            opponent_env_obs = self.env.observe(self.current_agent)

            opponent_obs = self.flatten_observation(opponent_env_obs)
            opponent_masked_action = opponent_env_obs['action_mask']
            # print(f'{opponent_obs.shape=}')
            # print(f'{opponent_masked_action.shape=}')

            # Select action for the opponent using the same policy
            opponent_action = self.select_opponent_action(opponent_obs, opponent_masked_action)
            # print(f'{opponent_action=}')

            # Apply the opponent's action
            self.env.step(opponent_action)

            # Check if the game is over after the opponent's move
            self.done = self.env.terminations[self.current_agent] or self.env.truncations[self.current_agent]
            # Update the reward: subtract the opponent's reward (zero-sum game assumption)
            reward -= self.env.rewards[self.current_agent]
            info = self.env.infos[self.current_agent]

            # Switch back to the original agent
            self.current_agent = self.env.agent_selection

        # Get the next observation for the agent
        obs = self.env.observe(self.current_agent)
        return obs, reward, self.done, info

    def select_opponent_action(self, obs, masked_action):
        # Use the same policy for the opponent
        # Ensure no gradients are computed for the opponent's action
        with torch.no_grad():
            action, _, _, _ = self.opponent_policy.choose_action(obs, masked_action)
        return action.item()

    def flatten_observation(self, obs):
        return obs['observation'].reshape(-1)

    def render(self, mode='human'):
        self.env.render()
