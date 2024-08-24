import numpy as np
import random 
import time
import gymnasium as gym
import logging
from collections import defaultdict
from gymnasium.wrappers import RecordVideo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pickle
from gymnasium import RewardWrapper

class CustomRewardFrozenLake(RewardWrapper):
    def __init__(self, env):
        super(CustomRewardFrozenLake, self).__init__(env)
    
    def reward(self, reward):
        current_state = self.env.unwrapped.s
        last_action = self.env.unwrapped.lastaction

        if current_state == 15:  # Assuming state 15 is the goal state in a 4x4 grid
            reward = 10  # Big reward for reaching the goal
        elif current_state in [5, 7, 11, 12]:  # Assuming these are hole states in a 4x4 grid
            reward = -5  # Big penalty for falling into a hole
        else:
            reward = -0.1  # Small penalty for each step to encourage faster reaching of the goal
        
        return reward

class EpsilonGreedy:
    def __init__(self):
        pass

    def choose_action(self, action_space, state, Q, epsilon):
        explor_exploit_tradeoff = random.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < epsilon:
            action = action_space.sample()
        # Exploitation (taking the biggest Q-value for this state)
        else:
            action = np.argmax(Q[state, :])
        return action

        
class MonteCarlo():
    def __init__(self, gamma, epsilon, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.explorer = EpsilonGreedy()
        self.reset_Q()
    
    def reset_Q(self):
        self.Q = np.zeros((self.state_size, self.action_size))
        self.returns_sum = np.zeros((self.state_size, self.action_size))
        self.returns_count = np.zeros((self.state_size, self.action_size))
        
    def train(self, env, episodes, epsilon_decay=0.995, epsilon_min=0.2):
        total_rewards = []

        for episode in range(episodes):
            state, _ = env.reset()
            episode_states_actions_rewards = []
            done = False
            total_reward = 0

            while not done:
                action = self.explorer.choose_action(env.action_space, state, self.Q, self.epsilon)
                next_state, reward, done, _, _ = env.step(action)
                episode_states_actions_rewards.append((state, action, reward))
                state = next_state
                total_reward += reward

            # Process episode and update Q-values using Monte Carlo method
            G = 0
            for state, action, reward in reversed(episode_states_actions_rewards):
                G = self.gamma * G + reward
                if (state, action) not in [(x[0], x[1]) for x in episode_states_actions_rewards[:-1]]:
                    self.returns_sum[state, action] += G
                    self.returns_count[state, action] += 1
                    self.Q[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]

            total_rewards.append(total_reward)
            self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)
            logger.info(f'Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}')

        return total_rewards

    def test(self, env, episodes):
        total_rewards = []

        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                env.render()
                action = self.explorer.choose_action(env.action_space, state, self.Q, epsilon=0.0)
                next_state, reward, done, _, _ = env.step(action)

                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)
            logger.info(f'Test Episode: {episode + 1}, Total Reward: {total_reward}')

        env.close()
        return total_rewards
    
    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.Q, f)
        print(f"Q-table saved to {filename}")
        
    def load_q_table(self, filename="q_table.pkl"):
        with open(filename, 'rb') as f:
            self.Q = pickle.load(f)
        print(f"Q-table loaded from {filename}")
        return self.Q
    
if __name__ == '__main__':
    # env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = CustomRewardFrozenLake(env)
    total_episodes = 100
    gamma = 0.999
    epsilon = 0.3
    action_size = env.action_space.n
    state_size = env.observation_space.n
    print(f"Action size: {action_size}")
    print(f"State size: {state_size}")

    monte_carlo_agent = MonteCarlo(gamma=gamma, epsilon=epsilon, state_size=state_size, action_size=action_size)

    # Train the agent
    monte_carlo_agent.train(env, episodes=total_episodes)
    monte_carlo_agent.save_q_table(filename="frozen_lake_mc_q_table.pkl")
    # Load and test the agent
    # monte_carlo_agent.load_q_table("frozen_lake_mc_q_table.pkl")
    # monte_carlo_agent.test(env, episodes=1)
