import numpy as np
import random 
import time
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import logging
from collections import defaultdict
from combine_video import combine_videos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pickle
from gymnasium import RewardWrapper

class CustomRewardFrozenLake(RewardWrapper):
    def __init__(self, env):
        super(CustomRewardFrozenLake, self).__init__(env)
    
    def reward(self, reward):
        """Modify the reward function."""
        # Access the current state (position in the grid)
        current_state = self.env.unwrapped.s
        # Access the last action taken
        last_action = self.env.unwrapped.lastaction

        # Custom reward logic:
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
        """Choose an action `a` in the current world state (s)."""
        explor_exploit_tradeoff = random.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < epsilon:
            action = action_space.sample()
        # Exploitation (taking the biggest Q-value for this state)
        else:
            action = np.argmax(Q[state, :])
        return action

        
class Q_Learning():
    def __init__(self, learning_rate, gamma, epsilon, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.explorer = EpsilonGreedy()
        self.reset_Q()
    
    def reset_Q(self):
        self.Q = np.zeros((self.state_size, self.action_size))
        
    def update_Q(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * delta

    def train(self, env, episodes, epsilon_decay=0.995, epsilon_min=0.2):
        total_rewards = []

        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.explorer.choose_action(env.action_space, state, self.Q, self.epsilon)
                next_state, reward, done, _, _ = env.step(action)

                self.update_Q(state, action, reward, next_state)

                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)

            self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)
            logger.info(f'Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}')

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
    env = gym.make("FrozenLake-v1", is_slippery=False)
    # env = gym.make("FrozenLake-v1", render_mode = 'rgb_array', is_slippery=False)
    # env = RecordVideo(env, video_folder="frozenlake-agent-qlearning", name_prefix="eval",
    #               episode_trigger=lambda episode_id: episode_id % 10 == 0)
    env = CustomRewardFrozenLake(env)
    reward_over_episode = 0
    total_episodes = 100
    learning_rate = 0.5
    gamma = 0.999
    epsilon = .3
    action_size = env.action_space.n
    state_size = env.observation_space.n
    print(f"Action size: {action_size}")
    print(f"State size: {state_size}")

    q_learning_agent = Q_Learning(learning_rate=learning_rate, gamma=gamma, epsilon=epsilon, state_size=state_size, action_size=action_size)

    # Train the agent
    total_rewards = q_learning_agent.train(env, episodes=total_episodes)
    print(f'Average Reward: {np.mean(total_rewards)}')
    q_learning_agent.save_q_table(filename="frozen_lake_q_table.pkl")
    import json
    with open('episode_rewards.json', 'w', encoding='utf-8') as f:
        json.dump(total_rewards, f, ensure_ascii=False, indent=4)

    # combine_videos('frozenlake-agent-qlearning', 'frozenlake-agent-qlearning.mp4')
    # Load and test the agent
    # q_learning_agent.load_q_table("frozen_lake_q_table.pkl")
    # q_learning_agent.test(env, episodes=1)
