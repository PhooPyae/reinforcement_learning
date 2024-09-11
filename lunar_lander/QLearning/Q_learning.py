import numpy as np
import random 
import time
import gymnasium as gym
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pickle

# Assume `Q` is your Q-table (a dictionary or defaultdict)

class Q_Learning():
    def __init__(self, epsilon, learning_rate, discount_factor, Q=None):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = Q if Q is not None else defaultdict(lambda: np.zeros(4))  # Initialize Q-table for 4 possible actions
    
    def discretize_state(self, state):
        bins = [
            np.linspace(-1, 1, 10),  # X position
            np.linspace(-1, 1, 10),  # Y position
            np.linspace(-1, 1, 10),  # X velocity
            np.linspace(-1, 1, 10),  # Y velocity
            np.linspace(-1, 1, 10),  # Angle
            np.linspace(-1, 1, 10),  # Angular velocity
            np.array([0, 1]),        # Left leg contact
            np.array([0, 1])         # Right leg contact
        ]
        
        indices = tuple(np.digitize(s, b) for s, b in zip(state, bins))
        return indices

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(4))  # Explore: choose a random action
        else:
            return np.argmax(self.Q[state])  # Exploit: choose the best known action

    def update_Q(self, state, next_state, action, reward):
        # Ensure state and next_state are in Q-table
        if state not in self.Q:
            self.Q[state] = np.zeros(4)

        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(4)

        # Update Q-value using the Q-learning formula
        self.Q[state][action] += self.learning_rate * \
            (reward + self.discount_factor * np.max(self.Q[next_state]) - self.Q[state][action])

    def train(self, env, episodes, epsilon_decay=0.995, epsilon_min=0.2):
        total_rewards = []

        for episode in range(episodes):
            state, _ = env.reset()
            state = self.discretize_state(state)
            total_reward = 0
            done = False

            while not done:
                # env.render()
                action = self.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                next_state = self.discretize_state(next_state)

                self.update_Q(state, next_state, action, reward)

                state = next_state
                total_reward += reward

            # Decay epsilon
            self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)

            total_rewards.append(total_reward)

            logger.info(f'Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}')

        return total_rewards

    def test(self, env, episodes):
        total_rewards = []

        for episode in range(episodes):
            state, _ = env.reset()
            state = self.discretize_state(state)
            total_reward = 0
            done = False

            while not done:
                env.render()
                action = self.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                next_state = self.discretize_state(next_state)

                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)
            logger.info(f'Test Episode: {episode + 1}, Total Reward: {total_reward}')

        env.close()
        return total_rewards

    def save_q_table(self, filename="q_table.pkl"):
        Q_dict = dict(self.Q)
        with open(filename, 'wb') as f:
            pickle.dump(Q_dict, f)
        print(f"Q-table saved to {filename}")
        
    def load_q_table(self, filename="q_table.pkl"):
        with open(filename, 'rb') as f:
            Q_dict = pickle.load(f)
        Q = defaultdict(lambda: np.zeros(4), Q_dict)
        print(f"Q-table loaded from {filename}")
        return Q
    
if __name__ == '__main__':
    # env = gym.make("LunarLander-v2")
    env = gym.make("LunarLander-v2", render_mode = 'human')
    q_learning_agent = Q_Learning(epsilon=0.0, learning_rate=0.1, discount_factor=0.99)

    # Train the agent
    # q_learning_agent.train(env, episodes=5000)
    # q_learning_agent.save_q_table(filename="lunar_lander_q_table.pkl")
    # Test the agent
    q_learning_agent.Q = q_learning_agent.load_q_table("lunar_lander_q_table.pkl")
    q_learning_agent.test(env, episodes=1)