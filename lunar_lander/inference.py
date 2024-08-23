import numpy as np
import gymnasium as gym
import logging
from collections import defaultdict
from Q_learning import Q_Learning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pickle

def test(agent, env, episodes):
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = agent.discretize_state(state)
        total_reward = 0
        done = False

        while not done:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = agent.discretize_state(next_state)

            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        logger.info(f'Test Episode: {episode + 1}, Total Reward: {total_reward}')

    env.close()
    return total_rewards
    
def load_q_table(filename="q_table.pkl"):
    with open(filename, 'rb') as f:
        Q_dict = pickle.load(f)
    Q = defaultdict(lambda: np.zeros(4), Q_dict)
    print(f"Q-table loaded from {filename}")
    return Q
    
if __name__ == '__main__':
    episodes = 1
    env = gym.make("LunarLander-v2", render_mode = 'human')
    agent = Q_Learning(epsilon=0.0, learning_rate=0.1, discount_factor=0.99)

    agent.Q = load_q_table("lunar_lander_q_table.pkl")
    test(agent, env, episodes=episodes)