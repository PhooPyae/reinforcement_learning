import numpy as np
import random 
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Q_Learning():
    def __init__(self, epsilon, learning_rate, discount_factor, Q):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = Q
        
    def update_Q(self, state, next_state, action, reward):
        if not state in self.Q:
            self.Q[state] = np.zeros(9)

        if not next_state in self.Q:
            self.Q[next_state] = np.zeros(9)

        self.Q[state][action] += self.learning_rate \
            * (reward + self.discount_factor * np.max(self.Q[next_state]) - self.Q[state][action])
        # logger.info(f'{self.Q=}')
        