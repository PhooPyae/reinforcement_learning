import torch
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states = torch.from_numpy(np.vstack(
            [e[0] for e in sample if e is not None]
        )).float()
        actions = torch.from_numpy(np.vstack(
            [e[1] for e in sample if e is not None]
        )).int()
        rewards = torch.from_numpy(np.vstack(
            [e[2] for e in sample if e is not None]
        )).float()
        next_states = torch.from_numpy(np.vstack(
            [e[3] for e in sample if e is not None]
        )).float()
        dones = torch.from_numpy(np.vstack(
            [e[4] for e in sample if e is not None]
        )).float()

        return states, actions, rewards, next_states, dones