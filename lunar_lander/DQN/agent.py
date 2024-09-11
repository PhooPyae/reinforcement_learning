from network import DeepQNetwork
from memory import ReplayBuffer
from config import Config as config

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Agent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = config.epsilon
        
        self.q_network = DeepQNetwork(self.state_space, self.action_space).to(config.device)
        self.q_target_network = DeepQNetwork(self.state_space, self.action_space).to(config.device)
        
        self.q_target_network.load_state_dict(self.q_network.state_dict())
        
        self.memory = ReplayBuffer(config.capacity)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(config.device)

        if np.random.rand() > config.epsilon:
            q_value = self.q_network(state)
            return np.argmax(q_value.cpu().data.numpy())
        
        return np.random.choice(self.action_space)

    def learn(self):
        self.epsilon = max(config.min_epsilon, self.epsilon * config.epsilon_decay)
        
        total_loss = 0
        n_iters = int(config.capacity/config.batch_size)
        
        for _ in range(n_iters):
            states , actions, rewards, next_states, dones = self.memory.sample(config.batch_size)
            states = states.to(config.device)
            actions = torch.tensor(actions.to(config.device), dtype=torch.int64)
            next_states = next_states.to(config.device)
            rewards = rewards.to(config.device)
            dones = dones.to(config.device)

            q_next_values = self.q_target_network(next_states)
            max_q_next_values, _ = torch.max(q_next_values, axis = 1)
            max_q_next_values = max_q_next_values.unsqueeze(1)

            q_target = rewards + config.gamma * max_q_next_values * (1 - dones)

            q_values = self.q_network(states)
            q_value_a = q_values.gather(1, actions)
            
            loss = nn.SmoothL1Loss()(q_value_a, q_target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return (total_loss/n_iters)