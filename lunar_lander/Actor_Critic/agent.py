import torch
import torch.nn as nn
import torch.optim as optim

from network import ActorCriticNetwork
from config import Config as config

class Agent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.device = config.device
        self.gamma = config.gamma
        
        self.actor_critic_network = ActorCriticNetwork(self.state_space, self.action_space)
        self.optimizer = optim.Adam(self.actor_critic_network.parameters(), lr=config.learning_rate)
        
        self.log_prob = None

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs, _ = self.actor_critic_network(state)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        self.log_prob = action_dist.log_prob(action)
        
        return action.item()
    
    def learn(self, state, next_state, reward):
        self.optimizer.zero_grad()

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

        _, v_state = self.actor_critic_network(state)
        _, v_next_state = self.actor_critic_network(next_state)

        delta = reward + self.gamma * v_next_state - v_state

        actor_loss = - delta * self.log_prob
        critic_loss = delta ** 2

        loss = actor_loss + critic_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()