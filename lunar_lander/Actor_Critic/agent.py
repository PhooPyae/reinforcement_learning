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
    
    def learn(self):
        return None