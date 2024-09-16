import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from networks import Actor, Critic
from config import Config as config
from noise import OUActionNoise
from replay import ReplayBuffer

class Agent:
    def __init__(self, state_space, action_space):
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha #actor learning rate
        self.beta = config.beta #critic learning rate
        self.max_size = config.max_size
        self.device = config.device

        self.replay_buffer = ReplayBuffer(config.max_size, state_space, action_space)

        self.noise = OUActionNoise(mu=np.zeros(action_space))
        
        self.actor_network = Actor(state_space, action_space)
        self.critic_network = Critic(state_space, action_space)

        self.actor_target_network = Actor(state_space, action_space)
        self.critic_target_network = Critic(state_space, action_space)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.beta)
    
    def choose_action(self, state):
        self.actor.eval()
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        noise = torch.tensor(self.noise, dtype=torch.float32).to(self.device)
        mu = self.actor_network(state)
        mu_prime = mu + noise
        
        self.actor.train()
        return mu_prime
    
    
    