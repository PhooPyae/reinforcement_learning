from torch.optim as optim

from network import PolicyNetwork
from config import Config as config
import sys

class Agent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        
        self.action_log_probs = []
        self.rewards = []
        self.expected_retun = []
        
        self.gamma = config.gamma
        self.device = config.device

        self.policy_network = PolicyNetwork(self.state_space, self.action_space).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr = config.learning_rate)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs = self.policy_network(state)

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        log_action_prob = action_dist.log_prob(action)
        self.action_log_probs.append(log_action_prob)

        return action.item()
    
    def learn(self):
        self.optimizer.zero_grad()
        G = []
        
        loss = 0
        G_t = 0

        for reward in self.rewards[::-1]:
            G_t = reward + self.gamma * G_t
            G.insert(0, G_t)

        G = torch.tensor(G)
        G = (G - G.mean()) / (G.std() + 1e-9)  # Normalize the return

        for g, log_prob in zip(G, self.action_log_probs):
            loss += - (g * log_prob)

        loss.backward()
        self.optimizer.step()
        
        self.action_log_probs = []
        self.rewards = []
        
        return loss.item()
        