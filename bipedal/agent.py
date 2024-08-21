import torch
from torch.distributions import Categorical
import torch.optim as optim
from model import PolicyNetwork
from torch.distributions import Normal

class REINFORCE:
    def __init__(self, obs_space_dims, hidden_state, action_space_dims, device):
        
        self.learning_rate = 1e-4 
        self.gamma = 0.99
        self.eps = 1e-6

        self.probs = []
        self.rewards = []
        self.loss_history = []
        self.reward_history = []

        self.device = device

        self.net = PolicyNetwork(obs_space_dims, hidden_state, action_space_dims).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
    
    def choose_action(self, state):
        action_means, action_stddevs = self.net(state)
        normal_dist = Normal(action_means + self.eps, action_stddevs + self.eps)
        action = normal_dist.sample()
        action = torch.tanh(action)
        prob = normal_dist.log_prob(action)
        
        self.probs.append(prob)
        return action
    
    def update(self):
        cumulative_rewards = 0
        discount_rewards = []
        
        for R in self.rewards[::-1]:
            cumulative_rewards = R + self.gamma * cumulative_rewards
            discount_rewards.insert(0, cumulative_rewards)
        
        discount_rewards = torch.tensor(discount_rewards).to(self.device)
#         discount_rewards = (discount_rewards - discount_rewards.mean()) / (discount_rewards.std() + 1e-9)  # Normalize   
        
        loss = 0
        for log_prob, discount_reward in zip(self.probs, discount_rewards):
            loss += log_prob.mean() * discount_reward * (-1)
            
        #update policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.reward_history.append(sum(self.rewards))
        self.loss_history.append(loss.item())
        
        self.probs = []
        self.rewards = []