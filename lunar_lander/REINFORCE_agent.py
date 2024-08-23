import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state, 128)
        self.fc2 = nn.Linear(128, action_space)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        action = self.softmax(x)
        return action
        
class PolicyGradientAgent:
    def __init__(self, obs_space, action_space, device):
        self.policy_network = PolicyNetwork(obs_space, action_space).to(device)
        self.gamma = 0.99
        self.learning_rate = 1e-6
        self.optimizer = optim.AdamW(self.policy_network.parameters(), lr=self.learning_rate)
        self.reward_memory = []
        self.action_memory = []
        self.device = device
        
    def choose_action(self, state):
        action_probs = self.policy_network(state.to(self.device))
        action_probs = torch.distributions.Categorical(action_probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()
    
    def store_reward(self, reward):
        self.reward_memory.append(reward)
        
    def learn(self):
        self.optimizer.zero_grad()

        running_g = 0
        gs = []
        for R in self.reward_memory[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)
            
        G = torch.tensor(gs).to(self.device)
        loss = 0
        for g, log_prob in zip(G, self.action_memory):
            loss += -(g * log_prob)
        loss.backward()
        self.optimizer.step()
        
        self.action_memory = []
        self.reward_memory = []
        