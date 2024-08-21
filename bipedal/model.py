import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state, hidden_state, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state, hidden_state)
        self.fc2 = nn.Linear(hidden_state, action_space) # output means
        self.fc3 = nn.Linear(hidden_state, action_space) #output log std deviation
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        mean = self.fc2(x)
        log_std = self.fc3(x)
        std = torch.exp(log_std)
        return mean, std
        
 