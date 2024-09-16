import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.mu = nn.Linear(300, output_dim)
        
        self.bn1 = nn.LayerNorm(self.fc1)
        self.bn2 = nn.LayerNorm(self.fc2)
        
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        self.init_weight()
        

    def init_weight(self):
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform(-f1, f1)
        self.fc1.bias.data.uniform(-f1, f1)
        
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform(-f2, f2)
        self.fc2.bias.data.uniform(-f2, f2)

        f3 = 3e-3
        self.fc3.weight.data.uniform(-f3, f3)
        self.fc3.bias.data.uniform(-f3, f3)
        
        
    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.tanh(self.fc3(x))

        return x

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.q = nn.Linear(300, 1)
        self.action_value = nn.Linear(action_dim, 300)
        
        self.bn1 = nn.LayerNorm(self.fc1)
        self.bn2 = nn.LayerNorm(self.fc2)
        
        self.activation = nn.ReLU()

        self.init_weight()
        

    def init_weight(self):
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform(-f1, f1)
        self.fc1.bias.data.uniform(-f1, f1)
        
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform(-f2, f2)
        self.fc2.bias.data.uniform(-f2, f2)

        f3 = 3e-3
        self.fc3.weight.data.uniform(-f3, f3)
        self.fc3.bias.data.uniform(-f3, f3)
         
        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform(-f4, f4)
        self.action_value.bias.data.uniform(-f4, f4)

    def forward(self, state, action):
        state_value = self.activation(self.bn1(self.fc1(state)))
        state_value = self.bn2(self.fc2(state_value))

        action_value = self.action_value(action)
        state_action_value = torch.tensor.add(state_value, action_value)
        state_action_value = self.action_value(state_action_value)

        state_action_value = self.q(state_action_value)

        return state_action_value