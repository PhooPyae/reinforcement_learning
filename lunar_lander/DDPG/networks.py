import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_dim = 400, fc2_dim = 300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(*input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.mu = nn.Linear(fc2_dim, output_dim)

        self.bn1 = nn.LayerNorm(fc1_dim)
        self.bn2 = nn.LayerNorm(fc2_dim)

        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

        self.init_weight()


    def init_weight(self):
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 3e-3
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)


    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.tanh(self.mu(x))

        return x

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, fc1_dim = 400, fc2_dim = 300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(*input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

        self.q = nn.Linear(fc2_dim, 1)

        self.action_value = nn.Linear(action_dim, fc2_dim)

        self.bn1 = nn.LayerNorm(fc1_dim)
        self.bn2 = nn.LayerNorm(fc2_dim)

        self.activation = nn.ReLU()

        self.init_weight()


    def init_weight(self):
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 3e-3
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

    def forward(self, state, action):
        state_value = self.activation(self.bn1(self.fc1(state)))
        state_value = self.bn2(self.fc2(state_value))

        action_value = self.action_value(action)

        state_action_value = self.activation(torch.add(state_value, action_value))

        state_action_value = self.q(state_action_value)

        return state_action_value