import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, output_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x  = self.softmax(self.fc3(x))
        dist = torch.distributions.Categorical(x)
        
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        value = self.activation(self.fc2(x))

        return value
