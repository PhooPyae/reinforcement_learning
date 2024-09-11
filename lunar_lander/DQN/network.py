import torch
import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x
    