import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)

        self.actor = nn.Linear(512, output_dim)
        self.critic = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.activation(self.dropout(self.fc1(x)))
        x = self.activation(self.dropout(self.fc2(x)))
        actor_output = self.softmax(self.actor(x))
        critic_output = self.critic(x)

        return (actor_output, critic_output)
    