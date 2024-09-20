import torch.nn as nn

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_dim, fc2_dim):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

        self.pi = nn.Linear(fc2_dim, output_dim)
        self.v = nn.Linear(fc2_dim, 1)

        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        # dist = torch.distributions.Categorical(self.softmax(self.pi(x)))
        pi = self.softmax(self.pi(x))
        value = self.v(x)

        return pi, value
