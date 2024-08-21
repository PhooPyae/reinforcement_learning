import numpy as np
import random 
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions, lr):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer5 = nn.Linear(hidden_size, num_actions)
        self.loss = nn.HuberLoss()
        self.dropout = nn.Dropout(0.5)
        # self.normalize = nn.BatchNorm(num_features=hidden_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=3e-3)
    
    def forward(self, x):
        # x = torch.flatten(torch.tensor(x, dtype=torch.float32))
        # logger.info(f'{x.shape}')
        out = self.layer1(torch.tensor(x, dtype=torch.float32))
        # out = self.normalize(out)
        out = self.relu(out)
        out = self.layer2(out)
        # # out = self.normalize(out)
        out = self.relu(out)
        # out = self.layer3(out)
        # out = self.relu(out)
        # out = self.layer4(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        out = self.layer5(out)
        # print(f'{out=}')
        return out

class DQN():
    def __init__(self, epsilon, learning_rate, discount_factor):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = MLP(18, 64, 9, learning_rate)
        self.Q_target = MLP(18, 64, 9, learning_rate)
        logger.info(f'{self.Q}')