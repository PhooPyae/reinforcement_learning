import torch

class Config:
    n_games = 10000
    epsilon = 0.8
    epsilon_decay = 0.01
    min_epsilon = 0.2
    gamma = 0.99
    max_step = 100
    learning_rate = 3e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = 40

    capacity = 512
    batch_size = 64