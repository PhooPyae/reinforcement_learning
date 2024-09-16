import torch

class Config:
    n_games = 1000
    max_steps = 5000
    
    tau = 0.001
    gamma = 0.999
    
    alpha = 1e-4
    beta = 1e-3

    max_size = 1_000_000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    