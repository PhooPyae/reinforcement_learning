import torch

class Config:
    n_games = 10000
    gamma = 0.99
    
    max_game_step = 1000
    
    learning_rate = 3e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = 40