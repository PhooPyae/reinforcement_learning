import torch

config = {
    "n_games": 1000,
    "gamma": 0.99,
    "max_steps": 5000,
    "tau": 0.001,
    "alpha": 1e-4,
    "beta": 1e-3,
    "max_size": 1_000_000,
    "batch_size": 64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 40
}
