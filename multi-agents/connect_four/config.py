import torch

config = {
    "num_episodes": 1_000_000,
    "max_steps_per_episode": 1000,
    "opponent_update_interval": 100,
    "gamma": 0.99,
    "n_epochs": 2,

    "alpha": 1e-4,
    "beta": 1e-3,
    "lambda": 0.95,
    "policy_clip": 0.2,
    "batch_size": 5,
    "fc1_dim" : 256,
    "fc2_dim": 256,
    
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 40
}