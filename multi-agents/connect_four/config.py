import torch

config = {
    "num_envs": 1_000_000,
    "gamma": 0.99,
    "n_epochs": 2,

    "alpha": 1e-4,
    "beta": 1e-3,
    "lambda": 0.95,
    "policy_clip": 0.2,
    "batch_size": 5,
    "fc1_dim" : 256,
    "fc2_dim": 256,

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 40
}
