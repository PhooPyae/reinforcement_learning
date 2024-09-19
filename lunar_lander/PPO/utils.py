import torch

def save_checkpoint(model, checkpoint):
    torch.save(model.state_dict(), checkpoint)

def load_checkpoint(model, checkpoint, device):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    