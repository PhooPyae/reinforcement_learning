import torch

def save_checkpoint(model, checkpoint):
    torch.save(model.state_dict(), checkpoint)

def load_checkpoint(model, checkpoint, device):
    model.load_state_dict(torch.load(checkpoint, map_location=device))

def evaluate(env, agent):

    for i in range(5):
        state, _ = env.reset()
        done = False
        step = 0
        cumulative_reward = 0

        while not done and step < 1000:
            action, _, _ = agent.choose_action(state)
            next_state, reward, done, info, _ = env.step(action.item())
            state = next_state
            cumulative_reward += reward
            step += 1
            
    return cumulative_reward/5