import torch

def evaluate(env, agent):
  with torch.no_grad():

    for i in range(3):
        state, _ = env.reset()
        done = False
        step = 0
        cumulative_reward = 0

        while not done and step < 1000:
            action = agent.choose_action(state)
            next_state, reward, done, info, _ = env.step(action)
            state = next_state
            cumulative_reward += reward
            step += 1

    return cumulative_reward/3
