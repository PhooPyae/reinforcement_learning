import torch
import numpy as np
from pettingzoo.classic import connect_four_v3

from agent import Agent

env = connect_four_v3.env(render_mode="human")
env.reset(seed=42)
state_dim = env.observation_space(env.possible_agents[0])
state_dim = np.prod(state_dim['observation'].shape)
action_dim = env.action_space(env.possible_agents[0]).n

print(f'{state_dim=}')
print(f'{action_dim=}')

ppo_policy = Agent(state_space=state_dim, action_space=action_dim)
print(ppo_policy.actor_critic_network)
ppo_policy.actor_critic_network.load_state_dict(torch.load('agent_policy_network.pth', map_location='cpu'))
ppo_policy.actor_critic_network.eval()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
        print(f'{agent} {reward}')
    else:
        state = observation['observation'].reshape(-1)
        mask = observation["action_mask"]
        # this is where you would insert your policy
        # action = env.action_space(agent).sample(mask)
        action, _, _, _ = ppo_policy.choose_action(state, mask)

    env.step(action)
env.close()
    