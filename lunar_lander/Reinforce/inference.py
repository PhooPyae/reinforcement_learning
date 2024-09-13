import gymnasium as gym
import torch

from agent import Agent

env = gym.make("LunarLander-v2",  render_mode="rgb_array")
# env = RecordVideo(env, video_folder="lunarlander-agent-reinforce", name_prefix="eval",
#                   episode_trigger=lambda x: True)

agent = Agent(env.observation_space.shape[0], env.action_space.n)
agent.policy_network.load_state_dict(torch.load('policy_network.pth', map_location=torch.device('cpu')))

state, _ = env.reset()
done = False
step = 0
cumulative_reward = 0

while not done and step < 400:
  action = agent.choose_action(state)
  next_state, reward, done, info, _ = env.step(action)
  state = next_state
  cumulative_reward += reward
  step += 1
  env.render()
  
print(cumulative_reward/step)