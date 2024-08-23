import gymnasium as gym
import torch
from REINFORCE_agent import PolicyGradientAgent
from gymnasium.wrappers import RecordVideo
from combine_video import combine_videos
import json

env = gym.make("LunarLander-v2",  render_mode="rgb_array")
env = RecordVideo(env, video_folder="lunarlander-agent-reinforce", name_prefix="eval",
                  episode_trigger=lambda x: True)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n
device = torch.device('cpu')

episodes = 100
output = []
agent = PolicyGradientAgent(state_space, action_space, device)
for episode in range(3):
    obs, info = env.reset()
    
    score = 0
    done = False
    while not done:
        state = torch.tensor(obs, dtype=torch.float32).to(device)
        action = agent.choose_action(state)
        obs_, reward, done, info, _ = env.step(action)
        
        agent.store_reward(reward)
        
        score += reward
    message = f'episode {episode}, total reward {score:.1f}'
    output.append(message)

    agent.learn()
    
with open('result_reinforce.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
env.close()
combine_videos('lunarlander-agent-reinforce', 'combined_video.mp4')
