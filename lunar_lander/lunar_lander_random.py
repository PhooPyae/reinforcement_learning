import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import logging
from combine_video import combine_videos
import json

logging.basicConfig(level=logging.WARNING)

if __name__ == '__main__':
    env = gym.make("LunarLander-v2",  render_mode="rgb_array")
    env = RecordVideo(env, video_folder="lunarlander-agent-random", name_prefix="eval",
                  episode_trigger=lambda x: True)

    episodes = 100
    output = []
    for episode in range(episodes):
        obs, info = env.reset()
        score = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, info, _ = env.step(action)
            score += reward
        message = f'episode {episode}, total reward {score:.1f}'
        output.append(message)
        
    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    env.close()
    combine_videos('lunarlander-agent-random', 'combined_video.mp4')
