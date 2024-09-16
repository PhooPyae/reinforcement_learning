import gym
from agent import Agent

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(env.observation_space.shape, env.action_space.shape[0])

    