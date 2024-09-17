import gymnasium as gym
import numpy as np

from config import Config as config
from evaluate import evaluate
from agent import Agent


env = gym.make('LunarLanderContinuous-v2')
print(env.observation_space.shape, env.action_space.shape)

agent = Agent(env.observation_space.shape, env.action_space.shape[0])
train_step = 0
actor_losses = []
critic_losses = []
scores = []

for i in range(config.n_games):
  state, _ = env.reset()
  # print(state.shape)
  done = False
  step = 0
  score = 0

  agent.noise.reset()

  while not done and step < 10000:
      action = agent.choose_action(state)
      state_, reward, done, _, _ = env.step(action)
      agent.replay_buffer.store_transition(state, action, reward, state_, done)

      state = state_
      score += reward

      critic_loss, actor_loss = agent.learn()

      step += 1
      train_step += 1

  scores.append(score)
  critic_losses.append(critic_loss)
  actor_losses.append(actor_loss)

  avg_score = np.mean(scores[-100:])
  avg_critic_loss = np.mean(critic_losses[-100:])
  avg_actor_loss = np.mean(actor_losses[-100:])

  eval_reward = evaluate(env, agent)
#   wandb.log({"episode": i, "reward": avg_score, "avg_actor_loss": avg_actor_loss, "avg_critic_loss": avg_critic_loss, 
#              "eval_reward": eval_reward})
  
#   wandb.log({
#     "OU_noise_component_1": agent.noise_value[0],
#     "OU_noise_component_2": agent.noise_value[1]
# })
            
  if (i+1) % 5 == 0:
    print(f'Episode {i+1}, Actor Loss : {avg_actor_loss:.4f}, Critic Loss : {avg_critic_loss:.4f} Reward : {avg_score:.4f}')