import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from networks import ActorCriticNetwork
from memory import Memory
import config as Config

config = Config.config

class Agent:
    def __init__(self, state_space, action_space):
        self.alpha = config["alpha"]
        self.n_epochs = config["n_epochs"]
        self.gamma = config["gamma"]
        self.gae_lambda = config["lambda"]
        self.policy_clip = config["policy_clip"]

        self.device = config["device"]
        
        self.fc1_dim = config["fc1_dim"]
        self.fc2_dim = config["fc2_dim"]
        
        self.actor_critic_network = ActorCriticNetwork(state_space, action_space, self.fc1_dim, self.fc2_dim).to(self.device)
        self.optim = optim.Adam(self.actor_critic_network.parameters(), lr=self.alpha)

        self.memory = Memory(config["batch_size"])

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        prob_dist, value = self.actor_critic_network(state)
        action = prob_dist.sample()

        log_probs = prob_dist.log_prob(action)
        entropy = prob_dist.entropy()
        
        return action.cpu().detach().numpy(), log_probs.cpu().detach().numpy(), entropy.cpu().detach().numpy(), value.cpu().detach().numpy()
    
    def learn(self):
        for _ in range(self.n_epochs):
            states_arr, actions_arr, probs_arr, values_arr, rewards_arr, dones_arr, batches = self.memory.generate_batches()

            #advantage = gamma * lambda * delta 
            # delta = reward + gamma * v(s+1) - v(s)
            values = values_arr

            advantage = np.zeros(len(rewards_arr), dtype=np.float32)

            for t in range(len(rewards_arr) - 1):
                discount = 1
                A_t = 0
                for k in range(t, len(rewards_arr) - 1):
                    delta = rewards_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k]
                    A_t += discount * delta
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = A_t
            advantage = torch.tensor(advantage).to(self.device)

            values = torch.tensor(values, dtype=torch.float32).to(self.device)

            for batch in batches:
                states = torch.tensor(states_arr[batch], dtype=torch.float32).to(self.device)
                actions = torch.tensor(actions_arr[batch]).to(self.device)
                old_probs = torch.tensor(probs_arr[batch], dtype=torch.float32).to(self.device)
                old_values = torch.tensor(values_arr[batch], dtype=torch.float32).to(self.device)

                ## actor_loss = min(ratio * advantage, clip(ratio, 1-policy_clip, 1+policy_clip) * advantage)
                dist, critic_value = self.actor_critic_network(states)
                #   critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp() # r(theta)/r(theta_old)
                # print(f'{prob_ratio=}')

                weighted_probs = advantage[batch] * prob_ratio
                clipped = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                weighted_clipped_probs = clipped * advantage[batch]

                actor_loss = - torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.optim.zero_grad()

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic_network.parameters(), config["max_grad_norm"])
                self.optim.step()

        self.memory.clear_memory()
          
        return actor_loss, critic_loss, total_loss