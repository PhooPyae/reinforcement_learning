import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from memory import Memory
import config as Config
from network import ActorCriticNetwork

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

        self.value_coef = config["value_coef"]  # Coefficient for value function loss
        self.entropy_coef = config["entropy_coef"]  # Coefficient for entropy bonus
        self.max_grad_norm = config["max_grad_norm"]  # Maximum gradient norm for clipping

    def choose_action(self, state, action_mask):

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        # print(f'{state.shape=}')
        # print(f'{action_mask.shape=}')
        # print(f'{state=}')
        # print(f'{action_mask=}')

        action_probs, value = self.actor_critic_network(state)

        masked_probs = action_probs * action_mask
        masked_probs = masked_probs / masked_probs.sum()

        prob_dist = torch.distributions.Categorical(masked_probs)

        action = prob_dist.sample()

        log_probs = prob_dist.log_prob(action)
        entropy = prob_dist.entropy()

        return action.cpu().detach().numpy(), log_probs.cpu().detach().numpy(), entropy.cpu().detach().numpy(), value.cpu().detach().numpy()

    def learn(self):
        for _ in range(self.n_epochs):
            states_arr, actions_arr, probs_arr, values_arr, rewards_arr, dones_arr, batches = self.memory.generate_batches()

            #advantage = gamma * lambda * delta
            # delta = reward + gamma * v(s+1) - v(s)

            advantages = np.zeros(len(rewards_arr), dtype=np.float32)
            last_gae_lam = 0

            for t in reversed(range(len(rewards_arr))):
              if t == len(rewards_arr) - 1:
                next_non_terminal = 1.0 - dones_arr[t]
                next_values = 0
              else:
                next_non_terminal = 1.0 - dones_arr[t+1]
                next_values = values_arr[t+1]

              delta = rewards_arr[t] + self.gamma * next_values * next_non_terminal - values_arr[t]
              last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
              advantages[t] = last_gae_lam

            returns = advantages + values_arr
            # print(f'{advantages=}')
            # print(f'{values_arr=}')
            # print(f'{returns=}')

            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            # print(f'{len(values_arr)=}')
            # print(f'{returns.shape=}')

            #normalized advantage
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)


            for batch in batches:
                # print(batch)
                states = torch.tensor(states_arr[batch], dtype=torch.float32).to(self.device)
                actions = torch.tensor(actions_arr[batch]).to(self.device)
                old_probs = torch.tensor(probs_arr[batch], dtype=torch.float32).to(self.device)
                old_values = torch.tensor(values_arr[batch], dtype=torch.float32).to(self.device)

                ## actor_loss = min(ratio * advantage, clip(ratio, 1-policy_clip, 1+policy_clip) * advantage)

                action_probs, critic_value = self.actor_critic_network(states)
                dist = torch.distributions.Categorical(action_probs)

                critic_value = torch.squeeze(critic_value)
                # print(f'{critic_value.shape=}')
                # print(f'{returns[batch].shape=}')

                new_probs = dist.log_prob(actions)
                ratio = (new_probs - old_probs).exp() # r(theta)/r(theta_old)
                # print(f'{prob_ratio=}')

                sur1 = advantages[batch] * ratio
                sur2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages[batch]

                actor_loss = - torch.min(sur1, sur2).mean()

                critic_loss = 0.5 * ((returns[batch] - critic_value) ** 2).mean()

                #entropy bonus
                entropy = dist.entropy().mean()

                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optim.zero_grad()

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic_network.parameters(), self.max_grad_norm)
                self.optim.step()


        self.memory.clear_memory()

        return actor_loss.item(), critic_loss.item(), total_loss.item()