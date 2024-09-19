import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from networks import Actor, Critic
from config import Config as config
from noise import OUActionNoise
from replay import ReplayBuffer

class Agent:
    def __init__(self, state_space, action_space):
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha #actor learning rate
        self.beta = config.beta #critic learning rate
        self.max_size = config.max_size
        self.batch_size = config.batch_size
        self.device = config.device

        self.replay_buffer = ReplayBuffer(config.max_size, state_space, action_space)

        self.noise = OUActionNoise(mu=np.zeros(action_space))

        self.actor_network = Actor(state_space, action_space).to(self.device)
        self.critic_network = Critic(state_space, action_space).to(self.device)

        self.actor_target_network = Actor(state_space, action_space).to(self.device)
        self.critic_target_network = Critic(state_space, action_space).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.alpha)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.beta)

        self.noise_value = None


    def choose_action(self, state):
        self.actor_network.eval()

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        noise = torch.tensor(self.noise(), dtype=torch.float32).to(self.device)
        mu = self.actor_network(state)
        mu_prime = mu + noise
        self.noise_value = noise.cpu().detach().numpy()

        self.actor_network.train()

        return mu_prime.cpu().detach().numpy()

    def learn(self):
        if self.replay_buffer.mem_cntr < self.batch_size:
            return None, None

        states, actions, rewards, states_, dones = self.replay_buffer.sample_buffer(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        # y = r + gamma * Q_prime(s_, mu_prime(s_,))
        mu_ = self.actor_target_network(states_)
        critic_value_ = self.critic_target_network(states_, mu_)

        #if done, 0
        critic_value_[dones] = 0.0
        critic_value_ = critic_value_.view(-1)

        critic_value = self.critic_network(states, actions)

        target = rewards + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        critic_loss = F.mse_loss(critic_value, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        mu = self.actor_network(states)
        actor_loss = - self.critic_network(states, mu)
        actor_loss = torch.mean(actor_loss)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_network()

        return critic_loss.item(), actor_loss.item()


    def update_target_network(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor_network.named_parameters()
        critic_params = self.critic_network.named_parameters()
        target_actor_params = self.actor_target_network.named_parameters()
        target_critic_params = self.critic_target_network.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.critic_target_network.load_state_dict(critic_state_dict)
        self.actor_target_network.load_state_dict(actor_state_dict)