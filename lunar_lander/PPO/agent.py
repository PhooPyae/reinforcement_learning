import torch
import torch.optim as optim
from networks import ActorNetwork, CriticNetwork
import config as Config

config = Config.config

class Agent:
    def __init__(self, state_space, action_space):
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.device = config["device"]
        self.fc1_dim = config["fc1_dim"]
        self.fc2_dim = config["fc2_dim"]
        
        self.actor = ActorNetwork(state_space, action_space, self.fc1_dim, self.fc2_dim).to(self.device)
        self.critic = CriticNetwork(state_space, action_space, self.fc1_dim, self.fc2_dim).to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.beta)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        prob_dist = self.actor(state)
        action = prob_dist.sample()

        value = self.critic(state)
        
        probs = prob_dist.log_prob(action)
        action = action.item()
        
        return action, probs, value