import numpy as np
import torch
import pprint

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.chosen_indicies = []

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        # pprint.pprint(f'{state=}')
        # pprint.pprint(f'{action=}')
        # pprint.pprint(f'{reward=}')
        # pprint.pprint(f'{state_=}')
        # pprint.pprint(f'{done=}')
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        # print(f'{batch=}')

        self.chosen_indicies.extend(batch)

        states = torch.tensor(self.state_memory[batch])
        actions = torch.tensor(self.action_memory[batch])
        rewards = torch.tensor(self.reward_memory[batch])
        states_ = torch.tensor(self.new_state_memory[batch])
        terminal = torch.tensor(self.terminal_memory[batch])

        return states, actions, rewards, states_, terminal