import sys
import matplotlib.pyplot as plt
sys.path.append('../../reinforcement_learning/')
import json
import pprint
import numpy as np
import random
from DQN import DQN
import torch
from replay_memory import ReplayBuffer

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# def get_TD_target(state, next_state, action, reward, gamma, Q):
#     target = reward + (gamma * max(Q(next_state)) - Q(state)[action])
#     logger.debug(f'target {target}')
#     return target
class TicTacToe():
    def __init__(self):
        self.QL = DQN(0.4, 1e-7, 0.4)
        self.state = '_' * 9
        self.batch_size = 200
        self.rm = ReplayBuffer(200, (18, ), 9)
        self.losses = []
        self.episodes = 1000
        
    def prepare_state(self, state, current_player):
        nn_state = np.zeros((2,9))
        logger.debug(f'state {state}')
        for i, char in enumerate(state):
            logger.debug(f'{i=} {char=}')
            if char == '_':
                continue
            if char == current_player:
                nn_state[0][i] = 1 
            else:
                nn_state[1][i] = 1 
        logger.debug(f'{nn_state=}')
        return torch.flatten(torch.tensor(nn_state, dtype=torch.float32))
    
    def check_win(self, state):
        # Define win conditions based on board positions
        win_conditions = [
            (0, 1, 2),  # Top row
            (3, 4, 5),  # Middle row
            (6, 7, 8),  # Bottom row
            (0, 3, 6),  # Left column
            (1, 4, 7),  # Middle column
            (2, 5, 8),  # Right column
            (0, 4, 8),  # Left-to-right diagonal
            (2, 4, 6)   # Right-to-left diagonal
        ]

        # Check if any winning condition is satisfied
        for condition in win_conditions:
            if state[condition[0]] == state[condition[1]] == state[condition[2]] != '_':
                return True
        return False

    def check_draw(self, state):
        return '_' not in state  # Return True if no empty spaces left, hence a draw
    
    def _get_reward(self, state):
        if self.check_draw(state):
            return 0, True
        if self.check_win(state):
            return 10, True
        return -1, False
    
    def _get_valid_actions(self, state):
        valid_actions = [i for i, x in enumerate(state) if x == '_']
        logger.debug(f'valid actions {valid_actions} {state=}')
        return valid_actions
    
    def _epsilon_greedy_policy(self, encoded_state, valid_actions):
        if random.uniform(0, 1) < self.QL.epsilon:
            action = random.choice(valid_actions)  # Exploration
            return action

        current_q_values = self.QL.Q(encoded_state)
        logger.debug(f'{type(current_q_values)}')
        np_q_values = current_q_values.detach().numpy()
        logger.debug(f'{np_q_values}')
        for (i, x) in enumerate(np_q_values):
            if i not in valid_actions:
                np_q_values[i] = -10000
        action = np.argmax(np_q_values)
        # logger.debug(f'Exploitation!')
        return action

    def get_current_player(self, state):
        if state.count('X') > state.count('O'):
            return 'O'  # O's turn since X has already played more times
        else:
            return 'X'  # X's turn
        
    def train(self):
        for episode in range(self.episodes):
            logger.debug(f'{episode=}')
            x_state_history, o_state_history = [], []
            state = self.state
            # self.QL.Q['terminal'] = np.zeros(9)
            
            action = None
            done = False
            while not done:
                current_player = self.get_current_player(state)
                logger.debug(f'====== {current_player=} Player Turn ======')
                encoded_state = self.prepare_state(state, current_player)
                
                valid_actions = self._get_valid_actions(state)
                action = self._epsilon_greedy_policy(encoded_state, valid_actions)
                if state[action] != '_':
                    raise 'Invalid Action!'
                
                next_state = state[:action] + current_player + state[action + 1:]
                reward, done = self._get_reward(next_state)
                if current_player == 'X':
                    x_state_history.append([state, action, reward])
                else:
                    o_state_history.append([state, action, reward])
                
                if done and reward == 10:
                    if current_player == 'X':
                        o_state_history[-1][-1] = -5
                    else:
                        x_state_history[-1][-1] = -5
                
                logger.debug(f'{x_state_history=} {o_state_history=}') 
            
                state = next_state
                # self.learn(episode)

            x_score = 0
            for i in range(len(x_state_history) - 1):
                state = self.prepare_state(x_state_history[i][0],'X')
                action = x_state_history[i][1]
                reward = x_state_history[i][2]
                next_state = self.prepare_state(x_state_history[i+1][0], 'X')

                self.rm.store_transition(state, action, reward, next_state, False)
                x_score += reward
            # logger.info(f'{x_score=}')
                
                
                # self.QL.update_Q(state= state, next_state=next_state, action=action, reward=reward)
            
            # self.QL.update_Q(state= x_state_history[-1][0], next_state='terminal', action=x_state_history[-1][1], reward=x_state_history[-1][2])
            last_state = self.prepare_state(x_state_history[-1][0], 'X')
            self.rm.store_transition(
                last_state, 
                x_state_history[-1][1], 
                x_state_history[-1][2], 
                np.ones(18),
                True
            )
            o_score = 0
            for i in range(len(o_state_history) - 1):
                state = self.prepare_state(o_state_history[i][0],'O')
                action = o_state_history[i][1]
                reward = o_state_history[i][2]
                next_state = self.prepare_state(o_state_history[i+1][0], 'O')

                self.rm.store_transition(state, action, reward, next_state, False)
                o_score += reward
                
            # logger.info(f'{o_score=}')
                # self.QL.update_Q(state= state, next_state=next_state, action=action, reward=reward)
            
            # self.QL.update_Q(state= x_state_history[-1][0], next_state='terminal', action=x_state_history[-1][1], reward=x_state_history[-1][2])
            last_state = self.prepare_state(o_state_history[-1][0], 'O')
            self.rm.store_transition(
                last_state, 
                o_state_history[-1][1], 
                o_state_history[-1][2], 
                np.ones(18),
                True
            )

            self.learn(episode)
            logger.info(f'Training Episode {episode} {self.losses[-1] if len(self.losses) else None}')
            
        # with open('store_transition.json', 'w', encoding='utf-8') as f:
        #     json.dump({key: value.tolist() for key, value in self.rm.store_transition.items()}, f, ensure_ascii=False, indent=4)
        torch.save(self.QL.Q, 'DQN_model.pt')
        self.plot()
        
    def learn(self, episode):
        if  self.rm.mem_cntr < self.batch_size:
            return 

        self.QL.Q.optimizer.zero_grad()
        states, actions, rewards, states_, dones = self.rm.sample_buffer(self.batch_size)
        # logger.info(f'{states=}')
        indicies = np.arange(self.batch_size)
        logger.debug(f'{states.shape=}')

        Q_pred = self.QL.Q.forward(states)[indicies, actions]
        if episode % 30 == 0:
            self.QL.Q_target.load_state_dict(self.QL.Q.state_dict())

        Q_next = self.QL.Q_target.forward(states_).max(dim=1)[0]

        Q_next[dones] = 0.0
        Q_target = rewards + self.QL.discount_factor * Q_next
        
        loss = self.QL.Q.loss(Q_target, Q_pred)
        # logger.info(f'Training Loss {loss}')
        self.losses.append(loss.detach().numpy())
        loss.backward()
        self.QL.Q.optimizer.step()
    
    def plot(self):
        # Plotting the training loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Training Loss')
        plt.title('Training Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')

if __name__ == '__main__':
    tictactoe = TicTacToe()             
    tictactoe.train()