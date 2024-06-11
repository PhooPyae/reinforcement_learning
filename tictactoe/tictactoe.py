import sys
sys.path.append('../../reinforcement_learning/')
import json
import pprint
import numpy as np
import random
from algorithms.Q_learning import Q_Learning
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TicTacToe():
    def __init__(self):
        self.QL = Q_Learning(0.6, 0.1, 0.9, {})
        self.state = '_' * 9
    
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
            return -1, True
        if self.check_win(state):
            return 1, True
        return -0.1, False
    
    def _get_valid_actions(self, state):
        valid_actions = [i for i, x in enumerate(state) if x == '_']
        logger.debug(f'valid actions {valid_actions} {state=}')
        return valid_actions
    
    def _epsilon_greedy_policy(self, Q, state, valid_actions):
        if random.uniform(0, 1) < self.QL.epsilon:
            action = random.choice(valid_actions)  # Exploration
            # logger.debug(f'Exploration!')
        else:
            if not state in Q:
                return random.choice(valid_actions)

            current_q_values = Q[state].copy()
            for (i, x) in enumerate(current_q_values):
                if i not in valid_actions:
                    current_q_values[i] = -10000
            action = np.argmax(current_q_values)
            # logger.debug(f'Exploitation!')
        return action
    
    def get_current_player(self, state):
        if state.count('X') > state.count('O'):
            return 'O'  # O's turn since X has already played more times
        else:
            return 'X'  # X's turn
        
    def train(self):
        for episode in range(10000):
            logger.debug(f'{episode=}')
            x_state_history, o_state_history = [], []
            state = self.state
            self.QL.Q['terminal'] = np.zeros(9)
            
            action = None
            done = False
            while not done:
                current_player = self.get_current_player(state)
                logger.debug(f'====== {current_player=} Player Turn ======')
                
                valid_actions = self._get_valid_actions(state)
                action = self._epsilon_greedy_policy(self.QL.Q, state, valid_actions)
                if state[action] != '_':
                    raise 'Invalid Action!'
                
                next_state = state[:action] + current_player + state[action + 1:]
                
                reward, done = self._get_reward(next_state)
                if current_player == 'X':
                    x_state_history.append([state, action, reward])
                else:
                    o_state_history.append([state, action, reward])
                
                if done and reward == 1:
                    if current_player == 'X':
                        o_state_history[-1][-1] = -2
                    else:
                        x_state_history[-1][-1] = -2
                
                logger.debug(f'{x_state_history=} {o_state_history=}') 
            
                state = next_state

            for i in range(len(x_state_history) - 1):
                state = x_state_history[i][0]
                action = x_state_history[i][1]
                reward = x_state_history[i][2]
                next_state = x_state_history[i+1][0]
                self.QL.update_Q(state= state, next_state=next_state, action=action, reward=reward)
            
            self.QL.update_Q(state= x_state_history[-1][0], next_state='terminal', action=x_state_history[-1][1], reward=x_state_history[-1][2])

            for i in range(len(o_state_history) - 1):
                state = o_state_history[i][0]
                action = o_state_history[i][1]
                reward = o_state_history[i][2]
                next_state = o_state_history[i+1][0]
                self.QL.update_Q(state= state, next_state=next_state, action=action, reward=reward)
            
            self.QL.update_Q(state= o_state_history[-1][0], next_state='terminal', action=o_state_history[-1][1], reward=o_state_history[-1][2])

        with open('tttoe_v3.json', 'w', encoding='utf-8') as f:
            json.dump({key: value.tolist() for key, value in self.QL.Q.items()}, f, ensure_ascii=False, indent=4)
            
if __name__ == '__main__':
    tictactoe = TicTacToe()             
    tictactoe.train()