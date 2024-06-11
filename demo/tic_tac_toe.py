import json
import pprint
import numpy as np
import random
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicTacToe():
    def __init__(self):
        self.state = '_' * 9

    def reset(self):
        self.state = '_' * 9
    
    def check_win(self):
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
            if self.state[condition[0]] == self.state[condition[1]] == self.state[condition[2]] != '_':
                return True
        return False

    def check_draw(self):
        return '_' not in self.state  # Return True if no empty spaces left, hence a draw
    
    def _get_reward(self):
        if self.check_win():
            return 20
        if self.check_draw():
            return -1
        return -1

    def step(self, action):
        if action < 0 or action > 8:
            raise ValueError('Invalid action')

        logger.debug(f'{action=}')

        current_player = self.get_current_player()
        next_state = self.state[:action] + current_player + self.state[action + 1:]
        logger.debug(f'{next_state=}')

        self.state = next_state
        reward = self._get_reward()
        logger.debug(f'{reward=}')

        done = self.check_win() or self.check_draw()
        logger.debug(f'{done=}')

        return next_state, reward, done
        
    def get_current_player(self):
        if self.state.count('X') > self.state.count('O'):
            return 'O'  # O's turn since X has already played more times
        else:
            return 'X'  # X's turn

    def get_available_actions(self):
        valid_actions = [i for i, x in enumerate(self.state) if x == '_']
        return valid_actions

    def print_board(self):
        # Check if the board string is valid
        if len(self.state) != 9:
            print("Invalid board length. Board must be a string of length 9.")
            return

        # Print the board in a 3x3 format
        print("Board layout:")
        for i in range(0, 9, 3):
            print(f"{self.state[i]} | {self.state[i+1]} | {self.state[i+2]}")
            if i < 6:
                print("---------")  # Print separating lines between rows
