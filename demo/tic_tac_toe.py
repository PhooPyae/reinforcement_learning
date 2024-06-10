import numpy as np
import random
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicTacToe():
    def __init__(self):
        self.state = '_'*9
        return None
    
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
                # return board_str[condition[0]]  # Return 'X' or 'O' for winner
                return True
        return False  # No winner found

    def check_draw(self):
        return '_' not in self.state  # Return True if no empty spaces left, hence a draw

    def step(self, action):
        logger.info(f'action {action}')
        current_player = self.get_current_player()
        self.state = self.state[:action] + current_player + self.state[action + 1:]
        is_terminal = self.check_win() or self.check_draw()
        logger.info(f'{self.state=}')
        logger.info(f'{is_terminal=}')
            
        return self.state, is_terminal
        
    def get_current_player(self):
        if self.state.count('X') > self.state.count('O'):
            return 'O'  # O's turn since X has already played more times
        else:
            return 'X'  # X's turn

    def game_status(self, board_str):
        winner = self.check_win(board_str)
        if winner:
            return f"Winner: {winner}"  # Announce winner
        elif self.check_draw(board_str):
            return "Draw"  # Declare a draw if board is full
        else:
            return "Continue"  # Game should continue
    
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

class QLearning():
    def __init__(self):
        self.tttoe = TicTacToe()
        self.epsilon = 0.1
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.episodes = 1000
        
        self.Q = {}
        self.actions = range(0,9)
        
    def _get_reward(self, state):
        if self.tttoe.check_win(state):
            return 1
        if self.tttoe.check_draw(state):
            return -0.1
        return -1
    
    def _get_valid_actions(self, state):
        valid_actions = [i for i, x in enumerate(state) if x == '_']
        logger.info(f'valid actions {valid_actions}')
        return valid_actions

    def _epsilon_greedy_policy(self, state, valid_actions):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(valid_actions)  # Exploration
            logger.info(f'Exploration!')
        else:
            action = np.argmax(self.Q[state]) if state in self.Q else valid_actions[0] # Exploitation
            logger.info(f'Exploitation!')
        return action
    
    def run(self):
        for episode in range(self.episodes):
            is_terminal = False 
            while not is_terminal:
                valid_actions = self._get_valid_actions(self.tttoe.state)
                action = self._epsilon_greedy_policy(self.tttoe.state, valid_actions)
                next_state, is_terminal = self.tttoe.step(action)
                self.tttoe.print_board()
            break
        return
            
            
        
    
if __name__ == '__main__':
    # tictactoe = TicTacToe()
    # # Example board configuration as a string
    # board_str = "X_O______"

    # # Check the game status
    # logger.info(tictactoe.game_status(board_str))
    q_learning = QLearning()
    q_learning.run()

            
        