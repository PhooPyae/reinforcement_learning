from tic_tac_toe import TicTacToe
import json 
import numpy as np

game = TicTacToe()

done = False
q_table = json.load(open('../tictactoe/tttoe_v3.json'))

def _get_valid_actions(state):
    valid_actions = [i for i, x in enumerate(state) if x == '_']
    # logger.debug(f'valid actions {valid_actions} {self.tttoe.state=}')
    return valid_actions
    
def get_action(state):
    # Get valid actions for the current state
    valid_actions = _get_valid_actions(state)
    
    if not valid_actions:
        return None  # No valid actions available

    # Fetch Q-values for all actions in the current state
    current_q_values = q_table[state].copy()
    for (i, x) in enumerate(current_q_values):
        if i not in valid_actions:
            current_q_values[i] = -np.inf

    print(f'{current_q_values=}')
    # c = [x for i, x in enumerate(current_q_values) if i in valid_actions else -10000]
    action = np.argmax(current_q_values)
    print(f'{action=}')
    return action
    
    # # Find the action with the maximum Q-value among the valid actions
    # best_action = max(valid_actions, key=lambda x: current_q_values[x])
    
    # return best_action

# def get_action(state):
#     valid_actions = _get_valid_actions(state)
#     return np.argmax(q_table[state])
    
while not done:
    game.print_board()
    com_action = get_action(game.state)
    print(f'{com_action=}')
    next_state, _, done = game.step(com_action)
    game.state = next_state
    game.next_state = next_state
    print(f'{next_state=}')
    game.print_board()
    if done:
        game.print_board()
        break
    action = int(input("Your Turn: "))
    next_state, _, done = game.step(action)
    game.state = next_state
    game.next_state = next_state
    game.print_board()