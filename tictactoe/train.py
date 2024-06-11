import sys
sys.path.append('../../reinforcement_learning/')
import random
import json

from demo.tic_tac_toe import TicTacToe
from algorithms.Q_learning import Q_Learning

env = TicTacToe()
rl = Q_Learning(epsilon=0.5, learning_rate=0.1, discount_factor=0.9, Q={})

done = False

for episode in range(1):
    env.reset()
    done = False

    while not done:
        available_actions = env.get_available_actions()
        action = random.choice(available_actions)
        state = env.state
        next_state, reward, done = env.step(action)
        rl.update_Q(state=state, next_state=next_state, action=action, reward=reward)
        print(f'{state=}, {action=}, {next_state=}, {reward=}, {done=}')
        # env.print_board()

with open('tictactoe_l.json', 'w', encoding='utf-8') as f:
    json.dump({key: value.tolist() for key, value in rl.Q.items()}, f, ensure_ascii=False, indent=4)


        