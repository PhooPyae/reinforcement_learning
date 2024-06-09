'''
Simple Grid World Problem
Initial state. The goal is to reach 'G'. Agent is at (0,0). X are obstacles.
    A - - - - 
    - X X - - 
    - - X - - 
    - - X - - 
    - - - - G 
    
# Parameters
learning_rate = 0.1  # Typical values range between 0 and 1
discount_factor = 0.9  # Discount factor for future rewards
epsilon = 0.1  # Exploration rate, ensuring some actions are chosen at random

# Initialize Q-table
Initialize Q(s, a) = 0 for all s in States, a in Actions
Q(Terminal_state, _) = 0  # No further rewards after terminal state

# Q-Learning Algorithm
For each episode:
    Initialize S to the starting state
    While S is not Terminal:
        If random() < epsilon:
            Choose A randomly from available actions in state S  # Explore
        Else:
            Choose A = argmax_a Q(S, a)  # Exploit: select the action with max Q-value in state S
        Take action A, observe reward R and new state S_next
        Q(S, A) = Q(S, A) + learning_rate * (Reward + discount_factor * max_a Q(S_next, a) - Q(S, A))
        S = S_next
    EndWhile
EndFor
'''
import numpy as np
import random 
import time

class Q_Learning():
    def __init__(self, n, epsilon, learning_rate, discount_factor, ):
        self.n = n
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.num_of_episode = 1000
        Q, action_dict, action_map, goal_position, obstacles = self.initialize(self.n)
        self.run(self.n, goal_position, obstacles, self.epsilon, Q, action_dict, action_map)
        
    def initialize(self, n):
        '''
        Initialize Q table
        '''
        Q = np.zeros((n*n, 4))
        
        # Action encoding: up = 0, down = 1, left = 2, right = 3
        action_dict = { 0: -n, 1: n, 2: -1, 3: 1}
        action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        goal_position = n * n - 1
        obstacles = {6, 7, 12, 17}
        return Q, action_dict, action_map, goal_position, obstacles
    
    def print_grid(self, n, state, goal_position, obstacles):
        for i in range(n):
            for j in range(n):
                pos = i * n + j
                if pos == state:
                    print('A', end=' ') #A for agent
                elif pos == goal_position:
                    print('G', end=' ') #G for goal
                elif pos in obstacles:
                    print('X', end=' ') #X for obstacles
                else:
                    print('-', end=' ') #- for empty space
            print()
        print() #Extra newline for better separation
        
        
    def get_reward(self, state, goal_position, obstacles):
        if state == goal_position:
            return 100
        elif state in obstacles:
            return -10
        else:
            return -1
    
    def get_optimal_policy(self, Q, state):
        return np.argmax(Q[state])
    
    def run(self, n, goal_position, obstacles, epsilon, Q, action_dict, action_map):
        start_time = time.time() 
        for episode in range(self.num_of_episode):
            state = 0 # top left corner
            steps = 0 # state taken in an episode
            
            while state != goal_position:
                print(f"========= Episode {episode + 1}, Step {steps + 1} =========")
                self.print_grid(n, state, goal_position, obstacles)
                
                if random.uniform(0, 1) > epsilon:
                    print(f'Exploration')
                    action = random.choice([0,1,2,3]) #explore
                    print(f'Chosen Action {action_map[action]}')
                else:
                    print(f'Exploitation Q[{state}] {Q[state]}')
                    action = np.argmax(Q[state])
                    print(f'Chosen Action {action_map[action]}')
            
                # Take action and observe new state and reward
                next_state = state + action_dict[action]
                print(f'state {state} next state {next_state}')
                next_state = max(0, min(n*n-1, next_state))
                reward = self.get_reward(next_state, goal_position, obstacles)
                print(f'reward {reward}')
                
                #Q-Learning update
                old_value = Q[state, action]
                Q[state, action] += self.learning_rate * (reward + self.discount_factor * np.max(Q[next_state]) - Q[state,action])
                print(f"Q-Value Updated: Q[{state}, {action}] from {old_value:.2f} to {Q[state, action]:.2f}")

                # Move to the next state
                state = next_state
                steps += 1
        
                print(f'Updated: Q-table: \n {Q}')
            
            end_time = time.time()  # Record the end time of the episode
            duration = end_time - start_time  # Calculate the duration of the episode
            print(f"Completed in {duration:.2f} seconds.")
            print("Final state of this episode:")
            self.print_grid(n, state, goal_position, obstacles)
            
            
    
if __name__ == '__main__':
    grid_size = 5
    epsilon = 0.1
    learning_rate = 0.1
    discount_factor = 0.9
    
    Qlearning = Q_Learning(n=grid_size, epsilon=epsilon, learning_rate=learning_rate, discount_factor=discount_factor)
        
        

