'''
Simple Grid World Problem
Initial state. The goal is to reach 'G'. Agent is at (0,0). X are obstacles. Actions [up, down, left, right]
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
    Initialize A from state S
    
    While S is not Terminal:
        If random() < epsilon:
            Choose A_next randomly from available actions in state S  # Explore
        Else:
            Choose A_next = argmax_a Q(S_next)  # Exploit: select the A' from S'
        Take action A, observe reward R and new state S_next
        Q(S, A) = Q(S, A) + learning_rate * (Reward + discount_factor * Q(S_next, A_next) - Q(S, A))
        S = S_next
        A = A_next
    EndWhile
EndFor
'''
import numpy as np
import random 
import time

class SARSA():
    def __init__(self, n, epsilon, learning_rate, discount_factor, ):
        self.n = n
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.num_of_episode = 1000
        
        # Action encoding: up = 0, down = 1, left = 2, right = 3
        self.action_dict = { 0: -n, 1: n, 2: -1, 3: 1}
        self.action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.obstacles = {6, 7, 12, 17}        
        self.goal_position = n * n - 1
        
        self.Q = np.zeros((self.n*self.n, 4))

        # Q, action_dict, action_map, goal_position, obstacles = self.initialize(self.n)
        self.run()
        
    def print_grid(self, state):
        for i in range(self.n):
            for j in range(self.n):
                pos = i * self.n + j
                if pos == state:
                    print('A', end=' ') #A for agent
                elif pos == self.goal_position:
                    print('G', end=' ') #G for goal
                elif pos in self.obstacles:
                    print('X', end=' ') #X for obstacles
                else:
                    print('-', end=' ') #- for empty space
            print()
        print() #Extra newline for better separation
        
        
    def get_reward(self, state):
        if state == self.goal_position:
            return 1
        return -1
    
    def epsilon_greedy_policy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([0, 1, 2, 3])  # Exploration
            print(f'Exploration: Chosen Action {self.action_map[action]}')
        else:
            action = np.argmax(self.Q[state])  # Exploitation
            print(f'Exploitation of Q[{state}]: Chosen Action {self.action_map[action]}')
        return action
    
    def print_grid_with_actions(self, n, obstacles, goal_position, action_map, max_indices):
        grid = [['-' for _ in range(n)] for _ in range(n)]
        
        # Set obstacles and goal in the grid
        for obs in obstacles:
            grid[obs // n][obs % n] = 'X'
        grid[goal_position // n][goal_position % n] = 'G'
        
        # Set actions in the grid
        for i in range(n * n):
            if i not in obstacles and i != goal_position:
                grid[i // n][i % n] = action_map[max_indices[i]][0].upper()  # Use the first letter of the action
        
        # Print the grid
        for row in grid:
            print(' '.join(row))
        print()
        
    def run(self):
        start_time = time.time() 
        for episode in range(self.num_of_episode):
            steps = 0
            state = 0  # Start at the top-left corner
            action = self.epsilon_greedy_policy(state)
            
            while state != self.goal_position:
                print(f"========= Episode {episode + 1}, Step {steps + 1} =========")
                
                # Take action and observe new state and reward
                if state % 5 == 0:
                    if action == 2:
                        next_state = state
                    else:
                        next_state = state + self.action_dict[action]
                else:
                    next_state = state + self.action_dict[action]
                
                if next_state in self.obstacles or next_state < 0 or next_state >= self.n*self.n:
                    next_state = state  # Stay in the current state if the move is illegal
                
                next_action = self.epsilon_greedy_policy(next_state)
                reward = self.get_reward(next_state)
                
                #Q-Learning update
                old_value = self.Q[state, action]
                # SARSA Update
                self.Q[state, action] += self.learning_rate * (
                    reward + self.discount_factor * self.Q[next_state, next_action] - self.Q[state, action])
                
                state = next_state
                action = next_action
                steps += 1

                self.print_grid(state)
                # print(f'Updated: Q-table: \n {self.Q}')
                
            max_indices = np.argmax(self.Q, axis=1)

            end_time = time.time()  # Record the end time of the episode
            duration = end_time - start_time  # Calculate the duration of the episode
            print(f"Completed in {duration:.2f} seconds.")
            print("Final state of this episode:")
            self.print_grid(state)
            self.print_grid_with_actions(self.n, self.obstacles, self.goal_position, self.action_map, max_indices)
            
            
    
if __name__ == '__main__':
    grid_size = 5
    epsilon = 0.1
    learning_rate = 0.1
    discount_factor = 0.9
    
    Qlearning = SARSA(n=grid_size, epsilon=epsilon, learning_rate=learning_rate, discount_factor=discount_factor)
        
        

