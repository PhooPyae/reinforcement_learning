import numpy as np

# Define the states and their available actions
states_actions = {
    'high': ['search', 'wait'],
    'low': ['search', 'wait', 'recharge']
}

# Define the transitions and rewards
transitions = {
    ('high', 'search'): {'high': (0.5, 1), 'low': (0.5, 1)},
    ('high', 'wait'): {'high': (1.0, 0.2)},
    ('low', 'search'): {'low': (0.7, -3), 'high': (0.3, -3)},
    ('low', 'wait'): {'low': (1.0, 0.1)},
    ('low', 'recharge'): {'high': (1.0, 0)}
}

def simulate_action(state, action):
    outcomes = transitions[(state, action)]
    next_states = list(outcomes.keys())
    probabilities, rewards = zip(*[outcomes[ns] for ns in next_states])
    next_state = np.random.choice(next_states, p=probabilities)
    reward = dict(outcomes)[next_state][1]
    return next_state, reward

# Initial state
state = 'high'

# Run the simulation
print("Recycling Robot Simulation")
print("----------------------------")

for _ in range(10):  # Limiting the number of steps for simplicity
    print(f"Current State: {state}")
    print("Available Actions: ", states_actions[state])
    action = input("Choose an action (type search, wait, or recharge): ")
    
    if action in states_actions[state]:
        state, reward = simulate_action(state, action)
        print(f"Action: {action}, New State: {state}, Reward: {reward}")
    else:
        print("Invalid action. Please choose a correct action.")
    
    print("----------------------------")

print("Simulation ended.")
