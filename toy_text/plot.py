import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot(rewards):
    running_average = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=running_average, label='Q Learning Agent')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()