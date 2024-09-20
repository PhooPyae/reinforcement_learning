from env_wrapper import SingleAgentSelfPlayEnv

def evaluate_agent(agent_policy, opponent_policy=None, agent_starts=True, num_episodes=100):
    # Create a new environment instance
    env = SingleAgentSelfPlayEnv(agent_starts=agent_starts)
    env.set_policies(agent_policy=agent_policy, opponent_policy=opponent_policy)
    
    win_count = 0
    draw_count = 0
    loss_count = 0

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = env.flatten_observation(obs)
            action_mask = obs['action_mask']
            action, _, _, _ = agent_policy.choose_action(state, action_mask)
            obs, reward, done, _ = env.step(action)
            
            if reward > 0:
              win_count += 1
            elif reward < 0:
              loss_count += 1
            else:
              draw_count += 1

    win_rate = win_count / num_episodes
    draw_rate = draw_count / num_episodes
    loss_rate = loss_count / num_episodes

    player = "Player 1" if agent_starts else "Player 2"
    print(f"Evaluation as {player}:")
    print(f"Win Rate: {win_rate:.2f}, Draw Rate: {draw_rate:.2f}, Loss Rate: {loss_rate:.2f}")

    return win_rate, draw_rate, loss_rate
