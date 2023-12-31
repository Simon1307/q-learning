import gymnasium as gym
import numpy as np
from utils import get_state, max_action, create_environment_space, plot_mean_rewards, save_action_value_func


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    # env._max_episode_steps = 1000
    n_episodes = 700
    alpha = 0.1
    gamma = 0.99
    eps = 1.0
    pos_space, vel_space, action_space = create_environment_space(20, 20, 3)

    states = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos, vel))

    Q = {}
    for state in states:
        for action in action_space:
            Q[state, action] = 0

    total_rewards = np.zeros(n_episodes)
    for i in range(n_episodes):
        done = False
        obs, _ = env.reset()
        state = get_state(obs, pos_space, vel_space)

        score = 0
        step_index = 0
        while not done:
            action = np.random.choice([0,1,2]) if np.random.random() < eps \
                    else max_action(Q, state)
            obs_, reward, done, _, _ = env.step(action)
            state_ = get_state(obs_, pos_space, vel_space)
            score += reward
            # Get next action with next state
            action_ = max_action(Q, state_)
            # Update the action value function
            Q[state, action] = Q[state, action] + \
                    alpha*(reward + gamma*Q[state_, action_] - Q[state, action])
            state = state_

        total_rewards[i] = score
        if (i+1) % 100 == 0:
            print('episode ', i+1, 'score ', score, 'epsilon %.3f' % eps)
        eps = eps - 2/n_episodes if eps > 0.01 else 0.01
    
    env.close()
    plot_mean_rewards(n_episodes, total_rewards, file_path='./artifacts/mountaincar.png')
    save_action_value_func(Q, file_path='./artifacts/mountaincar.pkl')
