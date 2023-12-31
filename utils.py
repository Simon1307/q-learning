import numpy as np
import matplotlib.pyplot as plt
import pickle


def get_state(observation, pos_space, vel_space):
    """
    Function to discretize the observation of the environment.
    
    Args:
        observation (numpy.ndarray): The continuous state of the environment.
        pos_space (numpy.ndarray): Discretized positions in the space.
        vel_space (numpy.ndarray): Discretized velocities in the space.
    
    Returns:
        tuple: The discritized state.
    """
    pos, vel =  observation
    pos_bin = int(np.digitize(pos, pos_space))
    vel_bin = int(np.digitize(vel, vel_space))
    return (pos_bin, vel_bin)

def max_action(Q, state, actions=[0, 1, 2]):
    """
    This function returns the action with highest value
    from the current action value function Q.
    
    Args:
        Q (dict): The action value function.
        state (tuple): Current discritized states.
        actions (list): List of actions available in the environment.
    
    Returns:
        int: The action with highest value
    """
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return action


def create_environment_space(num_pos_spaces, num_vel_spaces, num_actions):
    """
    Create discretized spaces for the MountainCar environment.

    Args:
        num_pos_spaces (int): Number of discretized positions in the space.
        num_vel_spaces (int): Number of discretized velocities in the space.
        num_actions (int): Number of actions available in the environment.

    Returns:
        tuple: A tuple containing the discretized position space, velocity space, and action space.
               The position and velocity spaces are created using linspace, and the action space
               is a list of integers ranging from 0 to num_actions - 1.
    """
    pos_space = np.linspace(-1.2, 0.6, num_pos_spaces)
    vel_space = np.linspace(-0.07, 0.07, num_vel_spaces)
    action_space = list(np.arange(num_actions))
    return pos_space, vel_space, action_space


def plot_mean_rewards(n_episodes, total_rewards, file_path):
    """
    Plot the moving window (50 episodes) of the mean of total rewards over episodes.

    Args:
        n_episodes (int): Total number of episodes during the training.
        total_rewards (numpy.ndarray): Array containing the total rewards for each episode.
        file_path (str): File path to save the plot.

    Returns:
        None: The function generates a plot and saves it to the specified file_path.
    """
    mean_rewards = np.zeros(n_episodes)
    for t in range(n_episodes):
        mean_rewards[t] = np.mean(total_rewards[max(0, t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.title('Total Rewards Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.grid(True)
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()


def save_action_value_func(Q, file_path):
    """
    Save the action value function (Q) to a binary file.

    Args:
        Q (dict): The action value function represented as a dictionary.
        file_path (str): File path to save the action value function.

    Returns:
        None: The function saves the action value function to the specified file_path.
    """    
    f = open(file_path, 'wb')
    pickle.dump(Q,f)
    f.close()


def load_action_value_func(file_path):
    """
    Load the action value function (Q) from a binary file.

    Args:
        file_path (str): File path to load the action value function.

    Returns:
        dict: The loaded action value function represented as a dictionary.
    """    
    pickle_in = open(file_path, 'rb')
    Q = pickle.load(pickle_in)
    return Q
