import pickle
import gymnasium as gym
from gymnasium.utils.save_video import save_video
from utils import get_state, max_action, create_environment_space, load_action_value_func


if __name__ == '__main__':
    Q = load_action_value_func(file_path='./artifacts/mountaincar.pkl')
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    pos_space, vel_space, action_space = create_environment_space(20, 20, 3)

    states = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos, vel))

    score = 0
    done = False
    obs, _ = env.reset()
    state = get_state(obs, pos_space, vel_space)

    step_index = 0
    frame_list = []
    while not done:
        action = max_action(Q, state)
        obs_, reward, done, _, _ = env.step(action)
        state_ = get_state(obs_, pos_space, vel_space)
        score += reward
        action_ = max_action(Q, state_)
        state = state_
        frame = env.render()
        frame_list.append(frame)

    save_video(
        frame_list,
        'artifacts',
        fps=env.metadata['render_fps'],
        step_starting_index=step_index,
    )
    step_index += 1

    env.close()
