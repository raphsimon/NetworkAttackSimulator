import nasim
import gymnasium as gym
import numpy as np


def binary_array_to_int(arr):
    """We use this functino to obtain a unique integer representation of
    the starting state of the environment. We want to 
    """
    # Ensure array is 1D
    arr = arr.flatten()
    # Convert to integer
    return np.packbits(arr).dot(2**np.arange(len(np.packbits(arr)))[::-1])

def test_reset(num_resets, seed):
    print(f"Testing MultiTaskNASimEnv reset for {num_resets} iters with initial seed {seed}.")
    env = gym.make('MultiTaskPO-v0', seed=seed)
    state_tracking = []
    for _ in range(num_resets):
        env.reset()
        obs_as_int = binary_array_to_int(np.array(env.current_state.tensor, dtype=int))
        print("State int repr after reset:", obs_as_int, "\tNetwork size: ", env.current_num_hosts)
        if obs_as_int not in state_tracking:
            state_tracking.append(obs_as_int)
    print(f"Encountered {len(state_tracking)} different states.")
        

#test_reset(24, 42)
#test_reset(24, 1234)
#test_reset(24, 77777)

genpo = gym.make('GenPO-v0')
multi_task = gym.make('MultiTaskPO-v0')

print("Observation spaces are equal:", genpo.observation_space.shape == multi_task.observation_space.shape)
print("Action spaces are equal:", genpo.action_space.n == multi_task.action_space.n)

genpo.reset()
multi_task.reset()


print("Observation spaces are equal:", genpo.observation_space.shape == multi_task.observation_space.shape)
print("Action spaces are equal:", genpo.action_space.n == multi_task.action_space.n)