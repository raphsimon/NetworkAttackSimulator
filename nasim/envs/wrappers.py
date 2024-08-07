import numpy as np
import gymnasium as gym
from nasim import make_benchmark


class StackedObsWrapper(gym.Wrapper):
    """This wrapper is used to stack the observations the agent gets during
    an episode. The purpose of this is to make the observations pseudo-Markovian
    since they would contain the history the observations.

    Args:
        gym (_type_): _description_
    """
    def __init__(self, env):
        super().__init__(env)

        self.host_vec_len = len(self.unwrapped.current_state.hosts[0][1].vector)
        self.last_obs, _ = self.reset()

        # TODO Look into keeping just the n last observations. Might have to
        # use a queue for this.

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        stacked_obs = np.maximum(self.last_obs, obs) # We overlay
        # We keep the auxiliary information from obs because 
        # we don't want to stack it
        stacked_obs[-self.host_vec_len:] = obs[-self.host_vec_len:]
        self.last_obs = stacked_obs # Update last_obs

        return stacked_obs, reward, terminated, truncated, info
    

class EmptyInfoWrapper(gym.Wrapper):
    """We use this wrapper to only return an empty dictionary as the info
    about the environment. This is done because the information contained
    in the dictionary is variable. This wrapper was specifically written
    for the Tianshou library.
    """
    def __init__(self, env):
        super(EmptyInfoWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, {}
    

class StochasticEpisodeStarts(gym.Wrapper):
    """This wrapper serves to allow for stochastic episode starts. To do this
    we utilize the generated environments capability of NASim to generate a
    buffer of environments. At episode reset, we assign a new environment s.t.
    the agent has to learn a more robust policy.
    """
    def __init__(self, env, num_envs=1000):
        super().__init__(env)

        self.envs_buffer = [make_benchmark(self.name, fully_obs=self.fully_obs)
                            for _ in range(num_envs)]
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.steps = 0
        new_env = np.random.choice(self.envs_buffer)
        # Map over important objects
        self.network = new_env.unwrapped.network
        self.current_state = new_env.unwrapped.current_state
        self.last_obs = self.current_state.get_initial_observation(
            self.fully_obs
        )

        if self.flat_obs:
            obs = self.last_obs.numpy_flat()
        else:
            obs = self.last_obs.numpy()

        return obs, {}