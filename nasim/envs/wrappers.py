import numpy as np
import gymnasium as gym


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

    def reset(self, **kwargs):
        # Modify the reset method if needed
        obs, info = self.env.reset(**kwargs)
        # Perform any additional processing on obs or info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        stacked_obs = np.maximum(self.last_obs, obs) # We overlay
        # We keep the auxiliary information from obs because 
        # we don't want to stack it
        stacked_obs[-self.host_vec_len:] = obs[-self.host_vec_len:]
        self.last_obs = stacked_obs # Update last_obs

        return stacked_obs, reward, terminated, truncated, info

    def render(self):
        # Modify the render method if needed
        return self.env.render()
    

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