import numpy as np
import gymnasium as gym
from gymnasium import spaces

import nasim
from nasim.envs.observation import Observation


class AugmentedObsWrapper(gym.Wrapper):
    """In NASim's Partially Observable setting, the agent only observes the
    outcom of it's latest action. This significantly increases the difficulty,
    and requires some form of memory or recurrence to find the optimal policy.

    With this wrapper the add information to the observation, in the form of
    and explicit belief representation. It also follows the logic that once
    information about hosts has been discovered, it remains valid thoughout
    the episode.

    The augemented observations as we call them, are contructed in the following
    manner:

    Augmented Observation = [Current Observation | Accumulated Knowledge Vector]

    Args:
        gym (_type_): _description_
    """
    def __init__(self, env):
        super().__init__(env)

        self.env = env
        self.host_vec_len = self.env.host_vec_len
        
        if self.env.flat_obs:
            obs_shape = self.last_obs.shape_flat()
            state_shape_flat = self.env.current_state.tensor.flatten().shape
            self.current_knowledge = np.zeros(state_shape_flat)
            obs_shape = (obs_shape[0] + state_shape_flat[0],)
        else:
            obs_shape = self.last_obs.shape()
            state_shape = self.env.current_state.tensor.shape
            self.current_knowledge = np.zeros(state_shape)
            obs_shape = tuple(obs_shape[0] + state_shape[0], obs_shape[1])

        obs_low, obs_high = Observation.get_space_bounds(self.env.scenario)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=obs_shape)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        #print(obs[:-self.host_vec_len])
        #print(self.current_knowledge)
        #print("-----")
        
        self.current_knowledge = np.maximum(self.current_knowledge, obs[:-self.host_vec_len])
        augmented_obs = np.concatenate((obs, self.current_knowledge))

        return augmented_obs, reward, terminated, truncated, info
    
    def reset(self, *, seed = None, options = None):
        self.current_knowledge[:] = 0 # Zero out the knowledge
        obs, info = super().reset(seed=seed, options=options)
        augmented_obs = np.concatenate((obs, self.current_knowledge))

        return augmented_obs, info
    
    def render(self):
        return self.env.render()
    
    def render_obs(self, mode="human", obs=None):
        return self.env.render_obs(mode=mode, obs=obs)

    def render_state(self, mode="human", state=None):
        return self.env.render_state(mode=mode, state=state)

    def render_action(self, action):
        return self.env.render_action(action)
    
    def action_masks(self):
        return self.env.action_masks()
    
    def close(self):
        return super().close()


if __name__ == '__main__':
    
    from generalization_env import NASimGenEnv

    env = NASimGenEnv()
    env = AugmentedObsWrapper(env)

    env.reset()

    for a in range(15):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(obs)
        print(reward)
        print(terminated)
        print(truncated)
        print(info)
        print()
        if terminated:
            break
