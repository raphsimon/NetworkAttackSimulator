import numpy as np
import gym
from nasim import make_benchmark


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
        
    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        self.steps = 0
        new_env = np.random.choice(self.envs_buffer)
        # Map over important objects
        self.network = new_env.unwrapped.network
        self.current_state = new_env.unwrapped.current_state
        self.current_state = self.network.reset(self.current_state)
        self.last_obs = self.current_state.get_initial_observation(
            self.fully_obs
        )

        if self.flat_obs:
            obs = self.last_obs.numpy_flat()
        else:
            obs = self.last_obs.numpy()

        return (obs, {}) if return_info else obs
    

if __name__ == '__main__':
    
    import nasim
    #env = make_benchmark('tiny')
    env = gym.make('SmallGenPO-v0')
    env = StochasticEpisodeStarts(env)

    for i in range(5):
        env.reset()
        print(env.current_state.tensor)
        print(env.network.get_sensitive_hosts())
        print('-------------------')
