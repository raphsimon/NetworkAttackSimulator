import numpy as np
import gymnasium as gym
import nasim
from nasim import make_benchmark


class AggregatedObsWrapper(gym.Wrapper):
    """This wrapper is used to aggregate the observations the agent gets during
    an episode. Instead of the observations being the result of the action, we
    alter the observation to converge towards the state by aggregating all prior
    observations into a single observation. The idea is basically to keep track
    of everything we know about the environment within one single observation.

    Args:
        gym (_type_): _description_
    """
    def __init__(self, env):
        super().__init__(env)

        self.host_vec_len = len(self.unwrapped.current_state.hosts[0][1].vector)
        self.last_obs, _ = self.reset()

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

        self.envs_buffer = [nasim.make_benchmark(self.unwrapped.name, fully_obs=self.unwrapped.fully_obs)
                            for _ in range(num_envs)]
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.steps = 0
        new_env = np.random.choice(self.envs_buffer)
        # Map over important objects
        self.network = new_env.unwrapped.network
        self.current_state = new_env.unwrapped.current_state
        self.current_state = self.network.reset(self.current_state)
        self.last_obs = self.current_state.get_initial_observation(
            self.unwrapped.fully_obs
        )

        if self.unwrapped.flat_obs:
            obs = self.last_obs.numpy_flat()
        else:
            obs = self.last_obs.numpy()

        return obs, {}
    

class BetterRewardFeedback(gym.Wrapper):
    """We use this class to provide a better reward feedback to the agent,
    one that gives the agent a reward of +2 for successful actions but does
    so only once. This is such that we don't end up in a 'reward hacking'
    scenario.
    In the originial scenario, there is no difference in rewards between
    having a successful action, and a unsuccessful one. So everything relies
    on getting somehow to the end, and bootstrapping on that.
    """
    def __init__(self, env):
        super(BetterRewardFeedback, self).__init__(env)
        # We use a dictionary to track which actions have been successful
        self.action_tracker = {}

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if info['success'] == True:
            if action not in self.action_tracker:
                reward += 2
                self.action_tracker[action] = True

        return obs, reward, done, truncated, info
    

if __name__ == '__main__':

    def get_observation_sequence(env, optimal_actions):
        env.reset()
        obs_list = []

        for a in optimal_actions:
            o, r, d, trunc, info = env.step(a)
            if info['success'] == False: 
                # Exploits can fail since they have a success probability
                # Retry until we succeed
                keep_trying = True
                max_retries = 10
                retries = 0
                while keep_trying:
                    o, r, d, trunc, info = env.step(a)
                    retries += 1
                    keep_trying = not info['success'] and not retries >= max_retries
                if retries >= max_retries:
                    continue # Skip this action and try another
            print(o, a, r, d)
            obs_list.append(o)
            if d == True: # If we are done earlier with the environment.
                return obs_list
        return obs_list
    
    def test_aggregated_obs_wrapper():
        optimal_actions_tiny = [4, 2, 16, 17, 10, 11]
        optimal_actions_small = [4, 5, 6, 2, 13, 14, 15, 16, 17, 31, 32, 33, 29, 67, 68, 69, 70, 71]

        print('=' * 30 + ' PO Env ' + '=' * 30)
        ground_truth_env = gym.make('SmallGenPO-v0')
        orig_obs = get_observation_sequence(ground_truth_env, optimal_actions_small)

        print(ground_truth_env.unwrapped.scenario.get_description())
        print(ground_truth_env.unwrapped.scenario.sensitive_hosts)
        print(ground_truth_env.unwrapped.network.get_total_sensitive_host_value())

        host_addresses = [k for k in ground_truth_env.unwrapped.network.hosts.keys()]
        hosts = [ground_truth_env.unwrapped.current_state.get_host(addr) for addr in host_addresses]
        sensitive = [bool(h.sensitive) for h in hosts]
        print(sensitive)
        sensitive_int = [h.sensitive for h in hosts]
        print(sensitive_int)

        
        """
        print('=' * 30 + ' Aggregated Obs ' + '=' * 30)
        aggregated_obs_env = AggregatedObsWrapper(ground_truth_env)
        aggr_obs = get_observation_sequence(aggregated_obs_env, optimal_actions_tiny)

        print('=' * 30 + ' Fully Obs Env ' + '=' * 30)
        fully_obs_env = make_benchmark('tiny', fully_obs=True)
        orig_obs = get_observation_sequence(fully_obs_env, optimal_actions_tiny)
        """

    def print_sensitive_hosts():
        for i in range(100):
            print('=' * 80)
            ground_truth_env = gym.make('SmallGenPO-v0')
            ground_truth_env.reset()

            host_addresses = [k for k in ground_truth_env.unwrapped.network.hosts.keys()]
            hosts = [ground_truth_env.unwrapped.current_state.get_host(addr) for addr in host_addresses]
            sensitive = [bool(h.sensitive) for h in hosts]
            print(sensitive)
            sensitive_int = [h.sensitive for h in hosts]
            print(sensitive_int)

    def test_stochastic_episode_starts():
        env = gym.make('SmallGenPO-v0') # Without wrapper: +- 57.000
        env = StochasticEpisodeStarts(env)

        iterations = 100
        theoretical_rewards = 0

        for i in range(iterations):
            env.reset()
            theoretical_rewards += env.unwrapped.get_score_upper_bound()
        
        print(f'Theoretical rewards per episode: {sum(theoretical_rewards) // iterations}')
        print(theoretical_rewards)
        

    # test_stochastic_episode_starts()
    # test_aggregated_obs_wrapper()
    print_sensitive_hosts()
