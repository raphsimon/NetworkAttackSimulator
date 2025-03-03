import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys

import nasim

from nasim.generalized_envs.generator import ModifiedScenarioGenerator
import time
import psutil

import nasim.scenarios
from nasim.envs.state import State
from nasim.envs.render import Viewer
from nasim.envs.network import Network
from nasim.envs.observation import Observation
from nasim.envs.action import Action, ActionResult, FlatActionSpace, ParameterisedActionSpace


"""
Some discussion points:
- Why generate the action space every time?
    Think it allows us to have the right addresses of each host for the 
    respective actions. We also defined the action space in such a way that 
    it generates the same, given the generation parameters. There is always
    one exploit for every OS-service combination. There is always one privsec 
    action for every OS-process combination.
"""

class NASimGenEnv(gym.Env):
    """This wrapper serves the purpose to create an environment that allows
    the agent to generalize of different size networks, with hosts that have
    the same number of services and processes.

    We can have up to m hosts in an environment, where each host has can choose
    between o operating systems, m services, and p processes.
    """

    metadata = {'render_modes': ["human", "ansi"]}
    render_mode = None
    reward_range = (-float('inf'), float('inf'))

    action_space = None
    observation_space = None
    current_state = None
    last_obs = None

    def __init__(self,
                 fully_obs=False,
                 flat_actions=True,
                 flat_obs=True,
                 render_mode=None):
        # TODO: These are hardcoded values for now, but they should be
        #       parameters that can be passed to the environment.
        self.max_num_hosts = 15
        self.min_num_hosts = 5
        self.num_os = 2
        self.num_services = 4
        self.num_processes = 3
        self.exploit_probs = 0.9
        self.privesc_probs = 0.9
        self.restrictiveness = 2
        self.step_limit = 1000
        # Calculate the number of exploits and privescs
        self.num_exploits = self.num_os * self.num_services
        self.num_privescs = self.num_os * self.num_processes
        # address_space_bounds : (int, int), optional
        #     bounds for the (subnet#, host#) address space. If None bounds will
        #     be determined by the number of subnets in the scenario and the max
        #     number of hosts in any subnet.
        self.address_space_bounds = (6, 5)
        # Basically we define it to be large enough, this means that everything
        # should fit into our address space. Hosts are indexed by subnet, and
        # then their host number within that subnet.
        # Using this parameter, our host vector size will remain constant accross 
        # all network sizes.
        # We use 5 hosts, because of the USER_SUBNET_SIZE = 5 constant. This allows
        # us to have a compacter observation space. So we don't define it larger
        # than it needs to be.

        # TODO Verify correct seeding
        # But for this, don't we have to remove any generation of environment
        # properties form the __init__() function? Since basically, we create
        # the environment. env = gym.make('NASimGenEnv'). Then this means that
        # already generate the state and action space etc. with one arbitrary
        # seed. Then if we call env.seed(1), only then we change the seed, but
        # we already generated something on a different seed.
        # Question: If we then call env.reset(), will they all reset to the same
        # state? If yes, then we are good, since we have to call env.reset()
        # before interacting with the environment.

        self.generator = ModifiedScenarioGenerator()
        # At the start we generate a scenario with the maximum number of hosts
        # such that we get the address and observation space bounds right. 
        self.scenario = self.generator.generate(num_hosts=self.max_num_hosts, 
                                                num_services=self.num_services,
                                                num_os=self.num_os,
                                                num_processes=self.num_processes,
                                                restrictiveness=self.restrictiveness,
                                                num_exploits=self.num_exploits,
                                                num_privescs=self.num_privescs,
                                                exploit_probs=self.exploit_probs,
                                                privesc_probs=self.privesc_probs,
                                                address_space_bounds=self.address_space_bounds,
                                                step_limit=self.step_limit)
        self.name = self.scenario.name
        self.fully_obs = fully_obs
        self.flat_actions = flat_actions
        self.flat_obs = flat_obs
        self.render_mode = render_mode

        self.current_num_hosts = len(self.scenario.hosts)
        print('Generated scenario dimensions in __init__():', self.scenario.get_observation_dims())

        self.network = Network(self.scenario)
        self.current_state = State.generate_initial_state(self.network)
        self.last_obs = self.current_state.get_initial_observation(self.fully_obs)
        self.host_vec_len = self.last_obs.shape()[1] # We have to take the host vector
        # length from here because when vectorizing the host, we do some more encoding,
        # therefore making it bigger.
        print(f"{self.last_obs.shape()=}")
        print(f"{self.host_vec_len=}")

        if self.flat_actions:
            self.action_space = FlatActionSpace(self.scenario)
        else:
            self.action_space = ParameterisedActionSpace(self.scenario)

        if self.flat_obs:
            obs_shape = self.last_obs.shape_flat()
            self.connection_error_obs = np.zeros(obs_shape)
        else:
            obs_shape = self.last_obs.shape()
            self.connection_error_obs = np.zeros(obs_shape)
        obs_low, obs_high = Observation.get_space_bounds(self.scenario)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=obs_shape
        )

        print(f"{self.observation_space=}")

        self.steps = 0
        
        # TODO Create buffer with generated scenarios
        # How about we generate 1000 environments? -> But that takes a lot of time, no? 
        # can't we just generate the starting state?
        #   Honestly, it doesn't look like it would take a lot of time.
        #   Also generating them on the go with a seed would mean that our environment
        #   Has a smaller footprint overall, which is something very good.
        # I think we can already improve the execution time by removing the action space generation
        
        # TODO When generated, look at the subnet sizes -> Do they change between scenarios?
        # Answer: Yes: The subnet sizes can change, and this affects the action space,
        # because a host is indexed by its subnet and number in the subnet. Does this matter for
        # the actions?

    def _generate_new_network(self):
        num_hosts = np.random.randint(self.min_num_hosts, self.max_num_hosts+1)
        scenario = self.generator.generate(num_hosts=num_hosts, 
                                                num_services=self.num_services,
                                                num_os=self.num_os,
                                                num_processes=self.num_processes,
                                                restrictiveness=self.restrictiveness,
                                                num_exploits=self.num_exploits,
                                                num_privescs=self.num_privescs,
                                                exploit_probs=self.exploit_probs,
                                                privesc_probs=self.privesc_probs,
                                                address_space_bounds=self.address_space_bounds,
                                                step_limit=self.step_limit)
        self.current_num_hosts = len(scenario.hosts)
        # Actions per host are all the exploits, privescs, and the scans (4)
        self.num_actions_per_host = len(scenario.exploits) + len(scenario.privescs) + 4
        self.network = Network(scenario)
        print('Newly generated scenario dimensions:', scenario.get_observation_dims())
        self.current_state = State.generate_initial_state(self.network)

        # TODO: Think about this. We get 
        if self.flat_actions:
            self.action_space = FlatActionSpace(scenario)
        else:
            self.action_space = ParameterisedActionSpace(scenario)
        
        self.scenario = scenario

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        """Run one step of the environment using action.

        Implements gymnasium.Env.step().

        Parameters
        ----------
        action : Action or int or list or NumpyArray
            Action to perform. If not Action object, then if using
            flat actions this should be an int and if using non-flat actions
            this should be an indexable array.

        Returns
        -------
        numpy.Array
            observation from performing action
        float
            reward from performing action
        bool
            whether the episode reached a terminal state or not (i.e. all
            target machines have been successfully compromised)
        bool
            whether the episode has reached the step limit (if one exists)
        dict
            auxiliary information regarding step
            (see :func:`nasim.env.action.ActionResult.info`)
        """
        # Make sure action is within bouds of currently generated network
        if action >= (self.current_num_hosts * self.num_actions_per_host):
            # We basically want to return a connection error here and not actually
            # execute the action on the network since it's out of bounds.
            action_res = ActionResult(False, 0.0, connection_error=True)
            reward = action_res.value - 1.0 # We set cost of the wrong action to -1.0

            done = False
            step_limit_reached = (
                self.step_limit is not None
                and self.steps >= self.step_limit
            )
            
            obs = Observation((self.max_num_hosts, self.host_vec_len))
            obs.from_action_result(action_res)
            self.last_obs = obs
            
            if self.flat_obs:
                obs = obs.numpy_flat()
            else:
                obs = obs.numpy()
            
            self.steps += 1

            return obs, reward, done, step_limit_reached, action_res.info()
        else:
            next_state, obs, reward, done, info = self.generative_step(
                self.current_state,
                action
            )
            self.current_state = next_state
            self.last_obs = obs

            obs_np = obs.numpy()
            print("Obs numpy shape:", obs_np.shape)
            num_hosts_to_insert = self.max_num_hosts - self.current_num_hosts
            new_row = np.zeros((num_hosts_to_insert, obs_np.shape[1]))  # Create a row with the same number of columns
            new_obs = np.insert(obs_np, obs_np.shape[0]-1, new_row, axis=0)

            assert new_obs.shape == (self.max_num_hosts+1, self.host_vec_len), \
                f"Error: Observation shape: {new_obs.shape}, is different from exptected shape: {(self.max_num_hosts+1, self.host_vec_len)}"

            # TODO: Check whether this works as intended.
            if self.flat_obs:
                obs = new_obs.flatten()
            else:
                obs = new_obs

            self.steps += 1

            step_limit_reached = (
                self.step_limit is not None
                and self.steps >= self.step_limit
            )
            return obs, reward, done, step_limit_reached, info

    def generative_step(self, state, action):
        """Run one step of the environment using action in given state.

        Parameters
        ----------
        state : State
            The state to perform the action in
        action : Action, int, list, NumpyArray
            Action to perform. If not Action object, then if using
            flat actions this should be an int and if using non-flat actions
            this should be an indexable array.

        Returns
        -------
        State
            the next state after action was performed
        Observation
            observation from performing action
        float
            reward from performing action
        bool
            whether a terminal state has been reached or not
        dict
            auxiliary information regarding step
            (see :func:`nasim.env.action.ActionResult.info`)
        """
        if not isinstance(action, Action):
            action = self.action_space.get_action(action)

        next_state, action_obs = self.network.perform_action(
            state, action
        )
        obs = next_state.get_observation(
            action, action_obs, self.fully_obs
        )
        done = self.goal_reached(next_state)
        reward = action_obs.value - action.cost
        return next_state, obs, reward, done, action_obs.info()
    
    def reset(self, *, seed=None, options=None):
        # Here we need to pick one of the generated scenarios and switch out 
        # everything related.
        super().reset(seed=seed, options=options)
        self.steps = 0
        self.scenario = self._generate_new_network()
        self.current_state = self.network.reset(self.current_state)
        self.last_obs = self.current_state.get_initial_observation(
            self.fully_obs
        )
        print("self.last_obs.shape() in env.reset()", self.last_obs.shape())

        last_obs_np = self.last_obs.numpy()
        print("In env.reset(): last_obs_np.shape =", last_obs_np.shape)
        num_hosts_to_insert = self.max_num_hosts - self.current_num_hosts
        print("In env.reset(): num_hosts_to_insert =", num_hosts_to_insert)
        new_row = np.zeros((num_hosts_to_insert, last_obs_np.shape[1]))  # Create a row with the same number of columns
        buffered_obs = np.insert(last_obs_np, last_obs_np.shape[0]-1, new_row, axis=0)

        assert buffered_obs.shape == (self.max_num_hosts+1, self.host_vec_len), \
            f"Error: Observation shape: {buffered_obs.shape}, is different from exptected shape: {(self.max_num_hosts+1, self.host_vec_len)}"

        if self.flat_obs:
            obs = buffered_obs.flatten()
        else:
            obs = buffered_obs

        print("In env.reset(): obs.shape = ", obs.shape)

        return obs, {}

    def render(self):
        """Render environment.

        Implements gymnasium.Env.render().

        See render module for more details on modes and symbols.

        """
        if self.render_mode is None:
            return
        return self.render_obs(mode=self.render_mode, obs=self.last_obs)
    
    def render_network_graph(self, ax=None, show=False):
        """Render a plot of network as a graph with hosts as nodes arranged
        into subnets and showing connections between subnets. Renders current
        state of network.

        Parameters
        ----------
        ax : Axes
            matplotlib axis to plot graph on, or None to plot on new axis
        show : bool
            whether to display plot, or simply setup plot and showing plot
            can be handled elsewhere by user
        """
        if self._renderer is None:
            self._renderer = Viewer(self.network)
        state = self.current_state
        self._renderer.render_graph(state, ax, show)
    
    def get_minimum_hops(self):
        """Get the minimum number of network hops required to reach targets.

        That is minimum number of hosts that must be traversed in the network
        in order to reach all sensitive hosts on the network starting from the
        initial state

        Returns
        -------
        int
            minumum possible number of network hops to reach target hosts
        """
        return self.network.get_minimal_hops()
    
    # TODO: Test this method.
    def action_masks(self):
        """Get a vector mask for valid actions. The mask is based on whether
        a host has been discovered or not.

        Returns
        -------
        ndarray
            numpy vector of 1's and 0's, one for each action. Where an
            index will be 1 if action is valid given current state, or
            0 if action is invalid.
        """
        assert isinstance(self.action_space, FlatActionSpace), \
            "Can only use action mask function when using flat action space"

        # Create a list of bools telling us if host i has been discovered
        discovered = [h[1].discovered for h in self.current_state.hosts]

        assert self.action_space.n / self.num_actions_per_host == len(discovered), \
            "Hosts don't all have the same amout of actions"
        
        # Extend discovered list to self.max_num_hosts
        discovered.extend([False] * (self.max_num_hosts - len(discovered)))

        # Repeat the bool num_actions_per_host times
        mask = np.repeat(discovered, self.num_actions_per_host)

        return mask

    def get_score_upper_bound(self):
        """Get the theoretical upper bound for total reward for scenario.

        The theoretical upper bound score is where the agent exploits only a
        single host in each subnet that is required to reach sensitive hosts
        along the shortest bath in network graph, and exploits the all
        sensitive hosts (i.e. the minimum network hops). Assuming action cost
        of 1 and each sensitive host is exploitable from any other connected
        subnet (which may not be true, hence being an upper bound).

        Returns
        -------
        float
            theoretical max score
        """
        max_reward = self.network.get_total_sensitive_host_value()
        max_reward += self.network.get_total_discovery_value()
        max_reward -= self.network.get_minimal_hops()
        return max_reward
    
    def goal_reached(self, state=None):
        """Check if the state is the goal state.

        The goal state is when all sensitive hosts have been compromised.

        Parameters
        ----------
        state : State, optional
            a state, if None will use current_state of environment
            (default=None)

        Returns
        -------
        bool
            True if state is goal state, otherwise False.
        """
        if state is None:
            state = self.current_state
        return self.network.all_sensitive_hosts_compromised(state)
    
    def __str__(self):
        output = [
            "NASimGenEnv:",
            f"name={self.name}",
            f"fully_obs={self.fully_obs}",
            f"flat_actions={self.flat_actions}",
            f"flat_obs={self.flat_obs}"
        ]
        return "\n  ".join(output)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == '__main__':

    print()
    print()
    print()

    env = gym.make('GenPO-v0')

    print("Observation space upron creation:", env.observation_space.shape)
    print("Action space upon creation:", env.action_space)

    obs, info = env.reset()
    print("Observation shape after reset:", obs.shape)
    print("Another reset:")
    obs, info = env.reset()
    print("Observation shape after reset:", obs.shape)

    # TODO: Fix obseration space problems. It's not consistent!

    obs, r, d, t, i = env.step(0)
    print(obs, r, d, t, i)
    print("Observation shape after step:", obs.shape)

    obs, r, d, t, i = env.step(156)
    print(obs, r, d, t, i)
    print("Observation shape after step:", obs.shape)

    sys.exit(0)
    # TRY NOT TO MODIFY: seeding
    #np.random.seed(4444)

    # TODO: Verify host values. Are they in line with our formalization?

    generator = ModifiedScenarioGenerator()    

    orig_generator = nasim.scenarios.ScenarioGenerator()  

    num_hosts = np.random.randint(5, 15)
    scenario = generator.generate(num_hosts=num_hosts, 
                                      num_services=4, 
                                      num_os=2, 
                                      num_processes=3, 
                                      restrictiveness=2, 
                                      exploit_probs=0.9,
                                      privesc_probs=0.9,
                                      address_space_bounds=(6,5))
    
    #scenario_2 = generator.generate(num_hosts=num_hosts, 
    #                                  num_services=4, 
    #                                  num_os=2, 
    #                                  num_processes=3, 
    #                                  restrictiveness=2, 
    #                                  exploit_probs=0.9,
    #                                  privesc_probs=0.9)
    #
    #print(scenario.exploits == scenario_2.exploits)
    #print(scenario.privescs == scenario_2.privescs)
    # Verification that the actions are indeed the same across different 
    # configurations. It looks like this is indeed the case.

    desc = scenario.get_description()
    # What's the size of the host vector?
    print(desc)
    print('Observation Dims:', desc['Observation Dims'])
    # Observation: The observations size changes with the number of hosts
    # We have num_hosts+1 rows, which is expected. But the column count
    # is not constant.
    # - Is it influenced by the host address?
    # - Is it influenced by the number of subnets?
    print('Address space size:', scenario.hosts)
    print('Subnet space size:', scenario.subnets)
    # Observation: When using 15 hosts, it looks like the address space
    # takes up 6 columns, and the subnet space takes up 5 columns. These
    # would then be our upper bounds.
    # The subnet sizes are just list to wich hosts are appended.

    print('='*90)
    print('='*23 + ' Comparison with original ScenarioGenerator ' + '='*23)
    print('='*90)

    orig_generator = nasim.scenarios.ScenarioGenerator()  

    num_hosts = np.random.randint(5, 15)
    orig_scenario = orig_generator.generate(num_hosts=num_hosts, 
                                      num_services=4, 
                                      num_os=2, 
                                      num_processes=3, 
                                      restrictiveness=2, 
                                      exploit_probs=0.9,
                                      privesc_probs=0.9,
                                      address_space_bounds=(6,5))

    orig_desc = orig_scenario.get_description()
    print(orig_scenario.exploits)
    print(orig_scenario.privescs)
    print(desc)
    print('Observation Dims:', orig_desc['Observation Dims'])
    print('Address space size:', orig_scenario.hosts)
    print('Subnet space size:', orig_scenario.subnets)

    """
    # Code for timing the generation.
    start_time = time.time()
    process = psutil.Process()

    envs = []

    for _ in range(10000):
        num_hosts = np.random.randint(5, 15)
        scenario = generator.generate(num_hosts=num_hosts, 
                                      num_services=4, 
                                      num_os=2, 
                                      num_processes=3, 
                                      restrictiveness=2, 
                                      exploit_probs=0.9,
                                      privesc_probs=0.9)
        envs.append(scenario)

    end_time = time.time()
    execution_time = end_time - start_time
    memory_info = process.memory_info()

    print(f"Execution time for 10,000 runs: {execution_time} seconds")
    print(f"Memory usage: {memory_info.rss / (1024 * 1024)} MB")
    """