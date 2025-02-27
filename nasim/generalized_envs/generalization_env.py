import numpy as np
import gymnasium as gym
from gymnasium import spaces

import nasim

from generator import ScenarioGeneratorWIP
import time
import psutil

import nasim.scenarios
from nasim.envs.state import State
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

class GeneralizationWrapper(gym.Wrapper):
    """This wrapper serves the purpose to create an environment that allows
    the agent to generalize of different size networks, with hosts that have
    the same number of services and processes.

    We can have up to m hosts in an environment, where each host has can choose
    between o operating systems, m services, and p processes.
    """
    def __init__(self, 
                 env: gym.Env, 
                 max_num_hosts: int=15, 
                 min_num_hosts: int=5, 
                 num_os: int=2, 
                 num_services: int=4, 
                 num_processes: int=3,
                 exploit_probs: float=0.9,
                 privesc_probs: float=0.9,
                 restrictiveness: int=2,
                 step_limit: int=1000):
        super().__init__(env)

        self.max_num_hosts = max_num_hosts
        self.min_num_hosts = min_num_hosts
        self.num_os = num_os
        self.num_services = num_services
        self.num_processes = num_processes
        self.exploit_probs = exploit_probs
        self.privesc_probs = privesc_probs
        self.restrictiveness = restrictiveness
        self.step_limit = step_limit
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

        self.generator = ScenarioGeneratorWIP
        # TODO: QUESTIN: Do we generate a max_num_hosts_scenario at the start to
        #       get the Action space and Observation space size right?
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
        
        self.current_num_hosts = len(self.scenario.hosts)

        self.network = Network(scenario)
        self.current_state = State.generate_initial_state(self.network)
        self._renderer = None
        self.reset()

        if self.flat_actions:
            self.action_space = FlatActionSpace(self.scenario)
        else:
            self.action_space = ParameterisedActionSpace(self.scenario)

        if self.flat_obs:
            obs_shape = self.last_obs.shape_flat()
        else:
            obs_shape = self.last_obs.shape()
        obs_low, obs_high = Observation.get_space_bounds(self.scenario)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=obs_shape
        )

        self.steps = 0
        
        # TODO Create buffer with generated scenarios
        # How about we generate 1000 environments? -> But that takes a lot of time, no? 
        # can't we just generate the starting state?
        #   Honestly, it doesn't look like it would take a lot of time.
        #   Also generating them on the go with a seed would mean that our environment
        #   Has a smaller footprint overall, which is something very good.
        # I think we can already improve the execution time by removing the action space generation
        # TODO Define observation space bounds
        # TODO Need to fix the space for addresses and subnets. 
        #      Currently they can vary in size.
        #      One very IMPORTANT question to answer is how this later looks in the state.tensor
        #      Basically when generating the state from a scenario, what can go wrong?
        # TODO Define action space bounds
        # TODO When generated, look at the subnet sizes -> Do they change between scenarios?

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
        self.current_num_hosts = len(self.scenario.hosts)
        # Actions per host are all the exploits, privescs, and the scans (4)
        self.num_actions_per_host = len(scenario.exploits) + len(scenario.privescs) + 4
        self.network = Network(scenario)
        self.current_state = State.generate_initial_state(self.network)


    def step(self, action):
        if action > (self.current_num_hosts * self.num_actions_per_host):
            # We basically want to return a connection error here and not actually
            # execute the action on the network since it's out of bounds.
            action = self.action_space.get_action(action)
            action_res = ActionResult(False, 0.0, connection_error=True)
            reward = action_res.value - action.cost
            self.steps += 1
            done = False
            step_limit_reached = (
                self.scenario.step_limit is not None
                and self.steps >= self.scenario.step_limit
            )
            # TODO define the observation to return. Should be all empty except the connection error.
            return obs, reward, done, step_limit_reached, action_res.info()
        else:
            obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        # Here we need to pick one of the generated scenarios and switch out 
        # everything related.
        pass

    def close(self):
        return super().close()


if __name__ == '__main__':

    # TRY NOT TO MODIFY: seeding
    #np.random.seed(4444)

    generator = ScenarioGeneratorWIP()    

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