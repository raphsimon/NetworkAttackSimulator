import numpy as np
import gymnasium as gym
import nasim
from nasim import make_benchmark

from generator import ScenarioGeneratorWIP
import time
import psutil

import nasim.scenarios

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
                 num_services: int=3, 
                 num_processes: int=3,
                 exploit_probs: float=0.8,
                 privesc_probs: float=0.8,
                 restrictiveness: int=2):
        super().__init__(env)

        self.num_exploits = num_os * num_services
        self.num_privescs = num_os * num_processes

        scenario = generator.generate(num_hosts=num_hosts, 
                                      num_services=4,
                                      num_os=2,
                                      num_processes=3,
                                      restrictiveness=2,
                                      num_exploits=self.num_exploits,
                                      num_privescs=self.num_privescs,
                                      exploit_probs=0.9,
                                      privesc_probs=0.9)
        
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

    def step(self, action):
        # TODO: Be careful about the selected action. If it surpasses the number
        #       if action > (num_hosts * actions_per_host), the we need to say
        #       that the action was unsuccessful.
        pass

    def reset(self):
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
                                      privesc_probs=0.9)
    
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
                                      privesc_probs=0.9)

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