""" The main Environment class for NASim: NASimEnv.

The NASimEnv class is the main interface for agents interacting with NASim.
"""

import numpy as np

from nasim.env.state import State
from nasim.env.action import Action
from nasim.env.render import Viewer
from nasim.env.network import Network


class NASimEnv:
    """ A simulated computer network environment for pen-testing.

    ...

    Attributes
    ----------
    current_state : State
        the current knowledge the agent has observed
    action_space : List(Action)
        list of all actions allowed for environment
    last_obs : the observability mode of the environment.


    For both modes the dimensions are the same for the returned
    state/observation. The only difference is for the POMDP mode, for parts of
    the state that were not observed the value returned will be a non-obs
    value (i.e. 0 in most cases).
    """
    rendering_modes = ["readable", "ASCI"]

    action_space = None
    current_state = None

    def __init__(self, scenario, fully_obs=True):
        """
        Parameters
        ----------
        scenario : Scenario
            Scenario object, defining the properties of the environment
        fully_obs : bool, optional
            The observability mode of environment, if True then uses fully
            observable mode, otherwise is partially observable (default=True)
        """
        self.scenario = scenario
        self.fully_obs = fully_obs

        self.network = Network(scenario)
        self.address_space = scenario.address_space
        self.action_space = Action.load_action_space(self.scenario)

        self.current_state = State.generate_initial_state(self.network)
        self.last_obs = None
        self._renderer = None
        self.reset()

    def reset(self):
        """Reset the state of the environment and returns the initial state.

        Returns
        -------
        Observation
            the initial observation of the environment
        """
        self.current_state = self.network.reset(self.current_state)
        self.last_obs = self.current_state.get_initial_observation(
            self.fully_obs
        )
        return self.last_obs

    def step(self, action):
        """Run one step of the environment using action.

        N.B. Does not return a copy of the observation, and observation
        is changed by simulator. So if you need to store the observation
        you will need to copy it.

        Parameters
        ----------
        action : Action or int
            Action object from action space or index of action in action space

        Returns
        -------
        obs : Observation
            current observation of environment
        reward : float
            reward from performing action
        done : bool
            whether the episode has ended or not
        info : dict
            other information regarding step (see ActionObservation.info())
        """
        next_state, obs, reward, done, info = self.generative_step(
            self.current_state,
            action
        )
        self.current_state = next_state
        self.last_obs = obs
        return obs, reward, done, info

    def generative_step(self, state, action):
        """Run one step of the environment using action in given state.

        Parameters
        ----------
        state : State
            The state to perform the action in
        action : Action or int
            Action object from action space or index of action in action space

        Returns
        -------
        next_state : State
            the state after action was performed
        obs : Observation
            current observation of environment
        reward : float
            reward from performing action
        done : bool
            whether the episode has ended or not
        info : dict
            other information regarding step (see ActionObservation.info())
        """
        assert isinstance(action, (Action, int)), \
            "Step action must be an integer or an Action object"
        if isinstance(action, int):
            action = self.action_space[action]

        next_state, action_obs = self.network.perform_action(
            state,
            action
        )
        obs = next_state.get_observation(action,
                                         action_obs,
                                         self.fully_obs)
        done = self.is_goal(next_state)
        reward = action_obs.value - action.cost
        return next_state, obs, reward, done, action_obs.info()

    def generate_random_initial_state(self):
        """Generates a random initial state for environment.

        This only randomizes the host configurations (os, services)
        using a uniform distribution, so may result in networks where
        it is not possible to reach the goal.

        Returns
        -------
        State
            A random initial state
        """
        return State.generate_random_initial_state(self.network)

    def generate_initial_state(self):
        """Generate the initial state for the environment.

        Note, this does not reset the current state of the environment (use the
        reset function for that).

        Returns
        -------
        State
            The initial state
        """
        return State.generate_initial_state(self.network)

    def render(self, mode="readable", obs=None):
        """Render observation.

        See render module for more details on modes and symbols.

        Parameters
        ----------
        mode : str
            rendering mode
        obs : Observation, optional
            the observation to render, if None will render last observation
            (default=None)
        """
        if obs is None:
            obs = self.last_obs
        if self._renderer is None:
            self._renderer = Viewer(self.network)

        if mode == "readable":
            self._renderer.render_readable(obs)
        else:
            print("Please choose correct render mode from :"
                  f"{self.rendering_modes}")

    def render_state(self, mode="readable", state=None):
        """Render state.

        See render module for more details on modes and symbols.

        If mode = ASCI:
            Machines displayed in rows, with one row for each subnet and
            hosts displayed in order of id within subnet

        Parameters
        ----------
        mode : str
            rendering mode
        state : State, optional
            the State to render, if None will render current state
            (default=None)
        """
        if state is None:
            state = self.current_state
        if self._renderer is None:
            self._renderer = Viewer(self.network)

        if mode == "readable":
            self._renderer.render_readable_state(state)
        else:
            print("Please choose correct render mode from :"
                  f"{self.rendering_modes}")

    def render_action(self, action):
        """Renders human readable version of action.

        This is mainly useful for getting a text description of the action
        that corresponds to a given integer.

        Parameters
        ----------
        action : int or Action
            the action to render
        """
        if isinstance(action, int):
            action = self.action_space[action]
        print(action)

    def render_episode(self, episode, width=7, height=7):
        """Render an episode as sequence of network graphs, where an episode
        is a sequence of (state, action, reward, done) tuples generated from
        interactions with environment.

        Parameters
        ----------
        episode : list
            list of (State, Action, reward, done) tuples
        width : int
            width of GUI window
        height : int
            height of GUI window
        """
        if self._renderer is None:
            self._renderer = Viewer(self.network)
        self._renderer.render_episode(episode)

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

    def get_state_shape(self, flat=True):
        """Get the shape of an environment state representation

        Parameters
        ----------
        flat : bool, optional
            whether to get shape of flattened state (True) or not (False)
            (default=True)

        Returns
        -------
        (int, int)
            shape of state representation
        """
        if flat:
            return self.current_state.flat_shape()
        return self.current_state.shape()

    def get_obs_shape(self, flat=True):
        """Get the shape of an environment observation representation

        Parameters
        -----------
        flat : bool, optional
            whether to get shape of flattened observation (True) or not (False)
            (default=True)

        Returns
        -------
        (int, int)
            shape of observation representation
        """
        # observation has same shape as state
        return self.get_state_shape(flat)

    def get_num_actions(self):
        """Get the size of the action space for environment

        Returns
        -------
        num_actions : int
            action space size
        """
        return len(self.action_space)

    def get_minimum_actions(self):
        """Get the minimum possible actions required to exploit all sensitive
        hosts from the initial state

        Returns
        -------
        minimum_actions : int
            minumum possible actions
        """
        return self.network.get_minimal_steps()

    def get_action_mask(self):
        """Get a vector mask for valid actions.

        Returns
        -------
        ndarray
            numpy vector of 1's and 0's, one for each action. Where an
            index will be 1 if action is valid given current state, or
            0 if action is invalid.
        """
        mask = np.zeros(len(self.action_space), dtype=np.float)
        for i, action in enumerate(self.action_space):
            if self.network.host_discovered(action.target):
                mask[i] = 1
        return mask

    def get_best_possible_score(self):
        """Get the best score possible for this environment, assuming action
        cost of 1 and each sensitive host is exploitable from any other
        connected subnet.

        The theoretical best score is where the agent only exploits a single
        host in each subnet that is required to reach sensitive hosts along
        the shortest bath in network graph, and exploits the two sensitive
        hosts (i.e. the minial steps)

        Returns
        -------
        max_score : float
            theoretical max score
        """
        max_reward = self.network.get_total_sensitive_host_value()
        max_reward -= self.network.get_minimal_steps()
        return max_reward

    def is_goal(self, state):
        """Check if the current state is the goal state.
        The goal state is  when all sensitive hosts have been compromised
        """
        return self.network.all_sensitive_hosts_compromised(state)
