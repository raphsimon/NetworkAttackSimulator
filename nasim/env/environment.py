""" The main Environment class for NASim: NASimEnv.

The NASimEnv class is the main interface for agents interacting with NASim.
"""
import gym
import numpy as np
from gym import spaces

from nasim.env.state import State
from nasim.env.render import Viewer
from nasim.env.network import Network
from nasim.env.action import Action, FlatActionSpace, ParameterisedActionSpace


class NASimEnv(gym.Env):
    """ A simulated computer network environment for pen-testing.

    Implements the OpenAI gym interface.

    ...

    Attributes
    ----------
    action_space : FlatActionSpace or ParameterisedActionSpace
        Action space for environment.
        If *flat_action=True* then this is a discrete action space (which
        subclasses gym.spaces.Discrete), so each action is represented by an
        integer.
        If *flat_action=False* then this is a parameterised action space (which
        subclasses gym.spaces.MultiDiscrete), so each action is represented
        using a list of parameters.
    observation_space : gym.spaces.Box
        observation space for environment.
        If *flat_obs=True* then observations are represented by a 1D vector,
        otherwise observations are represented as a 2D matrix.
    current_state : State
        the current state of the environment
    last_obs : Observation
        the last observation that was generated by environment
    """
    metadata = {'rendering.modes': ["readable", "ASCI"]}
    reward_range = (-float('inf'), float('inf'))

    action_space = None
    observation_space = None
    current_state = None
    last_obs = None

    def __init__(self,
                 scenario,
                 fully_obs=True,
                 flat_actions=True,
                 flat_obs=True):
        """
        Parameters
        ----------
        scenario : Scenario
            Scenario object, defining the properties of the environment
        fully_obs : bool, optional
            The observability mode of environment, if True then uses fully
            observable mode, otherwise is partially observable (default=True)
        flat_actions : bool, optional
            If true then uses a flat action space, otherwise will uses a
            parameterised action space (default=True).
        flat_obs : bool, optional
            If true then uses a 1D observation space, otherwise uses a 2D
            observation space (default=True)
        """
        self.scenario = scenario
        self.fully_obs = fully_obs
        self.flat_actions = flat_actions
        self.flat_obs = flat_obs

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
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs_shape)

    def reset(self):
        """Reset the state of the environment and returns the initial state.

        Implements gym.Env.reset().

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

        Implements gym.Env.step().

        Parameters
        ----------
        action : Action, int, list, NumpyArray
            Action to perform. If not Action object, then if using
            flat actions this should be an int and if using non-flat actions
            this should be an indexable array.

        Returns
        -------
        Observation
            observation from performing action
        float
            reward from performing action
        bool
            whether the episode has ended or not
        dict
            auxiliary information regarding step
            (see :func:`nasim.env.action.ActionResult.info`)
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
            whether the episode has ended or not
        dict
            auxiliary information regarding step
            (see :func:`nasim.env.action.ActionResult.info`)
        """
        if not isinstance(action, Action):
            action = self.action_space.get_action(action)

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

        Returns
        -------
        State
            The initial state

        Notes
        -----
        This does not reset the current state of the environment (use
        :func:`reset` for that).
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

    def get_minimum_actions(self):
        """Get the minimum number of actions required to reach the goal.

        That is minimum number of actions to exploit all sensitive hosts on
        the network starting from the initial state

        Returns
        -------
        int
            minumum possible actions to reach goal
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

    def get_score_upper_bound(self):
        """Get the theoretical upper bound for total reward for scenario.

        The theoretical upper bound score is where the agent exploits only a
        single host in each subnet that is required to reach sensitive hosts
        along the shortest bath in network graph, and exploits the two
        sensitive hosts (i.e. the minial steps). Assuming action cost of 1 and
        each sensitive host is exploitable from any other connected subnet
        (which may not be true, hence being an upper bound).

        Returns
        -------
        float
            theoretical max score
        """
        max_reward = self.network.get_total_sensitive_host_value()
        max_reward -= self.network.get_minimal_steps()
        return max_reward

    def is_goal(self, state):
        """Check if the current state is the goal state.

        The goal state is  when all sensitive hosts have been compromised.

        Parameters
        ----------
        state : State
            the current state

        Returns
        -------
        bool
            True if state is goal state, otherwise False.
        """
        return self.network.all_sensitive_hosts_compromised(state)
