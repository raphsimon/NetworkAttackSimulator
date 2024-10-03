import numpy as np

from nasim.envs.utils import AccessLevel
from nasim.envs.host_vector import HostVector


class Observation:
    """An observation for NASim.

    Each observation is a 2D tensor with a row for each host and an additional
    row containing auxiliary observations. Each host row is a host_vector (for
    details see :class:`HostVector`) while the auxiliary
    row contains non-host specific observations (see Notes section).

    ...

    Attributes
    ----------
    obs_shape : (int, int)
        the shape of the observation
    aux_row : int
        the row index for the auxiliary row
    tensor : numpy.ndarray
        2D Numpy array storing the observation

    Notes
    -----
    The auxiliary row is the final row in the observation tensor and has the
    following features (in order):

    1. Action success - True (1) or False (0)
        indicates whether the action succeeded or failed
    2. Connection error - True (1) or False (0)
        indicates whether there was a connection error or not
    3. Permission error - True (1) or False (0)
        indicates whether there was a permission error or not
    4. Undefined error - True (1) or False (0)
        indicates whether there was an undefined error or not (e.g. failure due
        to stochastic nature of exploits)

    Since the number of features in the auxiliary row is less than the number
    of features in each host row, the remainder of the row is all zeros.
    """

    # obs vector positions for auxiliary observations
    _success_idx = 0
    _conn_error_idx = _success_idx + 1
    _perm_error_idx = _conn_error_idx + 1
    _undef_error_idx = _perm_error_idx + 1

    def __init__(self, state_shape):
        """
        Parameters
        ----------
        state_shape : (int, int)
            2D shape of the state (i.e. num_hosts, host_vector_size)
        """
        self.obs_shape = (state_shape[0]+1, state_shape[1])
        self.aux_row = self.obs_shape[0]-1
        self.tensor = np.zeros(self.obs_shape, dtype=np.float32)

    @staticmethod
    def get_space_bounds(scenario):
        # We removed the value as being part of the bounds since it is not
        # part of the state or observations anymore. It has been replaced
        # with the boolean 'sensitive' property. Thus, it won't matter for
        # the space bounds.
        discovery_bounds = scenario.host_discovery_value_bounds
        obs_low = min(
            0,
            discovery_bounds[0]
        )
        obs_high = max(
            1,
            discovery_bounds[1],
            AccessLevel.ROOT,
            scenario.address_space_bounds[0],
            scenario.address_space_bounds[1]
        )
        return (obs_low, obs_high)

    @classmethod
    def from_numpy(cls, o_array, state_shape):
        obs = cls(state_shape)
        if o_array.shape != (state_shape[0]+1, state_shape[1]):
            o_array = o_array.reshape(state_shape[0]+1, state_shape[1])
        obs.tensor = o_array
        return obs

    def from_state(self, state):
        # Keep auxiliary row and replace everything above.
        self.tensor[:self.aux_row] = state.tensor

    def from_action_result(self, action_result):
        success = int(action_result.success)
        self.tensor[self.aux_row][self._success_idx] = success
        con_err = int(action_result.connection_error)
        self.tensor[self.aux_row][self._conn_error_idx] = con_err
        perm_err = int(action_result.permission_error)
        self.tensor[self.aux_row][self._perm_error_idx] = perm_err
        undef_err = int(action_result.undefined_error)
        self.tensor[self.aux_row][self._undef_error_idx] = undef_err

    def from_state_and_action(self, state, action_result):
        self.from_state(state)
        self.from_action_result(action_result)

    def update_from_host(self, host_idx, host_obs_vector):
        self.tensor[host_idx][:] = host_obs_vector

    @property
    def success(self):
        """Whether the action succeded or not

        Returns
        -------
        bool
            True if the action succeeded, otherwise False
        """
        return bool(self.tensor[self.aux_row][self._success_idx])

    @property
    def connection_error(self):
        """Whether there was a connection error or not

        Returns
        -------
        bool
            True if there was a connection error, otherwise False
        """
        return bool(self.tensor[self.aux_row][self._conn_error_idx])

    @property
    def permission_error(self):
        """Whether there was a permission error or not

        Returns
        -------
        bool
            True if there was a permission error, otherwise False
        """
        return bool(self.tensor[self.aux_row][self._perm_error_idx])

    @property
    def undefined_error(self):
        """Whether there was an undefined error or not

        Returns
        -------
        bool
            True if there was a undefined error, otherwise False
        """
        return bool(self.tensor[self.aux_row][self._undef_error_idx])

    def shape_flat(self):
        """Get the flat (1D) shape of the Observation.

        Returns
        -------
        (int, )
            the flattened shape of observation
        """
        return self.numpy_flat().shape

    def shape(self):
        """Get the (2D) shape of the observation

        Returns
        -------
        (int, int)
            the 2D shape of the observation
        """
        return self.obs_shape

    def numpy_flat(self):
        """Get the flattened observation tensor

        Returns
        -------
        numpy.ndarray
            the flattened (1D) observation tenser
        """
        return self.tensor.flatten()

    def numpy(self):
        """Get the observation tensor

        Returns
        -------
        numpy.ndarray
            the (2D) observation tenser
        """
        return self.tensor

    def get_readable(self):
        """Get a human readable version of the observation

        Returns
        -------
        list[dict]
            list of host observations as human-readable dictionary
        dict[str, bool]
            auxiliary observation dictionary
        """
        host_obs = []
        for host_idx in range(self.obs_shape[0]-1):
            host_obs_vec = self.tensor[host_idx]
            readable_dict = HostVector.get_readable(host_obs_vec)
            host_obs.append(readable_dict)

        aux_obs = {
            "Success": self.success,
            "Connection Error": self.connection_error,
            "Permission Error": self.permission_error,
            "Undefined Error": self.undefined_error
        }
        return host_obs, aux_obs

    def __str__(self):
        return str(self.tensor)

    def __eq__(self, other):
        return np.array_equal(self.tensor, other.tensor)

    def __hash__(self):
        return hash(str(self.tensor))


class ObservationWithActionInfo:
    """An observation for NASim.

    Each observation is a 2D tensor with a row for each host, and some additional
    information. Each host row is a host_vector (for details see :class:`HostVector`)
    and some additional information, such as the one-hot encoded action that was
    executed on that host, and the result signal of that action.

    This type of observation is only available in the partially observable setting,
    as the agent requires some more information to solve the environment, due ti it's
    complexity.

    ...

    Attributes
    ----------
    obs_shape : (int, int)
        The shape of the observation.
    num_actions: int
        the number of possible actions per host.
    tensor : numpy.ndarray
        2D Numpy array storing the observation.

    Notes
    -----
    The auxiliary row is the final row in the observation tensor and has the
    following features (in order):

    1. Action success - True (1) or False (0)
        indicates whether the action succeeded or failed
    2. Connection error - True (1) or False (0)
        indicates whether there was a connection error or not
    3. Permission error - True (1) or False (0)
        indicates whether there was a permission error or not
    4. Undefined error - True (1) or False (0)
        indicates whether there was an undefined error or not (e.g. failure due
        to stochastic nature of exploits)

    Since the number of features in the auxiliary row is less than the number
    of features in each host row, the remainder of the row is all zeros.
    """

    def __init__(self, state_shape, num_actions):
        """
        Parameters
        ----------
        state_shape : (int, int)
            2D shape of the state (i.e. num_hosts, host_vector_size)
        """
        #                                          4 action different results
        self.obs_shape = (state_shape[0], state_shape[1] + num_actions + 4)
        # obs vector positions for auxiliary observations
        self._action_idx = state_shape[1]
        self._success_idx = state_shape[1] + num_actions
        self._conn_error_idx = self._success_idx + 1
        self._perm_error_idx = self._conn_error_idx + 1
        self._undef_error_idx = self._perm_error_idx + 1
        self.tensor = np.zeros(self.obs_shape, dtype=np.float32)

    @staticmethod
    def get_space_bounds(scenario):
        # We removed the value as being part of the bounds since it is not
        # part of the state or observations anymore. It has been replaced
        # with the boolean 'sensitive' property. Thus, it won't matter for
        # the space bounds.
        discovery_bounds = scenario.host_discovery_value_bounds
        obs_low = min(
            0,
            discovery_bounds[0]
        )
        obs_high = max(
            1,
            discovery_bounds[1],
            AccessLevel.ROOT,
            scenario.address_space_bounds[0],
            scenario.address_space_bounds[1]
        )
        return (obs_low, obs_high)


    @classmethod
    def from_numpy(cls, o_array, state_shape, num_actions):
        # TODO If I look at the references, this method is not used anywhere.
        # Nevertheless, I'm not convinced that the code is right. We don't do
        # checks on the given o_array and it could be that it can't be resphaped
        # into the observation shape. Maybe use an assert?
        obs = cls(state_shape, num_actions)
        if o_array.shape != (state_shape[0], state_shape[1] + num_actions + 4):
            o_array = o_array.reshape(state_shape[0], 
                                      state_shape[1]+ num_actions + 4)
        obs.tensor = o_array
        return obs

    def from_state(self, state):
        assert len(state) == len(self.tensor), \
            "State and Observation should have same amount of rows"
        assert len(state[0]) == self._action_idx, \
            f"State has wrong dimensions compared to Observation dimensions.\n{len(state[0])=}\n{self._action_idx=}"

        for host in state:
            self.tensor[host][:self._action_idx] = state.tensor[host]

    def from_action_result(self, host, action_result):
        """
        We set the action result for the host given as argument.
        """
        success = int(action_result.success)
        self.tensor[host][self._success_idx] = success
        con_err = int(action_result.connection_error)
        self.tensor[host][self._conn_error_idx] = con_err
        perm_err = int(action_result.permission_error)
        self.tensor[host][self._perm_error_idx] = perm_err
        undef_err = int(action_result.undefined_error)
        self.tensor[host][self._undef_error_idx] = undef_err

    def from_performed_action(self, host, action):
        self.tensor[host][self._action_idx + action] = 1

    def from_state_and_action(self, state, action_result):
        self.from_state(state)
        self.from_action_result(action_result)

    def update_from_host(self, host_idx, host_obs_vector):
        self.tensor[host_idx][:self._action_idx] = host_obs_vector

    @property
    def success(self):
        """Whether the action succeded or not

        Returns
        -------
        bool
            True if the action succeeded, otherwise False
        """
        res = np.sum(self.tensor, axis=0)[self._success_idx]
        assert res > 1, f"Observation is compromised. Summing colmuns retuned > 1 for 'success'\n{self.tensor}"
        return bool(res)  

    @property
    def connection_error(self):
        """Whether there was a connection error or not

        Returns
        -------
        bool
            True if there was a connection error, otherwise False
        """
        res = np.sum(self.tensor, axis=0)[self._conn_error_idx]
        assert res > 1, f"Observation is compromised. Summing colmuns retuned > 1 for 'connection_error'\n{self.tensor}"
        return bool(res)     

    @property
    def permission_error(self):
        """Whether there was a permission error or not

        Returns
        -------
        bool
            True if there was a permission error, otherwise False
        """
        res = np.sum(self.tensor, axis=0)[self._perm_error_idx]
        assert res > 1, f"Observation is compromised. Summing colmuns retuned > 1 for 'permission_error'\n{self.tensor}"
        return bool(res)

    @property
    def undefined_error(self):
        """Whether there was an undefined error or not

        Returns
        -------
        bool
            True if there was a undefined error, otherwise False
        """
        res = np.sum(self.tensor, axis=0)[self._undef_error_idx]
        assert res > 1, f"Observation is compromised. Summing colmuns retuned > 1 for 'undefined_error'\n{self.tensor}"
        return bool(res)

    def shape_flat(self):
        """Get the flat (1D) shape of the Observation.

        Returns
        -------
        (int, )
            the flattened shape of observation
        """
        return self.numpy_flat().shape

    def shape(self):
        """Get the (2D) shape of the observation

        Returns
        -------
        (int, int)
            the 2D shape of the observation
        """
        return self.obs_shape

    def numpy_flat(self):
        """Get the flattened observation tensor

        Returns
        -------
        numpy.ndarray
            the flattened (1D) observation tenser
        """
        return self.tensor.flatten()

    def numpy(self):
        """Get the observation tensor

        Returns
        -------
        numpy.ndarray
            the (2D) observation tenser
        """
        return self.tensor

    def get_readable(self):
        """Get a human readable version of the observation

        Returns
        -------
        list[dict]
            list of host observations as human-readable dictionary
        dict[str, bool]
            auxiliary observation dictionary
        """
        host_obs = []
        for host_idx in range(self.obs_shape[0]):
            host_obs_vec = self.tensor[host_idx]
            readable_dict = HostVector.get_readable(host_obs_vec)
            host_obs.append(readable_dict)

        aux_obs = {
            "Success": self.success,
            "Connection Error": self.connection_error,
            "Permission Error": self.permission_error,
            "Undefined Error": self.undefined_error
        }
        return host_obs, aux_obs

    def __str__(self):
        return str(self.tensor)

    def __eq__(self, other):
        return np.array_equal(self.tensor, other.tensor)

    def __hash__(self):
        return hash(str(self.tensor))
