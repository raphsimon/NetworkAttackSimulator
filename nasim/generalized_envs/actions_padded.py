import numpy as np
from gymnasium import spaces
from nasim.envs.action import load_action_list, NoOp


class FlatActionSpacePadded(spaces.Discrete):
    """Flat Action and Padded space for StochNASim environment.

    Inherits and implements the gym.spaces.Discrete action space

    ...

    Attributes
    ----------
    n : int
        the number of actions in the action space
    actions : list of Actions
        the list of the Actions in the action space
    """

    def __init__(self, scenario, pad_to_length=None):
        """
        Parameters
        ---------
        scenario : Scenario
            scenario description
        """
        self.actions = load_action_list(scenario)
        
        if pad_to_length is not None:
            num_missing_actions = pad_to_length - len(self.actions)
            if num_missing_actions > 0:
                self.actions.extend([NoOp(cost=1.0)] * num_missing_actions)
        
        super().__init__(len(self.actions))

    def get_action(self, action_idx):
        """Get Action object corresponding to action idx

        Parameters
        ----------
        action_idx : int
            the action idx

        Returns
        -------
        Action
            Corresponding Action object
        """
        assert isinstance(action_idx, (int, np.integer)), \
            ("When using flat action space, action must be an integer"
             f" or an Action object. {type(action_idx)} is invalid")
        return self.actions[action_idx]