class RewardFunction:
    """This class determines the reward that the agent receives for executing
    action a in state s.
    In our particular environment, the reward detends on the following aspects:
    - Every step the agent gets a penalty of -1
    - 
    """
    def __init__(self, sensitive_host_val) -> None:
        self.sensitive_host_value = sensitive_host_val