import numpy as np

import gymnasium as gym
from gymnasium import spaces


class VarSpace:
    """
    A class for discrete or continuous variable spaces of inputs, outputs, their noises or uncertain dynamic parameters.

    Attributes:
        name (str): The name of the variable space.
        variables (dict): A dictionary to hold the variables.

    Methods:
        __init__(self, name: str): Initializes a new instance of the VarSpace class.
        __setitem__(self, key, value): Sets a variable as class variable.
        add_var_box(self, name: str, min: float, max: float): Adds a box variable to the variable space.
        add_var_discrete(self, name: str, n: int, start: int): Adds a discrete variable to the variable space.
    """

    def __init__(self, name: str):
        """
        Initializes a  VarSpace class.

        Parameters:
            name (str): The name of the variable space.
        """
        self.name = name
        self.variables = {}  # Create a dictionary to hold the variables

    def __setitem__(self, key, value):
        """
        Sets a variable as class variable.

        Parameters:
        - key: The identifier to set the value for.
        - value: The data to assign to the key.
        """
        self.variables[key] = value  # Assign the value to the key in the dictionary

    def add_var_box(self, name: str, min: float, max: float):
        """
        Adds a Box space to the variables dictionary with the key as the name and the value as a Box space.

        Parameters:
            name (str): The key under which the Box space will be stored.
            min (float): The lower bound of the Box space.
            max (float): The upper bound of the Box space.
        """
        self[name] = spaces.Box(low=min, high=max)

    def add_var_discrete(self, name: str, n: int, start: int):
        """
        Adds a Discrete space to the variables dictionary with the key as the name and the value as a Discrete space.

        Parameters:
            name (str): The key under which the Discrete space will be stored.
            n (int):The number of possible values in the Discrete space.
            start (int): The starting value of the Discrete space.
        """
        self[name] = spaces.Discrete(n, start=start)


class State2Out:
    """
    Represents a class that maps states to outputs.

    Attributes:
        name (str): The name of the configuration.
        variables (dict): A dictionary to hold the variables.
    """

    def __init__(self, name: str):
        """
        Initializes a new instance of the State2Out class.

        Parameters:
            name (str): The name of the configuration.
        """
        self.name = name
        self.variables = {}  # Create a dictionary to hold the variables

    def __setitem__(self, key, value):
        """
        Sets a variable as class variable.

        Parameters:
            key: The identifier to set the value for.
            value: The data to assign to the key.

        Returns:
            None
        """
        self.variables[key] = value  # Assign the value to the key in the dictionary

    def add_map(self, state: str, out: str):
        """
        Adds a mapping between a state and an output.

        Parameters:
            state (str): The FMU state name to map.
            out (str): The corresponding output name connected to the state.

        Returns:
            None
        """
        self[state] = out


class TargetValue:
    """
    Represents a configuration for target values used for starting conditions of system outputs.

    Attributes:
        name (str): The name of the configuration.
        variables (dict): A dictionary to hold the variables.

    Methods:
        __init__(self, name: str): Initialize a new instance of the TargetValue class.
        __setitem__(self, key, value): Sets a variable as class variable.
        add_target(self, name: str, value: float): Adds a target to the configuration.
    """

    def __init__(self, name: str):
        """
        Initialize a new instance of the TargetValue class.

        Parameters:
            name (str): The name of the configuration.

        Returns:
            None
        """
        self.name = name
        self.variables = {}  # Create a dictionary to hold the variables

    def __setitem__(self, key, value):
        """
        Sets a variable as class variable.

        Parameters:
            key: The identifier to set the value for.
            value: The data to assign to the key.

        Returns:
        None
        """

        self.variables[key] = value  # Assign the value to the key in the dictionary

    def add_target(self, name: str, value: float):
        """
        Adds a target value for an output to the configuration.

        Parameters:
            name (str): The FMU output name of the target.
            value (float): The target value.

        Returns:
            None
        """
        self[name] = value


class FMUGymConfig:
    """
    Configuration class for the FMUGym instance.

    This class represents the configuration for FMUGym, which is used to specify various parameters
    for the simulation and training environment.
    """

    def __init__(
        self,
        fmu_path: str,
        start_time: float = 0.0,
        stop_time: float = 10.0,
        sim_step_size: float = 0.01,
        action_step_size: float = 0.01,
        inputs: VarSpace | None = None,
        input_noise: VarSpace | None = None,
        outputs: VarSpace | None = None,
        output_noise: VarSpace | None = None,
        random_vars: VarSpace | None = None,
        set_point_map: State2Out | None = None,
        set_point_nominal_start: TargetValue | None = None,
        set_point_stop: VarSpace | None = None,
        terminations: VarSpace | None = None,
    ):
        """
        Initialize the FMUGymConfig object.

        Parameters:
            fmu_path (str): The path to the FMU file.
            start_time (float, optional): The start time of the simulation. Defaults to 0.0.
            stop_time (float, optional): The stop time of the simulation. Defaults to 10.0.
            sim_step_size (float, optional): The simulation step size. Defaults to 0.01.
            action_step_size (float, optional): The action step size. Defaults to 0.01.
            inputs (VarSpace|None, optional): The input variables. Defaults to None.
            input_noise (VarSpace|None, optional): The input noise variables. Defaults to None.
            outputs (VarSpace|None, optional): The output variables. Defaults to None.
            output_noise (VarSpace|None, optional): The output noise variables. Defaults to None.
            random_vars (VarSpace|None, optional): The random variables. Defaults to None.
            set_point_map (State2Out|None, optional): The set point map. Defaults to None.
            set_point_nominal_start (TargetValue|None, optional): The set point nominal start. Defaults to None.
            set_point_stop (VarSpace|None, optional): The set point stop variables. Defaults to None.
            terminations (VarSpace|None, optional): The termination variables. Defaults to None.
        """
        # add all the parameters as class variables
        vars(self).update(
            {key: value for key, value in locals().items() if key != "self"}
        )
