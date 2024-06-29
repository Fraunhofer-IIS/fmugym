from abc import ABC, abstractmethod

import shutil

from fmpy import read_model_description, extract
from fmpy.fmi1 import FMU1Slave
from fmpy.fmi2 import FMU2Slave
from fmpy.fmi3 import FMU3Slave

import gymnasium as gym

import numpy as np


class FMUGym(ABC, gym.Env):
    """
    A gym environment for interacting with an FMU (Functional Mock-up Unit).

    This class provides an interface for interacting with an FMU using the OpenAI Gym framework.
    It allows users to reset the environment, take steps, and obtain observations, rewards, and termination information.

    Attributes:
        start_time (float): The start time of the simulation.
        time (float): The current time of the simulation.
        stop_time (float): The stop time of the simulation.
        dt_sim (float): The simulation step size.
        dt_action (float): The action step size.
        y_start (dict): The initial set point values.
        y_stop_range (dict): The range of set point values.
        y_stop (dict): The target set point values.
        rand_starts (dict): The randomized set point values.
        terminations (dict): The termination conditions.
        fmu_description (object): The model description of the FMU.
        input_dict (dict): A dictionary mapping input variable names to their value references.
        input_noise (dict): A dictionary mapping input variable names to their noise limits.
        output_dict (dict): A dictionary mapping output variable names to their value references.
        output_noise (dict): A dictionary mapping output variable names to their noise limits.
        random_vars_refs (dict): A dictionary mapping randomized variable names to their value references and configurations.
        action_space (gym.Space): The action space of the environment.
        observation_space (gym.Space): The observation space of the environment.
        observation (dict): The current observation of the environment.
        fmu_path (str): The path to the FMU file.
        unzipdir (str): The directory where the FMU is extracted.
        is_fmi1 (bool): Indicates whether the FMU is FMI 1.0.
        is_fmi2 (bool): Indicates whether the FMU is FMI 2.0.
        is_fmi3 (bool): Indicates whether the FMU is FMI 3.0.
        FMUxSlave (class): The FMU slave class based on the FMI version.
        fmu (object): The FMU instance.
    """

    def __init__(self, config):
        """
        Initializes the FMUGym class from the fmugym_config.py file.

        Parameters:
            config (object): The configuration object containing the necessary parameters utilizing fmugym_config.py for the FMUGym instance.
        """

        self.start_time = config.start_time
        self.time = config.start_time
        self.stop_time = config.stop_time
        self.dt_sim = config.sim_step_size
        self.dt_action = config.action_step_size

        self.y_start = config.set_point_nominal_start.variables

        self.y_stop_range = config.set_point_stop.variables
        self.y_stop = {}
        # initialize y_stop target values with mean of set point range
        for ye in self.y_stop_range:
            self.y_stop[ye] = (
                self.y_stop_range[ye].high[0] - self.y_stop_range[ye].low[0]
            ) / 2.0

        self.rand_starts = config.set_point_map.variables
        self.terminations = config.terminations.variables

        self.fmu_description = read_model_description(config.fmu_path)

        # strongly related to https://github.com/CATIA-Systems/FMPy/blob/main/fmpy/examples/custom_input.py
        # collect the value references
        self.input_dict = {}  # for convenience to access inputs
        self.input_noise = {}  # to access limits of input noise
        self.output_dict = {}  # for convenience to access outputs
        self.output_noise = {}  # to access limits of output noise
        self.random_vars_refs = (
            {}
        )  # for convenience to access the randomized variables (domain or dynamics randomization)

        for variable in self.fmu_description.modelVariables:
            if variable.name in config.inputs.variables.keys():
                self.input_dict[variable.name] = variable.valueReference
                self.input_noise[variable.name] = config.input_noise.variables[
                    variable.name
                ]
            if variable.name in config.outputs.variables.keys():
                self.output_dict[variable.name] = variable.valueReference
                self.output_noise[variable.name] = config.output_noise.variables[
                    variable.name
                ]
            if variable.name in config.random_vars.variables.keys():
                self.random_vars_refs[variable.name] = [
                    variable.valueReference,
                    config.random_vars.variables[variable.name],
                ]

        # create action space
        self.action_space = self._create_action_space(config.inputs.variables)

        # create observation_space
        self.observation_space = self._create_observation_space(
            config.outputs.variables
        )

        # initialize observation
        self.observation = {}
        for sensor in self.output_dict:
            self.observation[sensor] = np.array([0], dtype=np.float32)

        # extract the FMU
        self.fmu_path = config.fmu_path
        self.unzipdir = extract(self.fmu_path)

        self.is_fmi1 = False
        self.is_fmi2 = False
        self.is_fmi3 = False
        if self.fmu_description.fmiVersion == "1.0":
            self.FMUxSlave = FMU1Slave
            self.is_fmi1 = True
        elif self.fmu_description.fmiVersion == "2.0":
            self.FMUxSlave = FMU2Slave
            self.is_fmi2 = True
        elif self.fmu_description.fmiVersion.startswith("3.0"):
            self.FMUxSlave = FMU3Slave
            self.is_fmi3 = True
        else:
            raise NotImplementedError(
                "We only support FMI 1.0, 2.0 and 3.0 at the moment."
            )

        self.fmu = self.FMUxSlave(
            guid=self.fmu_description.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.fmu_description.coSimulation.modelIdentifier,
            instanceName="instance1",
        )

        # initialize
        init_states = self._noisy_init()
        if self.is_fmi1:
            self.fmu.instantiate()
            self.fmu.setReal(
                init_states.keys(), init_states.values()
            )  # set randomized variables
            self.fmu.initialize(
                tStart=self.start_time, stopTime=self.stop_time + self.dt_sim
            )
        elif self.is_fmi2:
            self.fmu.instantiate()
            self.fmu.setupExperiment(startTime=self.start_time)
            self.fmu.enterInitializationMode()
            self.fmu.setReal(
                init_states.keys(), init_states.values()
            )  # set randomized variables
            self.fmu.exitInitializationMode()
        elif self.is_fmi3:
            self.fmu.instantiate()
            self.fmu.enterInitializationMode()
            self.fmu.setFloat64(
                init_states.keys(), init_states.values()
            )  # set randomized variables
            self.fmu.exitInitializationMode()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Parameters:
            seed (int): The random seed for the environment. (default: None)
            options (dict): Additional options for resetting the environment. (default: None)

        Returns:
            observation (object): The initial observation of the environment.
            info: Additional information about the environment's state.
        """

        # seed self.np_random
        super().reset(seed=seed, options=options)
        self.time = self.start_time

        ######################################################################################################################################
        ### this has to be done to properly reset the FMU (instead of self.fmu.reset()) without running into memory problems until the OMEdit FMU export is fixed:
        #   https://github.com/OpenModelica/OpenModelica/issues/11506
        #
        self.close()

        self.unzipdir = extract(self.fmu_path)
        self.fmu = self.FMUxSlave(
            guid=self.fmu_description.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.fmu_description.coSimulation.modelIdentifier,
            instanceName="instance1",
        )
        ######################################################################################################################################

        init_states = self._noisy_init()
        if self.is_fmi1:
            self.fmu.instantiate()
            self.fmu.setReal(
                init_states.keys(), init_states.values()
            )  # set randomized variables
            self.fmu.initialize(
                tStart=self.start_time, stopTime=self.stop_time + self.dt_sim
            )
        elif self.is_fmi2:
            self.fmu.instantiate()
            self.fmu.setupExperiment(startTime=self.start_time)
            self.fmu.enterInitializationMode()
            self.fmu.setReal(
                init_states.keys(), init_states.values()
            )  # set randomized variables
            self.fmu.exitInitializationMode()
        elif self.is_fmi3:
            self.fmu.instantiate()
            self.fmu.enterInitializationMode()
            self.fmu.setFloat64(
                init_states.keys(), init_states.values()
            )  # set randomized variables
            self.fmu.exitInitializationMode()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """
        Executes a single step in the environment.

        Parameters:
            action: The action to take in the environment, typically provided by the connected RL library.

        Returns:
            obs: The observations after taking the step.
            reward: The reward obtained from the step.
            terminated (bool): Whether the episode is terminated after this step.
            constraint_limit (bool): Whether constraints limiting the state space are reached.
            info: Additional information about the step.

        """

        terminated, constraint_limit = self._get_terminated()

        self.current_action = self._process_action(action)

        if self.is_fmi3:
            self.fmu.setFloat64(self.input_dict.values(), self.current_action)
        else:
            self.fmu.setReal(self.input_dict.values(), self.current_action)

        init_time = self.time
        while self.time < init_time + self.dt_action:
            self.fmu.doStep(
                currentCommunicationPoint=self.time, communicationStepSize=self.dt_sim
            )
            self.time = self.time + self.dt_sim

        info = self._get_info()
        obs = self._get_obs()

        reward = self._process_reward(obs, self.current_action, info)

        return obs, reward, terminated, constraint_limit, info

    def close(self):
        """
        Closes the FMU instance and cleans up the temporary files.

        This method terminates the FMU instance, frees the instance resources,
        and removes the temporary directory where the FMU was extracted.

        """
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self.unzipdir, ignore_errors=True)

    def _get_fmu_output(self):
        """
        Retrieves the output values from the FMU and stores them in the observation dictionary.

        Returns:
            observation (dict): A dictionary containing the output values from the FMU.
        """
        for out_name in self.output_dict:
            if self.is_fmi3:
                value = self.fmu.getFloat64([self.output_dict[out_name]])[0]
            else:
                value = self.fmu.getReal([self.output_dict[out_name]])[0]
            self.observation[out_name] = np.array(value, dtype=np.float32).flatten()
        return self.observation

    @abstractmethod
    def _get_info(self):
        """
        Used by step() and reset(), returns any relevant debugging information.

        Returns:
            The relevant debugging information.
        """
        pass

    @abstractmethod
    def _get_obs(self):
        """
        Retrieves FMU output values by possibly calling self.fmu._get_fmu_output for handling different FMU versions. This call stores data in the self.observation dictionary. It may also add output noise (using self._get_output_noise()) and update the set point (using self.setpoint_trajectory()) to return a goal-oriented observation dictionary.

        Returns:
            observations, e.g. as dict or array
        """
        pass

    @abstractmethod
    def _get_input_noise(self):
        """
        Returns input noise for each input component, potentially by sampling from the self.input_noise dictionary.

        Returns:
            noise values in shape of inputs to be added
        """
        pass

    @abstractmethod
    def _get_output_noise(self):
        """
        Returns output noise for each output component, potentially by sampling from the self.output_noise dictionary.

        Returns:
            noise values in shape of outputs to be added
        """
        pass

    @abstractmethod
    def _get_terminated(self):
        """
        Returns two booleans indicating first the termination and second truncation status.

        A tuple containing the following elements:
            termination (bool): episode ending after finite time horizon is exceeded.
            truncation (bool): episode ending after an externally defined condition (constraint limit exceeded), thereby interrupting the MDP.
        """
        pass

    @abstractmethod
    def _create_action_space(self, inputs):
        """
        Constructs the action space from a VarSpace object representing the inputs. It can use gymnasium.spaces.Box for continuous action spaces or gymnasium.spaces.Discrete for discrete action spaces.

        Parameters:
            inputs (VarSpace): The inputs used to create the action space.

        Returns:
            action_space: The action space constructed for the inputs.
        """
        pass

    @abstractmethod
    def _create_observation_space(self, outputs):
        """
        Constructs the observation space returning it possibly as a gymnasium.spaces.Dict for a goal oriented structure. The observation space typically includes observation, achieved_goal, and desired_goal, created from a VarSpace object.

        Parameters:
            outputs (VarSpace): The outputs used to create the observation space.

        Returns:
            observation_space: The observation space constructed for the outputs.
        """
        pass

    @abstractmethod
    def _noisy_init(self):
        """
        Random variations to initial system states and dynamic parameters by sampling from self.random_vars_refs and propagates to corresponding initial output values. It also allows for direct manipulation and randomization of set point goals using the self.y_stop class variable.

        Returns:
            init_states (VarSpace): The initial state values with noise added.
        """
        pass

    @abstractmethod
    def _process_action(self, action):
        """
        Called by self.step() to add noise to action from RL library. May be used to execute low-level controller and adapt action space.

        Parameters:
            action: The action to be processed.

        Returns:
            processed_action: The processed action.

        """
        pass

    @abstractmethod
    def setpoint_trajectory(self):
        """
        Determines the set point values at the current time step within the trajectory, called by self._get_obs().

        Returns:
            setpoint: The set point value for each output at the current time step.
        """
        pass

    @abstractmethod
    def _process_reward(self):
        """
        Preprocesses the reward to adjust for predefined interfaces of compute_reward expected by e.g. StableBaselines 3 and then computes reward by calling compute_reward() for the current time step.
        
        Returns:
            processed_reward: The processed reward value.
        """
        pass

    @abstractmethod
    def compute_reward(self, *args, **kwargs):
        """
        Computes and returns reward with interface compatible with RL library.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The computed reward value.

        """
        pass
