import os
import time
from collections import OrderedDict

import numpy as np

from fmugym import FMUGym, FMUGymConfig, VarSpace, State2Out, TargetValue

import gymnasium as gym

class FMUEnv(FMUGym):
    def __init__(self, config):
        super().__init__(config)

    # Used by step() and reset(), returns any relevant debugging information.
    def _get_info(self):
        return {'info_time':time.time()}

    # Retrieves FMU output values by possibly calling self.fmu_get_fmu_output for handling different FMU versions
    # and stores them in the self.observation dictionary. 
    # It can also add output noise (using self._get_output_noise()) and update the set point 
    def _get_obs(self):
        
        self._get_fmu_output()

        obs = np.array(list(self.observation.values())).flatten()
        
        noisy_observation = obs + self._get_output_noise()
        
        setpoint = self.setpoint_trajectory(self.y_start, self.y_stop, self.time)

        obs_dict = OrderedDict([
            ('observation', np.array(noisy_observation)),
            ('achieved_goal', np.array(noisy_observation)),
            ('desired_goal', setpoint.astype(np.float32))
        ])
        return obs_dict

    # Returns input noise for each input component, potentially by sampling from the self.input_noise dictionary.
    def _get_input_noise(self):
        input_noise = []
        for inp_name in self.input_dict:
            input_noise.append(self.input_noise[inp_name].sample()[0])
        return np.array(input_noise)

    # Similar to self._get_input_noise, generates output noise for each output component,
    # potentially by sampling from the self.output_noise dictionary.
    def _get_output_noise(self):
        output_noise = []
        for out_name in self.output_dict:
            output_noise.append(self.output_noise[out_name].sample()[0])
        return np.array(output_noise)

    # Returns two booleans indicating first the termination and second truncation status. 
    def _get_terminated(self):
        if self.time > self.stop_time:
                self.reset()
                return True, False
    
        for termination in self.terminations:
            min_value = self.terminations[termination].low[0]
            max_value = self.terminations[termination].high[0]
            if self.observation[termination] < min_value or self.observation[termination] > max_value:
                self.reset()
                print("truncated")
                return False, True
                    
        return False, False

    # Constructs the action space from a VarSpace object representing the inputs. 
    # It can use gymnasium.spaces.Box for continuous action spaces.
    def _create_action_space(self, inputs):
        lows = []
        highs = []
        for inp in inputs:
            lows.append(inputs[inp].low[0])
            highs.append(inputs[inp].high[0])
        action_space = gym.spaces.Box(low=np.array(lows), high=np.array(highs), dtype=np.float32)
        return action_space

    # Constructs the observation space returning it as a gymnasium.spaces.Dict. 
    # The observation space typically includes observation, achieved_goal, and desired_goal, each created from a VarSpace object.
    def _create_observation_space(self, outputs):
        lows = []
        highs = []
        for out in outputs:
            lows.append(outputs[out].low[0])
            highs.append(outputs[out].high[0])
        observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(low=np.array(lows), high=np.array(highs), dtype=np.float32),
            'achieved_goal': gym.spaces.Box(low=np.array(lows), high=np.array(highs), dtype=np.float32),
            'desired_goal': gym.spaces.Box(low=np.array(lows), high=np.array(highs), dtype=np.float32)
        })
        return observation_space

    # Random variations to initial system states and dynamic parameters by sampling from self.random_vars_refs 
    # and propagates to corresponding initial output values.  
    # It also allows for direct manipulation and randomization of set point goals using the self.y_stop class variable.
    def _noisy_init(self):
        # add noise to setpoint goals
        for ye in self.y_stop:
            self.y_stop[ye] = self.y_stop_range[ye].sample()[0]
        
        # add noise to initial system state
        init_states = {}
        for var in self.random_vars_refs:
            var_ref = self.random_vars_refs[var][0]
            uniform_value = self.random_vars_refs[var][1].sample()[0]
            init_states[var_ref] = uniform_value

            # domain randomization with noisy initial y_start
            if var in self.rand_starts.keys():
                input_string = self.rand_starts[var]
                self.y_start[input_string] = float(uniform_value)
        
        return init_states

    # Called by self.step() to add noise to action from RL library.
    # May be used to execute low-level controller and adapt action space.
    def _process_action(self, action):
        processed_action = action + self._get_input_noise()
        return processed_action

    # Determines the set point values at the current time step within the trajectory, called by self._get_obs().
    def setpoint_trajectory(self, y_start, y_stop, time):
        y = []
        for y0, ye in zip(y_start.values(), y_stop.values()):
            y.append((ye - y0)/(self.stop_time-self.start_time)*(time-self.start_time) + y0)
        return np.array(y)

    # Interface between step method of fmugym and compute_reward. adjusts the necessary paramters for reward method of RL library (SKRL, SB3).
    def _process_reward(self, obs, acts, info):
        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]
        reward = self.compute_reward(achieved_goal, desired_goal, info)
        return reward

    # Computes and returns a scalar reward value from achieved_goal, desired_goal, and possibly further parameters.
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        compute reward for HER
            achieved_goal: outputs of FMU
            desired_goal: current setpoint of trajectory
            info: [NOT USED]
        Returns:
            float: environment reward
        """

        # Deceptive reward: it is positive (0) only when the goal is achieved
        # Here we are using a vectorized version
        boundary_accuracy = 0.1
        control_error = achieved_goal - desired_goal
        if len(control_error) == self.observation_space["observation"].shape[0]:
                boundary_violated = False
                for err in control_error:
                    boundary_violated = boundary_violated or (abs(err)>boundary_accuracy)            
                reward = -(boundary_violated.astype(np.float32))
                
        else:
            reward = []
            for ctrl_err in control_error:
                boundary_violated = False
                for err in ctrl_err:
                    boundary_violated = boundary_violated or (abs(err)>boundary_accuracy)
                reward.append(-(boundary_violated.astype(np.float32)))
        
        return np.array(reward)
    
# providing inputs, outputs and their noises with range of values
inputs = VarSpace("inputs")
inputs.add_var_box("input1", -2.0, 2.0)
inputs.add_var_box("input2", -2.0, 2.0)

input_noise = VarSpace("input_noise")
input_noise.add_var_box("input1", 0.0, 0.0)
input_noise.add_var_box("input2", 0.0, 0.0)

outputs = VarSpace("outputs")
outputs.add_var_box("output1", -1e6, 1e6)
outputs.add_var_box("output2", -1e6, 1e6)

output_noise = VarSpace("output_noise")
output_noise.add_var_box("output1", 0.0, 0.0)
output_noise.add_var_box("output2", 0.0, 0.0)

# randomized dynamics parameters and their range of values
random_vars = VarSpace("random_vars")
random_vars.add_var_box("firstOrder.k", 4, 6)
random_vars.add_var_box("firstOrder1.k", 4, 6)
# randomized initial initial states and their range of values
random_vars.add_var_box("firstOrder.y_start", -0.5, 0.5)
random_vars.add_var_box("firstOrder1.y_start", -0.5, 0.5)

# map state variables to corresponding outputs of Modelica Model
set_point_map = State2Out("set_point_map")
set_point_map.add_map("firstOrder.y_start", "output1")
set_point_map.add_map("firstOrder1.y_start", "output2")

# set point values for the start and stop of the trajectory
set_point_nominal_start = TargetValue("set_point_nominal_start")
set_point_nominal_start.add_target("output1", 0.0)
set_point_nominal_start.add_target("output2", 0.0)

set_point_stop = VarSpace("set_point_stop")
set_point_stop.add_var_box("output1", 1.0, 2.5)
set_point_stop.add_var_box("output2", 1.2, 3.0)

# allowed range of output values, if exceeded termination (or rather truncation) of the episode
terminations = VarSpace("terminations")
terminations.add_var_box("output1", -0.6, 3.0)
terminations.add_var_box("output2", -0.6, 3.5)

# create FMUGymConfig object from all the above defined parameters
config = FMUGymConfig(fmu_path=os.path.abspath('examples/FMUs/dummy_for_FMU_cosim.fmu'),
                      start_time=0.0,
                      stop_time=10.0,
                      sim_step_size=0.01,
                      action_step_size=0.01,
                      inputs=inputs,
                      input_noise=input_noise,
                      outputs=outputs,
                      output_noise=output_noise,
                      random_vars=random_vars,
                      set_point_map=set_point_map,
                      set_point_nominal_start=set_point_nominal_start,
                      set_point_stop=set_point_stop,
                      terminations=terminations
                     )   

def get_dummy_config():
    return config

def get_dummy_FMUEnv():
    return FMUEnv(config)                     