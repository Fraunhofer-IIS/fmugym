# Testing the compliance with gymnasium standard: https://gymnasium.farama.org/api/env/
from dummy_FMUEnv import get_dummy_FMUEnv
import gymnasium as gym

# create FMUEnv object from manually validated dummy environment
fmugym_test = get_dummy_FMUEnv()


def test_step():
    action = fmugym_test.action_space.sample()
    obs, reward, terminated, constraint_limit, info = fmugym_test.step(action)
    
    assert fmugym_test.observation_space.contains(obs)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(constraint_limit, bool)

def test_reset():
    obs, info = fmugym_test.reset()

    assert fmugym_test.observation_space.contains(obs)

def test_attributes():
    act_space = fmugym_test.action_space
    obs_space = fmugym_test.observation_space

    assert isinstance(act_space, gym.spaces.Box) or isinstance(act_space, gym.spaces.Discrete) or isinstance(act_space, gym.spaces.MultiDiscrete)

    assert isinstance(obs_space, gym.spaces.Box) or isinstance(obs_space, gym.spaces.Discrete) or isinstance(obs_space, gym.spaces.MultiDiscrete) or isinstance(obs_space, gym.spaces.Dict)

