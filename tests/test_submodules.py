"""Tests that we can import code from submodules."""

import importlib

import prbench
from prbench_models.utils import MultiEnvWrapper


def test_submodule_imports():
    """Dynamically test that submodules can be imported."""

    for module in ["prbench", "prbench_models"]:
        importlib.import_module(module)


def test_multi_env_wrapper():
    """Test basic functionality of MultiEnvWrapper."""
    # Register all PRBench environments
    prbench.register_all_environments()

    # Create environment factory function
    env_fn = lambda: prbench.make("prbench/StickButton2D-b5-v0")

    # Create multi-environment wrapper with 3 environments
    num_envs = 3
    multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs)

    assert multi_env.num_envs == num_envs
    assert hasattr(multi_env, "action_space")
    assert hasattr(multi_env, "observation_space")

    # Test reset
    obs_batch, info_batch = multi_env.reset(seed=123)
    assert obs_batch.shape[0] == num_envs
    assert isinstance(info_batch, dict)

    # Test step with random actions
    actions = multi_env.action_space.sample()
    assert actions.shape[0] == num_envs

    obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(actions)
    assert obs_batch.shape[0] == num_envs
    assert rewards.shape == (num_envs,)
    assert terminated.shape == (num_envs,)
    assert truncated.shape == (num_envs,)
    assert isinstance(info_batch, dict)

    # Test a few more steps to verify functionality
    for _ in range(3):
        actions = multi_env.action_space.sample()
        obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(actions)
        assert obs_batch.shape[0] == num_envs

    # Close environments
    multi_env.close()
