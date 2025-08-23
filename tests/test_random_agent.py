"""Tests for the random agent."""

import numpy as np
from gymnasium import spaces

from prbench_rl.random_agent import RandomAgent


def test_random_agent_discrete_initialization():
    """Test random agent initialization with discrete action space."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)
    agent = RandomAgent(obs_space, action_space, seed=42)

    assert agent.observation_space == obs_space
    assert agent.action_space == action_space
    assert agent._last_observation is None
    assert agent._timestep == 0


def test_random_agent_continuous_initialization():
    """Test random agent initialization with continuous action space."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Box(low=-1, high=1, shape=(2,))
    agent = RandomAgent(obs_space, action_space, seed=42)

    assert agent.observation_space == obs_space
    assert agent.action_space == action_space


def test_random_agent_discrete_action_sampling():
    """Test that discrete actions are sampled correctly."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)
    agent = RandomAgent(obs_space, action_space, seed=42)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    agent.reset(mock_obs, {})

    # Sample multiple actions and verify they're in valid range
    actions = []
    for _ in range(10):
        action = agent.step()
        actions.append(action)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 3  # Discrete(3) should give actions 0, 1, 2

    # With 10 samples, we should get some variety (not all the same)
    assert len(set(actions)) > 1


def test_random_agent_continuous_action_sampling():
    """Test that continuous actions are sampled correctly."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Box(low=-1, high=1, shape=(2,))
    agent = RandomAgent(obs_space, action_space, seed=42)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    agent.reset(mock_obs, {})

    # Sample multiple actions and verify they're in valid range
    actions = []
    for _ in range(10):
        action = agent.step()
        actions.append(action)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)  # Box(shape=(2,))
        assert np.all(action >= -1) and np.all(action <= 1)  # Within bounds

    # Actions should be different
    actions_array = np.array(actions)
    assert not np.allclose(actions_array[0], actions_array[1])


def test_random_agent_seeded_reproducibility_discrete():
    """Test that seeded agents produce reproducible results."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)

    # Create two agents with same seed
    agent1 = RandomAgent(obs_space, action_space, seed=123)
    agent2 = RandomAgent(obs_space, action_space, seed=123)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    agent1.reset(mock_obs, {})
    agent2.reset(mock_obs, {})

    # Actions should be the same for same seed
    for _ in range(5):
        action1 = agent1.step()
        action2 = agent2.step()
        assert action1 == action2


def test_random_agent_seeded_reproducibility_continuous():
    """Test that seeded agents produce reproducible results."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Box(low=-1, high=1, shape=(2,))

    # Create two agents with same seed
    agent1 = RandomAgent(obs_space, action_space, seed=456)
    agent2 = RandomAgent(obs_space, action_space, seed=456)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    agent1.reset(mock_obs, {})
    agent2.reset(mock_obs, {})

    # Actions should be the same for same seed
    for _ in range(5):
        action1 = agent1.step()
        action2 = agent2.step()
        np.testing.assert_array_equal(action1, action2)


def test_random_agent_different_seeds():
    """Test that different seeds produce different action sequences."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)

    # Create agents with different seeds
    agent1 = RandomAgent(obs_space, action_space, seed=111)
    agent2 = RandomAgent(obs_space, action_space, seed=222)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    agent1.reset(mock_obs, {})
    agent2.reset(mock_obs, {})

    # Collect action sequences
    actions1 = [agent1.step() for _ in range(10)]
    actions2 = [agent2.step() for _ in range(10)]

    # Sequences should be different
    assert actions1 != actions2


def test_random_agent_update_behavior():
    """Test that update calls don't affect random action generation."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)
    agent = RandomAgent(obs_space, action_space, seed=42)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    agent.reset(mock_obs, {})

    # Take action, then update, then take another action
    action1 = agent.step()

    new_obs = [0.5, 0.6, 0.7, 0.8]
    agent.update(new_obs, 1.0, False, {})

    action2 = agent.step()

    # Both should be valid actions
    assert 0 <= action1 < 3
    assert 0 <= action2 < 3


def test_random_agent_train_eval_modes():
    """Test that train/eval mode doesn't affect random actions."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)
    agent = RandomAgent(obs_space, action_space, seed=42)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    agent.reset(mock_obs, {})

    # Test in eval mode
    agent.eval()
    action1 = agent.step()

    # Test in train mode
    agent.train()
    action2 = agent.step()

    # Both should be valid actions
    assert 0 <= action1 < 3
    assert 0 <= action2 < 3
