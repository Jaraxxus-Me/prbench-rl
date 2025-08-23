"""Tests for the PPO agent."""

from gymnasium import spaces

from prbench_rl.agent import BaseRLAgent
from prbench_rl.ppo_agent import PPOAgent


def test_ppo_agent_initialization_default_params():
    """Test PPO agent initialization with default hyperparameters."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)
    agent = PPOAgent(obs_space, action_space, seed=42)

    assert agent.observation_space == obs_space
    assert agent.action_space == action_space
    assert agent.learning_rate == 3e-4
    assert agent.gamma == 0.99
    assert agent.gae_lambda == 0.95
    assert agent.clip_coef == 0.2
    assert agent.ent_coef == 0.01
    assert agent.vf_coef == 0.5


def test_ppo_agent_initialization_custom_params():
    """Test PPO agent initialization with custom hyperparameters."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)

    custom_params = {
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "gae_lambda": 0.9,
        "clip_coef": 0.1,
        "ent_coef": 0.02,
        "vf_coef": 1.0,
    }

    agent = PPOAgent(obs_space, action_space, seed=42, **custom_params)

    assert agent.learning_rate == 1e-3
    assert agent.gamma == 0.95
    assert agent.gae_lambda == 0.9
    assert agent.clip_coef == 0.1
    assert agent.ent_coef == 0.02
    assert agent.vf_coef == 1.0


def test_ppo_agent_discrete_actions():
    """Test that PPO agent can produce discrete actions."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)
    agent = PPOAgent(obs_space, action_space, seed=42)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    agent.reset(mock_obs, {})

    # For now, PPO is just sampling random actions (stub implementation)
    action = agent.step()
    assert isinstance(action, int)
    assert 0 <= action < 3  # Discrete(3) should give actions 0, 1, 2


def test_ppo_agent_continuous_actions():
    """Test that PPO agent can produce continuous actions."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Box(low=-1, high=1, shape=(2,))
    agent = PPOAgent(obs_space, action_space, seed=42)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    agent.reset(mock_obs, {})

    # For now, PPO is just sampling random actions (stub implementation)
    action = agent.step()
    assert hasattr(action, "shape")  # Should be numpy array-like
    assert len(action) == 2  # Box(shape=(2,))


def test_ppo_agent_learn_from_transition():
    """Test that learn_from_transition doesn't crash (stub implementation)."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)
    agent = PPOAgent(obs_space, action_space, seed=42)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    agent.reset(mock_obs, {})
    action = agent.step()

    next_obs = [0.5, 0.6, 0.7, 0.8]
    reward = 1.0
    done = False
    info = {"test": "data"}

    # Should not raise any errors (stub implementation)
    agent.update(next_obs, reward, done, info)


def test_ppo_agent_train_eval_modes():
    """Test switching between train and eval modes."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)
    agent = PPOAgent(obs_space, action_space, seed=42)

    assert agent._train_or_eval == "eval"

    agent.train()
    assert agent._train_or_eval == "train"

    agent.eval()
    assert agent._train_or_eval == "eval"


def test_ppo_agent_save_load():
    """Test that save/load methods don't crash (stub implementation)."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)
    agent = PPOAgent(obs_space, action_space, seed=42)

    # Should not raise errors (stub implementation)
    agent.save("/tmp/test_ppo_agent.pkl")
    agent.load("/tmp/test_ppo_agent.pkl")


def test_ppo_agent_hyperparameter_storage():
    """Test that hyperparameters are stored correctly."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)

    # Test with various hyperparameter combinations
    test_cases = [
        {"learning_rate": 1e-4, "gamma": 0.999},
        {"clip_coef": 0.3, "ent_coef": 0.005},
        {"gae_lambda": 0.8, "vf_coef": 0.25},
    ]

    for params in test_cases:
        agent = PPOAgent(obs_space, action_space, seed=42, **params)

        for key, expected_value in params.items():
            assert getattr(agent, key) == expected_value


def test_ppo_agent_multiple_episodes():
    """Test agent behavior across multiple episodes."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)
    agent = PPOAgent(obs_space, action_space, seed=42)

    for episode in range(3):
        mock_obs = [0.1 * episode, 0.2 * episode, 0.3 * episode, 0.4 * episode]
        agent.reset(mock_obs, {})

        # Simulate a short episode
        for step in range(5):
            action = agent.step()
            assert 0 <= action < 3

            next_obs = [
                0.1 + step * 0.1,
                0.2 + step * 0.1,
                0.3 + step * 0.1,
                0.4 + step * 0.1,
            ]
            reward = 0.1 * step
            done = step == 4  # End episode on last step

            agent.update(next_obs, reward, done, {})


def test_ppo_agent_inheritance():
    """Test that PPOAgent properly inherits from BaseRLAgent."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(3)
    agent = PPOAgent(obs_space, action_space, seed=42)

    assert isinstance(agent, BaseRLAgent)

    # Test inherited methods work
    mock_obs = [0.1, 0.2, 0.3, 0.4]
    agent.reset(mock_obs, {})
    assert agent._last_observation == mock_obs
    assert agent._timestep == 0

    action = agent.step()
    assert agent._last_action == action
    assert agent._timestep == 1
