"""Tests for the base RL agent."""

from gymnasium import spaces

from prbench_rl.agent import BaseRLAgent


class MockAgent(BaseRLAgent):
    """Mock concrete implementation of BaseRLAgent for testing."""

    def __init__(self, observation_space, action_space, seed):
        super().__init__(observation_space, action_space, seed)
        self.actions = [1, 2, 3]  # Mock actions to return
        self.action_index = 0

    def _get_action(self):
        """Return mock actions in sequence."""
        if self.action_index < len(self.actions):
            action = self.actions[self.action_index]
            self.action_index += 1
            return action
        return self.actions[-1]


def test_base_agent_initialization():
    """Test agent initialization."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(2)
    agent = MockAgent(obs_space, action_space, seed=42)

    assert agent.observation_space == obs_space
    assert agent.action_space == action_space
    assert agent._last_observation is None
    assert agent._last_action is None
    assert agent._timestep == 0
    assert agent._train_or_eval == "eval"


def test_base_agent_reset():
    """Test agent reset functionality."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(2)
    agent = MockAgent(obs_space, action_space, seed=42)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    mock_info = {"test": "data"}

    agent.reset(mock_obs, mock_info)

    assert agent._last_observation == mock_obs
    assert agent._last_info == mock_info
    assert agent._timestep == 0


def test_base_agent_step():
    """Test agent step functionality."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(2)
    agent = MockAgent(obs_space, action_space, seed=42)

    mock_obs = [0.1, 0.2, 0.3, 0.4]
    mock_info = {"test": "data"}
    agent.reset(mock_obs, mock_info)

    action = agent.step()

    assert action == 1  # First mock action
    assert agent._last_action == 1
    assert agent._timestep == 1

    # Test second step
    action2 = agent.step()
    assert action2 == 2  # Second mock action
    assert agent._timestep == 2


def test_base_agent_update():
    """Test agent update functionality."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(2)
    agent = MockAgent(obs_space, action_space, seed=42)

    # Initialize agent
    mock_obs = [0.1, 0.2, 0.3, 0.4]
    mock_info = {"test": "data"}
    agent.reset(mock_obs, mock_info)
    agent.step()  # Take an action

    # Update with new observation
    new_obs = [0.5, 0.6, 0.7, 0.8]
    new_info = {"new": "data"}
    reward = 1.0
    done = False

    agent.update(new_obs, reward, done, new_info)

    assert agent._last_observation == new_obs
    assert agent._last_info == new_info


def test_base_agent_train_eval_modes():
    """Test switching between train and eval modes."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(2)
    agent = MockAgent(obs_space, action_space, seed=42)

    assert agent._train_or_eval == "eval"

    agent.train()
    assert agent._train_or_eval == "train"

    agent.eval()
    assert agent._train_or_eval == "eval"


def test_base_agent_save_load_methods():
    """Test that base save/load methods don't raise errors."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(2)
    agent = MockAgent(obs_space, action_space, seed=42)

    # Base implementations should not raise errors
    agent.save("/tmp/test_agent.pkl")
    agent.load("/tmp/test_agent.pkl")


def test_base_agent_learn_from_transition():
    """Test that base _learn_from_transition doesn't raise errors."""
    obs_space = spaces.Box(low=0, high=1, shape=(4,))
    action_space = spaces.Discrete(2)
    agent = MockAgent(obs_space, action_space, seed=42)

    obs = [0.1, 0.2, 0.3, 0.4]
    action = 1
    next_obs = [0.5, 0.6, 0.7, 0.8]
    reward = 1.0
    done = False
    info = {"test": "data"}

    # Base implementation should not raise errors
    agent._learn_from_transition(obs, action, next_obs, reward, done, info)
