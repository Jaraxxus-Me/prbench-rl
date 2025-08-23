"""Random action baseline agent for PRBench environments."""

from typing import Hashable, TypeVar

from gymnasium import spaces

from .agent import BaseRLAgent

_O = TypeVar("_O", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)


class RandomAgent(BaseRLAgent[_O, _U]):
    """Random action baseline agent."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        seed: int,
    ) -> None:
        super().__init__(observation_space, action_space, seed)
        # Validate that we support the action space types
        if not isinstance(action_space, (spaces.Discrete, spaces.Box)):
            # Will use fallback sampling for other space types
            pass
        # Seed the action space for reproducible random actions
        self.action_space.seed(seed)

    def _get_action(self) -> _U:
        """Sample a random action from the action space."""
        return self.action_space.sample()

    def train(self) -> None:
        """Set the agent to training mode (no-op for random agent)."""
        self._train_or_eval = "train"
