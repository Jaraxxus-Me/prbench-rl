"""Base RL agent interface for PRBench environments."""

import abc
from typing import Any, Hashable, TypeVar

from gymnasium import spaces
from prpl_utils.gym_agent import Agent

_O = TypeVar("_O", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)


class BaseRLAgent(Agent[_O, _U]):
    """Base class for RL agents in PRBench environments."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        seed: int,
    ) -> None:
        super().__init__(seed)
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def _get_action(self) -> _U:
        """Get action from the current policy."""

    @abc.abstractmethod
    def train(self) -> None:
        """Train the agent (set to training mode)."""

    def _learn_from_transition(
        self,
        obs: _O,
        act: _U,
        next_obs: _O,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Learn from a single transition (to be implemented by RL algorithms)."""
        # Base implementation does nothing
        del obs, act, next_obs, reward, done, info

    def save(self, filepath: str) -> None:
        """Save agent parameters."""
        # Base implementation does nothing
        del filepath

    def load(self, filepath: str) -> None:
        """Load agent parameters."""
        # Base implementation does nothing
        del filepath
