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

    def _get_action(self) -> _U:
        """Sample a random action from the action space."""
        if isinstance(self.action_space, spaces.Discrete):
            # type: ignore[return-value]
            return self._rng.integers(0, self.action_space.n)
        if isinstance(self.action_space, spaces.Box):
            # type: ignore[return-value]
            return self._rng.uniform(
                low=self.action_space.low,
                high=self.action_space.high,
                size=self.action_space.shape,
            ).astype(self.action_space.dtype)
        # Fallback to action space's own sampling for other space types
        return self.action_space.sample()  # type: ignore[return-value]
