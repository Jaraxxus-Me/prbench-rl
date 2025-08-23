"""PPO agent for PRBench environments."""

from typing import Any, Hashable, TypeVar

from gymnasium import spaces

from .agent import BaseRLAgent

_O = TypeVar("_O", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)


class PPOAgent(BaseRLAgent[_O, _U]):
    """PPO agent for PRBench environments (stub for future implementation)."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        seed: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
    ) -> None:
        super().__init__(observation_space, action_space, seed)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        # Networks and optimizer initialization will be implemented later

    def _get_action(self) -> _U:
        """Get action from current policy network."""
        # Policy network implementation will be added later
        # For now, just sample randomly like RandomAgent
        if isinstance(self.action_space, spaces.Discrete):
            # type: ignore[return-value]
            return int(self._rng.integers(0, self.action_space.n))
        # type: ignore[return-value]
        return self.action_space.sample()

    def _learn_from_transition(
        self,
        obs: _O,
        act: _U,
        next_obs: _O,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Store transition in replay buffer."""
        # Transition storage will be implemented later
        del obs, act, next_obs, reward, done, info
