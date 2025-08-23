"""PRBench RL package."""

from gymnasium.core import Env
from omegaconf import DictConfig

from .agent import BaseRLAgent
from .random_agent import RandomAgent

__all__ = ["create_rl_agents"]


def create_rl_agents(agent_cfg: DictConfig, env: Env, seed: int) -> BaseRLAgent:
    """Create agent based on configuration."""
    if agent_cfg.name == "random":
        return RandomAgent(env.observation_space, env.action_space, seed)
    raise ValueError(f"Unknown agent type: {agent_cfg.name}")
