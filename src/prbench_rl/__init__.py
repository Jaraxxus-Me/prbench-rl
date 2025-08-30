"""PRBench RL package."""

from gymnasium.core import Env
from omegaconf import DictConfig

from prbench_rl.agent import BaseRLAgent
from prbench_rl.ppo_agent import PPOAgent
from prbench_rl.random_agent import RandomAgent

__all__ = ["create_rl_agents"]


def create_rl_agents(agent_cfg: DictConfig, env: Env, seed: int) -> BaseRLAgent:
    """Create agent based on configuration."""
    if agent_cfg.name == "random":
        return RandomAgent(env.observation_space, env.action_space, seed, agent_cfg)
    elif agent_cfg.name == "ppo":
        return PPOAgent(env.observation_space, env.action_space, seed, agent_cfg)
    raise ValueError(f"Unknown agent type: {agent_cfg.name}")
