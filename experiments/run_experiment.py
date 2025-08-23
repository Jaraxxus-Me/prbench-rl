"""Main entry point for running RL experiments.

Examples:
    python experiments/run_experiment.py agent=random env=obstruction2d-o0 seed=0

    python experiments/run_experiment.py -m agent=ppo env=obstruction2d-o0 \
        seed='range(0,10)'

    python experiments/run_experiment.py -m agent=ppo env=obstruction2d-o0 seed=0 \
        train_steps=10000,50000,100000
"""

import logging
import os
from typing import Any, Dict

import hydra
import numpy as np
import pandas as pd
import prbench
from gymnasium.core import Env
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from prbench_rl import BaseRLAgent, PPOAgent, RandomAgent


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    logging.info(
        f"Running agent={cfg.agent.name}, env={cfg.env.env_name}, seed={cfg.seed}"
    )

    # Create the environment
    prbench.register_all_environments()
    env = prbench.make(**cfg.env.make_kwargs)

    # Create the agent
    agent = _create_agent(cfg.agent, env, cfg.seed)

    if cfg.mode == "train":
        # Training pipeline
        logging.info("Starting training...")
        train_metrics = _run_training(
            agent,
            env,
            cfg.train_steps,
            cfg.eval_frequency,
            cfg.eval_episodes,
            cfg.max_eval_steps,
        )

        # Save trained agent
        current_dir = HydraConfig.get().runtime.output_dir
        agent_path = os.path.join(current_dir, "agent.pkl")
        agent.save(agent_path)
        logging.info(f"Saved trained agent to {agent_path}")

        # Save training metrics
        results_path = os.path.join(current_dir, "train_results.csv")
        pd.DataFrame(train_metrics).to_csv(results_path, index=False)
        logging.info(f"Saved training results to {results_path}")

    elif cfg.mode == "eval":
        # Evaluation pipeline
        logging.info("Starting evaluation...")
        if cfg.get("load_agent_path"):
            agent.load(cfg.load_agent_path)
            logging.info(f"Loaded agent from {cfg.load_agent_path}")

        agent.eval()
        eval_metrics = _run_evaluation(
            agent,
            env,
            cfg.eval_episodes,
            cfg.max_eval_steps,
            cfg.seed,
        )

        # Save evaluation results
        current_dir = HydraConfig.get().runtime.output_dir
        results_path = os.path.join(current_dir, "eval_results.csv")
        pd.DataFrame(eval_metrics).to_csv(results_path, index=False)
        logging.info(f"Saved evaluation results to {results_path}")

    # Save config
    current_dir = HydraConfig.get().runtime.output_dir
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)
    logging.info(f"Saved config to {config_path}")


def _create_agent(agent_cfg: DictConfig, env: Env, seed: int) -> BaseRLAgent:
    """Create agent based on configuration."""
    if agent_cfg.name == "random":
        return RandomAgent(env.observation_space, env.action_space, seed)
    if agent_cfg.name == "ppo":
        return PPOAgent(
            env.observation_space,
            env.action_space,
            seed,
            **agent_cfg.get("params", {}),
        )
    raise ValueError(f"Unknown agent type: {agent_cfg.name}")


def _run_training(
    agent: BaseRLAgent,
    env: Env,
    total_train_steps: int,
    eval_frequency: int,
    eval_episodes: int,
    max_eval_steps: int,
) -> list[Dict[str, Any]]:
    """Run the training loop."""
    agent.train()
    training_metrics = []

    step = 0
    episode = 0

    while step < total_train_steps:
        episode += 1
        obs, info = env.reset()
        agent.reset(obs, info)
        episode_reward = 0.0
        episode_steps = 0

        for _ in range(max_eval_steps):
            action = agent.step()
            next_obs, reward, done, truncated, info = env.step(action)
            agent.update(next_obs, float(reward), done or truncated, info)

            episode_reward += float(reward)
            episode_steps += 1
            step += 1

            if done or truncated or step >= total_train_steps:
                break

        # Log training progress
        logging.info(f"Episode {episode}, Steps {step}, Reward {episode_reward:.2f}")

        # Periodic evaluation
        if step % eval_frequency == 0:
            agent.eval()
            eval_metrics = _run_evaluation(
                agent, env, eval_episodes, max_eval_steps, step
            )
            avg_reward = np.mean([m["episode_reward"] for m in eval_metrics])
            success_rate = np.mean([m["success"] for m in eval_metrics])

            training_metrics.append(
                {
                    "step": step,
                    "episode": episode,
                    "eval_avg_reward": avg_reward,
                    "eval_success_rate": success_rate,
                    "train_episode_reward": episode_reward,
                }
            )

            logging.info(
                f"Eval at step {step}: avg_reward={avg_reward:.2f}, "
                f"success_rate={success_rate:.2f}"
            )
            agent.train()

    return training_metrics


def _run_evaluation(
    agent: BaseRLAgent,
    env: Env,
    num_episodes: int,
    max_steps: int,
    seed: int | None = None,
) -> list[Dict[str, Any]]:
    """Run evaluation episodes."""
    eval_metrics = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode if seed is not None else None)
        agent.reset(obs, info)

        episode_reward = 0.0
        episode_steps = 0
        success = False

        for _ in range(max_steps):
            action = agent.step()
            obs, reward, done, truncated, info = env.step(action)
            agent.update(obs, float(reward), done or truncated, info)

            episode_reward += float(reward)
            episode_steps += 1

            if done:
                success = True
                break
            if truncated:
                break

        eval_metrics.append(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "success": success,
            }
        )

    return eval_metrics


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
