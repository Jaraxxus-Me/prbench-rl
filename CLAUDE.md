# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PRBench-RL is a reinforcement learning baselines repository for PRBench (Planning and Reasoning Benchmark). The project provides bilevel planning agents and experiment infrastructure for evaluating RL approaches on planning tasks.

## Architecture

The repository structure consists of:

- **Main package**: `src/prbench_rl/` - Contains the core bilevel planning agent implementation
- **Experiments**: `experiments/` - Contains experiment running infrastructure with Hydra configuration
- **Submodules**: `third-party/` - Contains two key submodules:
  - `third-party/prbench/` - The core PRBench environment library
  - `third-party/cleanrl/` - Clean RL implementations for reference

### Key Components

1. **BaseRLAgent** (`src/prbench_rl/agent.py`): Abstract base class for RL agents, inheriting from `prpl_utils.gym_agent.Agent`
2. **RandomAgent** (`src/prbench_rl/random_agent.py`): Random action baseline agent with seeded reproducibility
3. **PPOAgent** (`src/prbench_rl/ppo_agent.py`): PPO agent implementation (currently a stub for future development)
4. **Experiment Runner** (`experiments/run_experiment.py`): Hydra-based experiment orchestration supporting both training and evaluation
5. **Environment Integration**: Leverages PRBench environments for 2D/3D planning tasks

## Development Commands

### Environment Setup
```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/Jaraxxus-Me/prbench-rl.git

# Install in development mode
pip install -e ".[develop]"

# Install submodules
pip install -e third-party/prbench
```

### Code Quality
```bash
# Auto-format code
./run_autoformat.sh

# Run full CI checks (includes formatting, type checking, linting, and tests)
./run_ci_checks.sh

# Individual commands:
mypy .                                    # Type checking
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc  # Linting
pytest tests/                            # Run tests
```

### Running Experiments
```bash
# Single evaluation run
python experiments/run_experiment.py agent=random env=obstruction2d-o0 seed=0

# Training mode
python experiments/run_experiment.py mode=train agent=ppo env=obstruction2d-o0 train_steps=10000

# Multi-run experiments (Hydra multirun mode)
python experiments/run_experiment.py -m agent=random env=obstruction2d-o0 seed='range(0,10)'

# Parameter sweeps
python experiments/run_experiment.py -m agent=ppo mode=train env=obstruction2d-o0 train_steps=1000,5000,10000
```

## Configuration

- **Python version**: 3.11+ required
- **Code style**: Black formatting (line length 88), isort imports
- **Type checking**: MyPy with strict settings
- **Linting**: Pylint with custom configuration in `.pylintrc`
- **Testing**: pytest with pylint integration

## Important Notes

- The project excludes `third-party/` directories from linting and type checking
- Experiments use Hydra for configuration management (though no .yaml config files were found in `experiments/conf/`)
- The codebase depends on bilevel planning libraries and PRBench environments
- All shell scripts should be run from the repository root directory