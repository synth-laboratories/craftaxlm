# Craftax LM
A wrapper around the Craftax agent benchmark, for evaluating digital agents over extremely long time horizons.

<p align="middle">
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/dungeon_crawling.gif" width="200" />
</p>

## Craftax-Classic
| LM | Algorithm | Score (% max) |                                              Code                                               |
|:----------|---------------:|:-----------------------------------------------------------------------------------------------:|:---------------------------------------:|
| claude-3-7-sonnet-latest (default) | ReAct   |            18.0 | |
| claude-3-5-sonnet-20241022 | ReAct   |            17.8 | |
| claude-3-5-sonnet-20240620 | ReAct   |            15.7 | |
| o3-mini | ReAct   |            12.6 | |
| gpt-4o | ReAct   |            7.0 | |

* Note - this is a limited evaluation where trajectories are terminated after 30 api calls, or roughly 150 in-game steps. 10 trajectories are rolled-out, yielding a log-weighted score as per the Crafter [paper](https://arxiv.org/abs/2109.06780). Reproducible code forthcoming.

# Usage
First, download the package with ```pip install craftaxlm```. Next, import the agent-computer interface of your choice via
```
from craftaxlm import CraftaxACI, CraftaxClassicACI
```
This package is early in development, so for implementation examples, please refer to the [baseline ReAct implementation](https://github.com/JoshuaPurtell/Apropos/blob/main/apropos/bench/craftax)

# Leaderboard
In order to make experiments reasonable to run across a range of LMs, currently the leaderboard evaluates agents in the following manner:
1. Five rollouts are sampled from the agent, with a hard cap of 300 actions per rollout.
2. The agent is evaluated using a modified version of the original Crafter score - 
    ```
    sum(ln(1 + P(1_achievement_obtained)) for achievement in achievements) / (sum(ln(2) * len(achievements)))
    ```
    where P(1_achievement_obtained) is the probability of the achievement being obtained in a single rollout. The key idea is that incremental progress towards difficult achievements ought to weigh more heavily in the score.

## Craftax-Full
| LM | Algorithm | Score (% max) |                                              Code                                               |
|:----------|---------------:|:-----------------------------------------------------------------------------------------------:|:---------------------------------------:|

# Dev Instructions
```
pyenv virtualenv craftax_env
poetry install
```

When in doubt

```
from jax import debug
...
debug.breakpoint()
```

# 📚 Citation
To learn more about Craftax, check out the paper [website](https://craftaxenv.github.io) here.
To cite the underlying Craftax environment, see:
```
@inproceedings{matthews2024craftax,
    author={Michael Matthews and Michael Beukman and Benjamin Ellis and Mikayel Samvelyan and Matthew Jackson and Samuel Coward and Jakob Foerster},
    title = {Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning},
    booktitle = {International Conference on Machine Learning ({ICML})},
    year = {2024}
}
```
To cite the Crafter benchmark, see:
```
@article{hafner2021crafter,
  title={Benchmarking the Spectrum of Agent Capabilities},
  author={Danijar Hafner},
  year={2021},
  journal={arXiv preprint arXiv:2109.06780},
}
```

# Contributing
## Setup
```
uv venv craftaxlm-dev
source craftaxlm-dev/bin/activate
uv sync
uv run ruff format .
```
## Help Wanted
- General code quality suggestions or improvements. Especially those that improve speed or reduce tokens.
- PRs to fix issues or add afforances that help your LM agent perform well
- Leaderboard submissions that demonstrate improved performance using algorithms for learning from data
