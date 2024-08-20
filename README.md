# Craftax LM
A wrapper around the Craftax agent benchmark, for evaluating digital agents.

<p align="middle">
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/dungeon_crawling.gif" width="200" />
</p>

# Usage
First, download the package with ```pip install craftaxlm```. Next, import the agent-computer interface of your choice via
```
from craftaxlm import CraftaxACI, CraftaxClassicACI
```
This package is early in development, so for implementation examples, please refer to the [baseline ReAct implementation](https://github.com/JoshuaPurtell/Apropos/blob/main/apropos/bench/craftax)

# Leaderboard

## Craftax-Classic
| LM | Algorithm | Reward (% max) |                                              Code                                               |
|:----------|---------------:|:-----------------------------------------------------------------------------------------------:|:---------------------------------------:|
| gpt-4o-mini | ReAct   |            18.4 | [CraftaxLM_Baselines](https://github.com/JoshuaPurtell/Apropos/blob/main/apropos/bench/craftax/test.py) |

## Craftax-Full
| LM | Algorithm | Reward (% max) |                                              Code                                               |
|:----------|---------------:|:-----------------------------------------------------------------------------------------------:|:---------------------------------------:|
| gpt-4o-mini | ReAct   |            02.9 | [CraftaxLM_Baselines](https://github.com/JoshuaPurtell/Apropos/blob/main/apropos/bench/craftax/test.py) |

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
```
uv venv craftaxlm-dev
source craftaxlm-dev/bin/activate
uv run ruff format .
```