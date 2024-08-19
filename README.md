# Craftax LM
A wrapper around the Craftax agent benchmark, for evaluating digital agents
<p align="middle">
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/dungeon_crawling.gif" width="200" />
</p>

# Leaderboard

## Craftax-Classic
| LM | Algorithm | Reward (% max) |                                              Code                                               |
|:----------|---------------:|:-----------------------------------------------------------------------------------------------:|:---------------------------------------:|
| gpt-4o-mini | ReAct   |            14.2 | [CraftaxLM_Baselines](https://github.com/JoshuaPurtell/Apropos/blob/main/apropos/bench/crafter/test.py) |


## Craftax-Full
| LM | Algorithm | Reward (% max) |                                              Code                                               |
|:----------|---------------:|:-----------------------------------------------------------------------------------------------:|:---------------------------------------:|
| gpt-4o-mini | ReAct   |            01.2 | [CraftaxLM_Baselines](https://github.com/JoshuaPurtell/Apropos/blob/main/apropos/bench/crafter/test.py) |

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