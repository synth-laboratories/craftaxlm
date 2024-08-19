# Craftax LM
A wrapper around the Craftax agent benchmark, for evaluating digital agents

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
To cite the underlying Craftax environment, please cite:
```
@inproceedings{matthews2024craftax,
    author={Michael Matthews and Michael Beukman and Benjamin Ellis and Mikayel Samvelyan and Matthew Jackson and Samuel Coward and Jakob Foerster},
    title = {Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning},
    booktitle = {International Conference on Machine Learning ({ICML})},
    year = {2024}
}
```