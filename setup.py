from setuptools import setup, find_packages

setup(
    name="craftaxlm",
    version="0.0.4",
    packages=find_packages(),
    install_requires=[
        "absl-py==2.1.0",
        "backports.tarfile==1.2.0",
        "black==24.4.2",
        "build==1.2.1",
        "certifi==2024.7.4",
        "cfgv==3.4.0",
        "charset-normalizer==3.3.2",
        "chex==0.1.86",
        "click==8.1.7",
        "cloudpickle==3.0.0",
        "contourpy==1.2.1",
        "craftax==1.4.3",
        "cycler==0.12.1",
        "decorator==5.1.1",
        "distlib==0.3.8",
        "distrax==0.1.5",
        "dm-tree==0.1.8",
        "docker-pycreds==0.4.0",
        "docutils==0.21.2",
        "etils==1.9.2",
        "Farama-Notifications==0.0.4",
        "filelock==3.15.4",
        "flax==0.8.5",
        "fonttools==4.53.1",
        "fsspec==2024.6.1",
        "gast==0.6.0",
        "gitdb==4.0.11",
        "GitPython==3.1.43",
        "gym==0.26.2",
        "gym-notices==0.0.8",
        "gymnasium==0.29.1",
        "gymnax==0.0.8",
        "identify==2.6.0",
        "idna==3.7",
        "imageio==2.34.2",
        "importlib_metadata==8.2.0",
        "importlib_resources==6.4.0",
        "jaraco.classes==3.4.0",
        "jaraco.context==5.3.0",
        "jaraco.functools==4.0.2",
        "jax==0.4.30",
        "jaxlib==0.4.30",
        "keyring==25.3.0",
        "kiwisolver==1.4.5",
        "markdown-it-py==3.0.0",
        "matplotlib==3.9.1",
        "mdurl==0.1.2",
        "ml-dtypes==0.4.0",
        "more-itertools==10.4.0",
        "msgpack==1.0.8",
        "mypy-extensions==1.0.0",
        "nest-asyncio==1.6.0",
        "nh3==0.2.18",
        "nodeenv==1.9.1",
        "numpy==2.0.1",
        "opt-einsum==3.3.0",
        "optax==0.2.3",
        "orbax-checkpoint==0.5.23",
        "packaging==24.1",
        "pandas==2.2.2",
        "pathspec==0.12.1",
        "pillow==10.4.0",
        "pkginfo==1.10.0",
        "platformdirs==4.2.2",
        "pre-commit==3.8.0",
        "protobuf==5.27.2",
        "psutil==6.0.0",
        "pygame==2.6.0",
        "Pygments==2.18.0",
        "pyparsing==3.1.2",
        "pyproject_hooks==1.1.0",
        "python-dateutil==2.9.0.post0",
        "pytz==2024.1",
        "PyYAML==6.0.1",
        "readme_renderer==44.0",
        "requests==2.32.3",
        "requests-toolbelt==1.0.0",
        "rfc3986==2.0.0",
        "rich==13.7.1",
        "scipy==1.14.0",
        "seaborn==0.13.2",
        "sentry-sdk==2.11.0",
        "setproctitle==1.3.3",
        "six==1.16.0",
        "smmap==5.0.1",
        "tensorflow-probability==0.24.0",
        "tensorstore==0.1.63",
        "toolz==0.12.1",
        "twine==5.1.1",
        "typing_extensions==4.12.2",
        "tzdata==2024.1",
        "urllib3==2.2.2",
        "virtualenv==20.26.3",
        "wandb==0.17.5",
        "zipp==3.19.2",
    ],
    author="Josh Purtell",
    author_email="jmvpurtell@gmail.com",
    description="Craftax LM: Benchmarking LM Agents with Craftax",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JoshuaPurtell/craftaxlm",
    license="MIT",
)
