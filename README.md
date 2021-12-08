## Rl_lib

### a customizable RL lib.

#### 1.test DQN
                python test_dqn.py
#### 2.test DDPG
                python test.py
#### 3.test image_input DDPG
                python test_image_input.py
#### 4.test MADDPG(last version do not support now)
                python test_maddpy.py
#### 5.test MAGAIL(last version do not support now)
                python test_magail.py




# JueRu

JueRu is a customizable RL lib based on Pytorch. The algorithms implemented now include 
DQN, DDPG, SAC, GAIL, MADDPG, MAGAIL.

**Note: despite its simplicity of use, Jueru assumes you have some knowledge about Reinforcement Learning (RL).

## Main Features

**The performance of each algorithm was tested** (see *Results* section in their respective page),



| **Features**                | **JueRu** |**Stable-Baselines3** | 
| --------------------------- | ----------------------| --- |
| Custom Agent | :heavy_check_mark: |:x:|
| Custom Learning | :heavy_check_mark: |:x:|
| Custom Functor | :heavy_check_mark: |:x:|
| Custom Updator | :heavy_check_mark: |:x:|
| Documentation               | :heavy_check_mark: |:heavy_check_mark:|
| Custom environments         | :heavy_check_mark: |:heavy_check_mark:|
| Custom policies             | :heavy_check_mark: |:heavy_check_mark:|
| Common interface            | :heavy_check_mark: |:heavy_check_mark:|
| Ipython / Notebook friendly | :heavy_check_mark: | :heavy_check_mark: |
| Tensorboard support         | :heavy_check_mark: | :heavy_check_mark: |
| `Dict` observation space support  | :x: |:heavy_check_mark:|
| State of the art RL methods |:x:| :heavy_check_mark: |




### Planned features

Please take a look at the [Roadmap](https://github.com/DLR-RM/stable-baselines3/issues/1) and [Milestones](https://github.com/DLR-RM/stable-baselines3/milestones).


## Documentation


## SB3-Contrib: Experimental RL Features

We implement experimental features in a separate contrib repository: [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)

This allows SB3 to maintain a stable and compact core, while still providing the latest features, like Truncated Quantile Critics (TQC), Quantile Regression DQN (QR-DQN) or PPO with invalid action masking (Maskable PPO).

Documentation is available online: [https://sb3-contrib.readthedocs.io/](https://sb3-contrib.readthedocs.io/)


## Installation

**Note:** Jueru supports PyTorch >= 1.7.1.

### Prerequisites
Jueru requires python 3.7+.

### Install using pip
Install the Jueru package:
```
pip install stable-baselines3[extra]
```
**Note:** Some shells such as Zsh require quotation marks around brackets, i.e. `pip install 'stable-baselines3[extra]'` ([More Info](https://stackoverflow.com/a/30539963)).

This includes an optional dependencies like Tensorboard, OpenCV or `atari-py` to train on atari games. If you do not need those, you can use:
```
pip install stable-baselines3
```


## Example

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run DQN on a cartpole environment:
```python
import gym
import numpy as np

from jueru.Agent_set import Agent
from jueru.algorithms import  DQNAlgorithm
from jueru.datacollection import Replay_buffer
from jueru.updator import critic_updator_dqn
from jueru.user.custom_actor_critic import FlattenExtractor, dqn_critic

env = gym.make('CartPole-v0')

feature_extractor = FlattenExtractor(env.observation_space)

critic = dqn_critic(env.action_space, feature_extractor, np.prod(env.observation_space.shape))

DQN_Agent = Agent

data_collection = Replay_buffer

functor_dict = {}

lr_dict = {}

updator_dict = {}

functor_dict['critic'] = critic

lr_dict['critic'] = 1e-3

updator_dict['critic_update'] = critic_updator_dqn

dqn = DQNAlgorithm(agent_class=DQN_Agent,
                   functor_dict=functor_dict,
                   lr_dict=lr_dict,
                   updator_dict=updator_dict,
                   data_collection=data_collection,
                   env=env,
                   buffer_size=1e6,
                   gamma=0.99,
                   batch_size=100,
                   tensorboard_log="./DQN_tensorboard",
                   render=True,
                   action_noise=0.1,
                   min_update_step=1000,
                   update_step=100,
                   polyak=0.995,

                   )

dqn.learn(num_train_step=1000000)
env.close()
```

Or just train a model with a one liner if [the environment is registered in Gym](https://github.com/openai/gym/wiki/Environments) and if [the policy is registered](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html):

```python
from stable_baselines3 import PPO

model = PPO('MlpPolicy', 'CartPole-v1').learn(10000)
```

Please read the [documentation](https://stable-baselines3.readthedocs.io/) for more examples.


## Try it online with Colab Notebooks !

All the following examples can be executed online using Google colab notebooks:

- [Full Tutorial](https://github.com/araffin/rl-tutorial-jnrr19)
- [All Notebooks](https://github.com/Stable-Baselines-Team/rl-colab-notebooks/tree/sb3)
- [Getting Started](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb)
- [Training, Saving, Loading](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb)
- [Multiprocessing](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb)
- [Monitor Training and Plotting](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb)
- [Atari Games](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb)
- [RL Baselines Zoo](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb)
- [PyBullet](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb)


## Implemented Algorithms

| **Name**         | **Recurrent**      | `Box`          | `Discrete`     | `MultiDiscrete` | `MultiBinary`  | **Multi Processing**              |
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| A2C   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DDPG  | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| DQN   | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark: |
| HER   | :x: | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :x:                | :x: |
| PPO   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| SAC   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| TD3   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| QR-DQN<sup>[1](#f1)</sup>  | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark: |
| TQC<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x: | :heavy_check_mark: |
| Maskable PPO<sup>[1](#f1)</sup>   | :x: | :x: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:  |

<b id="f1">1</b>: Implemented in [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) GitHub repository.

Actions `gym.spaces`:
 * `Box`: A N-dimensional box that containes every point in the action space.
 * `Discrete`: A list of possible actions, where each timestep only one of the actions can be used.
 * `MultiDiscrete`: A list of possible actions, where each timestep only one action of each discrete set can be used.
 * `MultiBinary`: A list of possible actions, where each timestep any of the actions can be used in any combination.



## Testing the installation
All unit tests in stable baselines3 can be run using `pytest` runner:
```
pip install pytest pytest-cov
make pytest
```

You can also do a static type check using `pytype`:
```
pip install pytype
make type
```

Codestyle check with `flake8`:
```
pip install flake8
make lint
```

## Projects Using Stable-Baselines3

We try to maintain a list of project using stable-baselines3 in the [documentation](https://stable-baselines3.readthedocs.io/en/master/misc/projects.html),
please tell us when if you want your project to appear on this page ;)

## Citing the Project

To cite this repository in publications:

```bibtex
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```

## Maintainers

Stable-Baselines3 is currently maintained by [Ashley Hill](https://github.com/hill-a) (aka @hill-a), [Antonin Raffin](https://araffin.github.io/) (aka [@araffin](https://github.com/araffin)), [Maximilian Ernestus](https://github.com/ernestum) (aka @ernestum), [Adam Gleave](https://github.com/adamgleave) (@AdamGleave) and [Anssi Kanervisto](https://github.com/Miffyli) (@Miffyli).

**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email.
Please post your question on the [RL Discord](https://discord.com/invite/xhfNqQv), [Reddit](https://www.reddit.com/r/reinforcementlearning/) or [Stack Overflow](https://stackoverflow.com/) in that case.


## How To Contribute

To any interested in making the baselines better, there is still some documentation that needs to be done.
If you want to contribute, please read [**CONTRIBUTING.md**](./CONTRIBUTING.md) guide first.

## Acknowledgments

The initial work to develop Stable Baselines3 was partially funded by the project *Reduced Complexity Models* from the *Helmholtz-Gemeinschaft Deutscher Forschungszentren*.

The original version, Stable Baselines, was created in the [robotics lab U2IS](http://u2is.ensta-paristech.fr/index.php?lang=en) ([INRIA Flowers](https://flowers.inria.fr/) team) at [ENSTA ParisTech](http://www.ensta-paristech.fr/en).


Logo credits: [L.M. Tenkes](https://www.instagram.com/lucillehue/)
