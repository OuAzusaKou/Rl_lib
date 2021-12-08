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

from jueru.Agent_set import DQN_agent
from jueru.algorithms import BaseAlgorithm, DQNAlgorithm
from jueru.datacollection import Replay_buffer
from jueru.updator import critic_updator_dqn, actor_updator_ddpg, soft_update
from jueru.user.custom_actor_critic import MLPfeature_extractor, ddpg_critic, FlattenExtractor, dqn_critic

env = gym.make('CartPole-v0')

feature_extractor = FlattenExtractor(env.observation_space)

# actor = dqn_actor(env.action_space, feature_extractor, 3)

critic = dqn_critic(env.action_space, feature_extractor, np.prod(env.observation_space.shape))

data_collection = Replay_buffer

functor_dict = {}

lr_dict = {}

updator_dict = {}

functor_dict['critic'] = critic

lr_dict['critic'] = 1e-3

updator_dict['critic_update'] = critic_updator_dqn

dqn = DQNAlgorithm(agent_class=DQN_agent,
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

dqn.learn(num_train_step=5)

agent = DQN_agent.load('Base_model_address')

obs = env.reset()
for i in range(1000):
    action = agent.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
```


## Implemented Algorithms

| **Name**         | **Recurrent**      | `Box`          | `Discrete`     |
| ------------------- | ------------------ | ------------------ | ------------------ |
| DDPG  | :x: | :heavy_check_mark: | :x:                |
| DQN   | :x: | :x: | :heavy_check_mark: |
| SAC   | :x: | :heavy_check_mark: | :x:                |
| MADDPG   | :x: | :heavy_check_mark: | :x:                |
| GAIL   | :x: | :heavy_check_mark: | :x:                |
| MAGAIL   | :x: | :heavy_check_mark: | :x:                |

<b id="f1">1</b>: Implemented in Jueru GitHub repository.

Actions `gym.spaces`:
 * `Box`: A N-dimensional box that containes every point in the action space.
 * `Discrete`: A list of possible actions, where each timestep only one of the actions can be used.



## Testing the installation
All unit tests in Jueru can be run using `pytest` runner:
```
pip install pytest pytest-cov
make pytest
```


## Projects Using Jueru

We try to maintain a list of project using Jueru in the documentation,
please tell us when if you want your project to appear on this page ;)



## Maintainers

Jueru is currently maintained by Zihang Wang, Jiayuan Li, Dunqi Yao.

**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email.
Please post your question on the [RL Discord](https://discord.com/invite/xhfNqQv), [Reddit](https://www.reddit.com/r/reinforcementlearning/) or [Stack Overflow](https://stackoverflow.com/) in that case.


## How To Contribute

To any interested in making the Jueru better, there is still some documentation that needs to be done.

