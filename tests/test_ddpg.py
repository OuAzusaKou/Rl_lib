import argparse

import gym
import pytest
from jueru.Agent_set import DDPG_agent
from jueru.algorithms import BaseAlgorithm
from jueru.datacollection import Replay_buffer
from jueru.updator import critic_updator_ddpg, actor_updator_ddpg, soft_update
from jueru.user.custom_actor_critic import MLPfeature_extractor, ddpg_actor, ddpg_critic

def test_dqn():

    env = gym.make('Pendulum-v1')

    feature_extractor = MLPfeature_extractor(env.observation_space, 3)

    actor = ddpg_actor(env.action_space, feature_extractor, 3)

    critic = ddpg_critic(env.action_space, feature_extractor, 3)

    data_collection_dict = {}

    data_collection_dict['replay_buffer'] = Replay_buffer(env=env, size=1e6)

    functor_dict = {}

    lr_dict = {}

    updator_dict = {}

    functor_dict['actor'] = actor

    functor_dict['critic'] = critic

    functor_dict['actor_target'] = None

    functor_dict['critic_target'] = None

    lr_dict['actor'] = 1e-3

    lr_dict['critic'] = 1e-3

    lr_dict['actor_target'] = 1e-3

    lr_dict['critic_target'] = 1e-3

    updator_dict['actor_update'] = actor_updator_ddpg

    updator_dict['critic_update'] = critic_updator_ddpg

    updator_dict['soft_update'] = soft_update

    ddpg = BaseAlgorithm(agent_class=DDPG_agent,
                         functor_dict=functor_dict,
                         lr_dict=lr_dict,
                         updator_dict=updator_dict,
                         data_collection_dict=data_collection_dict,
                         env=env,
                         buffer_size=1e6,
                         gamma=0.99,
                         batch_size=100,
                         tensorboard_log="./DQN_tensorboard/",
                         render=True,
                         action_noise=0.1,
                         min_update_step=1000,
                         update_step=100,
                         polyak=0.995,
                         save_interval= 2000,
                         )

    ddpg.learn(num_train_step=2500)