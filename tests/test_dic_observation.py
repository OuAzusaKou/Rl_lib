import os
from copy import deepcopy

import numpy as np
import pytest
import torch as th
from gym import spaces, GoalEnv

from jueru.Agent_set import DDPG_agent
from jueru.algorithms import BaseAlgorithm, DQNAlgorithm, SACAlgorithm
from jueru.datacollection import Dict_Replay_buffer
from jueru.envs.uav_env.uav_env import Uav_env
from jueru.updator import actor_updator_ddpg, critic_updator_ddpg, soft_update
from jueru.user.custom_actor_critic import CombinedExtractor, ddpg_actor, ddpg_critic


@pytest.mark.parametrize("algorithm_class", [BaseAlgorithm, DQNAlgorithm, SACAlgorithm])
def test_dict_observation(algorithm_class):
    if algorithm_class == BaseAlgorithm:
        env = Uav_env(world_size=240, step_size=5, obstacle_num=5,
                      max_step_num=100, display=False, fixed=False,
                      obs_size=66)

        feature_dim = 128

        feature_extractor = CombinedExtractor(env.observation_space, feature_dim)

        actor = ddpg_actor(env.action_space, feature_extractor, feature_dim+16)

        critic = ddpg_critic(env.action_space, feature_extractor, feature_dim+16)

        data_collection = Dict_Replay_buffer

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
                             data_collection=data_collection,
                             env=env,
                             buffer_size=1e6,
                             gamma=0.99,
                             batch_size=100,
                             tensorboard_log="./DDPG_tensorboard/",
                             render=False,
                             action_noise=0.1,
                             min_update_step=1000,
                             update_step=100,
                             polyak=0.995,
                             start_steps=1000,
                             )

        ddpg.learn(num_train_step=5000)
