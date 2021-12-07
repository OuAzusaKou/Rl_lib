import gym
import numpy as np

from jueru.Agent_set import Agent
from jueru.algorithms import BaseAlgorithm
from jueru.datacollection import Replay_buffer
from jueru.updator import critic_updator_ddpg, actor_updator_ddpg, soft_update
from jueru.user.custom_actor_critic import ddpg_actor, ddpg_critic, FlattenExtractor
from jueru.uav_env.uav_env import Uav_env

#env = Uav_env(world_size=240, step_size=5, obstacle_num=5, max_step_num=100, display=False, fixed= False,obs_size = 66)

env = gym.make('Walker2d-v2')

feature_extractor = FlattenExtractor(env.observation_space)

actor = ddpg_actor(env.action_space, feature_extractor, np.prod(env.observation_space.shape))

critic = ddpg_critic(env.action_space, feature_extractor, np.prod(env.observation_space.shape))

DDPG_agent = Agent

data_collection = Replay_buffer

ddpg = BaseAlgorithm(agent_class=DDPG_agent,
                     actor=actor,
                     critic=critic,
                     critic_update=critic_updator_ddpg,
                     actor_update=actor_updator_ddpg,
                     soft_update=soft_update,
                     data_collection=data_collection,
                     env=env,
                     train=None,
                     buffer_size=1e6,
                     learning_rate=4e-4,
                     gamma=0.99,
                     batch_size=100,
                     tensorboard_log="./DQN_tensorboard/",
                     render=False,
                     action_noise=0.1,
                     min_update_step=1000,
                     update_step=100,
                     lr_actor=1e-3,
                     lr_critic=1e-3,
                     polyak=0.995,

                     )

ddpg.learn(num_train_step=1000000)
