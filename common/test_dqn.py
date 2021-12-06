import gym
import numpy as np

from common.Agent_set import Agent
from common.algorithms import BaseAlgorithm, DQNAlgorithm
from common.datacollection import Replay_buffer
from common.updator import critic_updator_dqn, actor_updator_ddpg, soft_update
from common.user.custom_actor_critic import MLPfeature_extractor, ddpg_critic, FlattenExtractor, dqn_critic

env = gym.make('CartPole-v0')

feature_extractor = FlattenExtractor(env.observation_space)

#actor = dqn_actor(env.action_space, feature_extractor, 3)

critic = dqn_critic(env.action_space, feature_extractor, np.prod(env.observation_space.shape))

DQN_Agent = Agent

data_collection = Replay_buffer

dqn = DQNAlgorithm(agent_class=DQN_Agent,
            actor = None,
            critic = critic,
            critic_update=critic_updator_dqn,
            actor_update=actor_updator_ddpg,
            soft_update=soft_update,
            data_collection=data_collection,
            env=env,
            train=None,
            buffer_size = 1e6,
            learning_rate = 4e-4,
            gamma = 0.99,
            batch_size = 100,
            tensorboard_log = "./DQN_tensorboard",
            render = True,
            action_noise = 0.1,
            min_update_step = 1000,
            update_step = 100,
            lr_actor = 1e-3,
            lr_critic= 1e-3,
            polyak = 0.995,

)

dqn.learn(num_train_step= 1000000)
