import gym

from common.Agent_set import Agent
from common.algorithms import BaseAlgorithm
from common.datacollection import Replay_buffer
from common.updator import critic_updator_ddpg, actor_updator_ddpg, soft_update
from common.user.custom_actor_critic import MLPfeature_extractor, ddpg_actor, ddpg_critic

env = gym.make('Pendulum-v1')

feature_extractor = MLPfeature_extractor(env.observation_space, 3)

actor = ddpg_actor(env.action_space, feature_extractor, 3)

critic = ddpg_critic(env.action_space, feature_extractor, 3)

DDPG_agent = Agent

data_collection = Replay_buffer

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
                     tensorboard_log="./DQN_tensorboard/",
                     render=True,
                     action_noise=0.1,
                     min_update_step=1000,
                     update_step=100,
                     polyak=0.995,
                     )

ddpg.learn(num_train_step=1000000)
