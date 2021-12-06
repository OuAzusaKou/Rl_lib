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
                     render=True,
                     action_noise=0.1,
                     min_update_step=1000,
                     update_step=100,
                     lr_actor=1e-3,
                     lr_critic=1e-3,
                     polyak=0.995,

                     )

ddpg.learn(num_train_step=1000000)
