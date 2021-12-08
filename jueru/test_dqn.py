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