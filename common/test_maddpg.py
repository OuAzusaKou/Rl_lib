import gym

from common.Agent_set import Agent
from common.algorithms import BaseAlgorithm, MAAlgorithm, MAGAAlgorithms
from common.datacollection import Replay_buffer, SignleReplay_buffer, GAIL_DataSet
from common.updator import critic_updator_ddpg, actor_updator_ddpg, soft_update, discriminator_updator, \
    actor_updator_gail
from common.user.custom_actor_critic import MLPfeature_extractor, ddpg_actor, ddpg_critic
from ma_env.ma_elec_env import wrapped_env

elec = 135.5
env = wrapped_env(elec)

actor_list = []
critic_list = []
DDPG_agent_class_list = []
data_collection_list = []
discriminator_list = []

env.reset()


for agent_name in env.agents:
    feature_dim = env.observation_spaces[agent_name].shape[0]

    feature_extractor = MLPfeature_extractor(env.observation_spaces[agent_name], feature_dim)

    actor_list.append(ddpg_actor(env.action_spaces[agent_name], feature_extractor, feature_dim))

    critic_list.append(ddpg_critic(env.action_spaces[agent_name], feature_extractor, feature_dim))

    DDPG_agent_class_list.append(Agent)

    data_collection_list.append(SignleReplay_buffer)

    discriminator_list.append(None)


ddpg = MAAlgorithm(agent_class_list=DDPG_agent_class_list,
                     actor_list=actor_list,
                     critic_list=critic_list,
                     critic_update=critic_updator_ddpg,
                     actor_update=actor_updator_ddpg,
                     soft_update=soft_update,
                     #dicriminator_update=dicriminator_updator,
                     data_collection_list=data_collection_list,
                     discriminator_list=discriminator_list,
                     env=env,
                     train=None,
                     buffer_size=1e6,
                     learning_rate=4e-4,
                     gamma=0.99,
                     batch_size=100,
                     tensorboard_log="./DQN_tensorboard/",
                     render=False,
                     action_noise=0.1,
                     min_update_step=100,
                     save_step= 500,
                     update_step=100,
                     lr_actor=1e-3,
                     lr_critic=1e-3,
                     polyak=0.995,
                     start_steps = 200,
                     )

ddpg.learn(num_train_step=5000)
