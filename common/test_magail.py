import os

from common.Agent_set import Agent
from common.algorithms import MAGAAlgorithms
from common.datacollection import GAIL_DataSet
from common.updator import actor_updator_gail, discriminator_updator
from common.user.custom_actor_critic import MLPfeature_extractor, ddpg_actor, ddpg_critic, gail_discriminator
from ma_env.ma_elec_env import wrapped_env
elec = 135.5
env = wrapped_env(elec)

actor_list = []
critic_list = []
DDPG_agent_class_list = []
data_collection_list = []
discriminator_list = []
data_collection_args_dict = {}
demonstrator_agent ={}

logdir = "./MAGAIL_tensorboard"

os.makedirs(logdir, exist_ok=True)


env.reset()
i = 0
for agent_name in env.agents:
    agent_num = i
    demonstrator_agent[agent_name] = Agent.load(actor_address='ac'+str(agent_num),critic_address='cri'+str(agent_num))
    i = i + 1
for agent_name in env.agents:

    feature_dim = env.observation_spaces[agent_name].shape[0]

    feature_extractor = MLPfeature_extractor(env.observation_spaces[agent_name], feature_dim)

    actor_list.append(ddpg_actor(env.action_spaces[agent_name], feature_extractor, feature_dim))

    critic_list.append(ddpg_critic(env.action_spaces[agent_name], feature_extractor, feature_dim))

    DDPG_agent_class_list.append(Agent)

    data_collection_list.append(GAIL_DataSet)

    discriminator_list.append(gail_discriminator(env.action_spaces[agent_name], feature_extractor, feature_dim))

    data_collection_args_dict[agent_name] = {'demonstrator_agent': demonstrator_agent[agent_name]}





magail = MAGAAlgorithms(agent_class_list=DDPG_agent_class_list,
                     actor_list=actor_list,
                     critic_list=critic_list,
                     #critic_update=critic_updator_ddpg,
                     actor_update=actor_updator_gail,
                     #soft_update=soft_update,
                     discriminator_update=discriminator_updator,
                     data_collection_args_dict=data_collection_args_dict,
                     data_collection_list=data_collection_list,
                     discriminator_list=discriminator_list,
                     env=env,
                     train=None,
                     buffer_size=1e6,
                     learning_rate=4e-4,
                     gamma=0.99,
                     batch_size=256,
                     tensorboard_log=logdir,
                     render=False,
                     action_noise=0.1,
                     min_update_step=4000,
                     update_step=100,
                     lr_actor=5e-4,
                     lr_critic=1e-3,
                     lr_discriminator=1e-4,
                     polyak=0.995,
                     start_steps = 2000,
                     )

magail.learn(num_train_step=1000000)