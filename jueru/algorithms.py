import copy
import os
from random import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

# import spinup.ddpg
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

from jueru.utils import get_linear_fn

from jueru.utils import get_latest_run_id


class BaseAlgorithm:
    def __init__(
            self,
            agent_class,
            data_collection_dict: Dict,
            env: Any,
            updator_dict=None,
            functor_dict=None,
            optimizer_dict=None,
            lr_dict=None,
            exploration_rate: float = 0.1,
            exploration_start: float = 1,
            exploration_end: float = 0.05,
            exploration_fraction: float = 0.2,
            polyak: float = 0.9,
            agent_args: Dict[str, Any] = None,
            device: Union[torch.device, str] = "auto",
            max_episode_steps = None,
            eval_func = None,
            gamma: float = 0.95,
            batch_size: int = 512,
            tensorboard_log: str = "./DQN_tensorboard/",
            tensorboard_log_name: str = "run",
            render: bool = False,
            action_noise: float = 0.1,
            min_update_step: int = 1000,
            update_step: int = 100,
            start_steps: int = 10000,
            model_address: str = "./Base_model_address",
            save_mode: str = 'step',
            save_interval: int = 5000,
            eval_freq: int = 100,
            eval_num_episode: int = 10,


    ):
        self.env = env
        self.eval_func = eval_func
        os.makedirs(tensorboard_log, exist_ok=True)
        os.makedirs(model_address, exist_ok=True)
        latest_run_id = get_latest_run_id(tensorboard_log, tensorboard_log_name)

        save_path = os.path.join(tensorboard_log, f"{tensorboard_log_name}_{latest_run_id + 1}")

        os.makedirs(save_path, exist_ok=True)

        self.device = device
        self.max_episode_steps = max_episode_steps
        self.eval_num_episode = eval_num_episode
        self.save_mode = save_mode
        self.eval_freq = eval_freq
        self.model_address = model_address
        self.save_interval = save_interval
        self.writer = SummaryWriter(save_path)
        self.exploration_rate = exploration_rate
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.exploration_fraction = exploration_fraction
        self.exploration_func = get_linear_fn(start=exploration_start,
                                              end=exploration_end,
                                              end_fraction=self.exploration_fraction)


        if agent_args:

            self.agent = agent_class(functor_dict=functor_dict,
                                     optimizer_dict=optimizer_dict,
                                     lr_dict=lr_dict,
                                     device=self.device,
                                     **agent_args)
        else:
            self.agent = agent_class(functor_dict=functor_dict,
                                     optimizer_dict=optimizer_dict,
                                     lr_dict=lr_dict,
                                     device=self.device
                                     )

        self.updator_dict = updator_dict

        self.data_collection_dict = data_collection_dict

        self.render = render

        self.action_noise = action_noise

        self.min_update_step = min_update_step

        self.update_step = update_step

        self.batch_size = batch_size

        self.gamma = gamma

        self.polyak = polyak

        self.start_steps = start_steps





    def eval_performance(self, num_episode, step):
        obs = self.env.reset()
        target_count = 0
        count_episode = 0
        episode_reward = 0
        list_episode_reward = []
        success_num = 0
        while count_episode < num_episode:
            action = self.agent.predict(obs)
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            # env.render()
            if done:
                if reward > 0:
                    success_num += 1
                obs = self.env.reset()
                count_episode += 1
                list_episode_reward.append(copy.deepcopy(episode_reward))
                episode_reward = 0
        success_rate = success_num / num_episode
        average_reward = sum(list_episode_reward) / len(list_episode_reward)
        self.writer.add_scalar('eval_average_reward', average_reward, global_step=step)
        self.writer.add_scalar('eval_success_rate', success_rate, global_step=step)
        return average_reward

    def learn(self, num_train_step):
        """interact"""
        self.agent.functor_dict['actor'].train()
        self.agent.functor_dict['critic'].train()
        # self.agent.actor_target.train()
        # self.agent.critic_target.train()

        step = 0
        episode_num=0
        average_reward_buf = - 1e6
        while step <= (num_train_step):

            state = self.env.reset()
            episode_reward = 0

            while True:

                if self.render:
                    self.env.render()

                if step >= self.start_steps:
                    action = self.agent.choose_action(state, self.action_noise)
                    print(action)
                else:
                    action = self.env.action_space.sample()

                next_state, reward, done, _ = self.env.step(action)

                if step >= self.max_episode_steps:
                    done = True

                done_value = 0 if done else 1
                
                
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                #print(state)
                self.data_collection_dict['replay_buffer'].store(state, action, reward, next_state, done_value)

                state = next_state.copy()

                episode_reward += reward

                if step >= self.min_update_step and step % self.update_step == 0:
                    for i in range(self.update_step):
                        batch = self.data_collection_dict['replay_buffer'].sample_batch(self.batch_size)  # random sample batch
                        critic_loss = self.updator_dict['critic_update'](self.agent, state=batch['state'], action=batch['action'],
                                                           reward=batch['reward'], next_state=batch['next_state'],
                                                           done_value=batch['done'], gamma=self.gamma)
                        if i % 4 == 0:
                            actor_loss = self.updator_dict['actor_update'](self.agent, state=batch['state'], action=batch['action'],
                                                              reward=batch['reward'], next_state=batch['next_state'],
                                                              gamma=self.gamma)

                            self.updator_dict['soft_update'](self.agent.functor_dict['actor_target'],
                                                             self.agent.functor_dict['actor'],
                                                             polyak=self.polyak)

                            self.updator_dict['soft_update'](self.agent.functor_dict['critic_target'],
                                                             self.agent.functor_dict['critic'],
                                                             polyak=self.polyak)

                        self.writer.add_scalar('critic_loss', critic_loss, global_step=(step+i))
                        self.writer.add_scalar('actor_loss', actor_loss, global_step=(step + i))


                step += 1

                # if step >= self.min_update_step and step % self.save_interval == 0:
                #     self.agent.save(address=self.model_address)
                if done:
                    episode_num += 1

                    self.writer.add_scalar('episode_reward', episode_reward, global_step=step)

                    if self.save_mode == 'eval':
                        if step >= self.min_update_step and episode_num % self.eval_freq == 0:
                            average_reward = self.eval_performance(num_episode=self.eval_num_episode, step=step)
                            if average_reward > average_reward_buf:
                                self.agent.save(address=self.model_address)
                                average_reward_buf = average_reward
                    break


class DQNAlgorithm(BaseAlgorithm):
    def learn(self, num_train_step):
        """interact"""
        self.agent.functor_dict['critic'].train()
        # self.agent.actor_target.train()
        # self.agent.critic_target.train()
        self.step_num = 0
        episode_num=0
        average_reward_buf = - 1e6
        while self.step_num <= (num_train_step):

            state = self.env.reset()
            episode_reward = 0

            while True:

                if self.render:
                    self.env.render()

                rand = np.random.rand(1)

                if rand > self.exploration_rate:

                    action = self.agent.choose_action_by_critic(state)
                    action = action.numpy()

                else:

                    action = self.env.action_space.sample()

                next_state, reward, done, _ = self.env.step(action)

                done_value = 0 if done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                self.data_collection_dict['replay_buffer'].store(state, action, reward, next_state, done_value)

                state = next_state

                episode_reward += reward

                if self.step_num >= self.min_update_step and self.step_num % self.update_step == 0:
                    for i in range(self.update_step):
                        batch = self.data_collection_dict['replay_buffer'].sample_batch(self.batch_size)  # random sample batch
                        loss = self.updator_dict['critic_update'](self.agent, state=batch['state'], action=batch['action'],
                                                           reward=batch['reward'], next_state=batch['next_state'],
                                                           done_value=batch['done'], gamma=self.gamma)

                        self.writer.add_scalar('loss', loss, global_step=(self.step_num+i))
                self.step_num += 1

                self.exploration_rate = self.exploration_func((1 - self.step_num / num_train_step))
                # if step >= self.min_update_step and step % self.save_interval == 0:
                #     self.agent.save(address=self.model_address)
                if done:

                    episode_num += 1

                    self.writer.add_scalar('episode_reward', episode_reward, global_step=self.step_num)

                    if self.save_mode == 'eval':
                        print('eval')
                        if self.step_num >= self.min_update_step and episode_num % self.eval_freq == 0:
                            average_reward = self.eval_performance(num_episode=self.eval_num_episode, step=self.step_num)
                            if average_reward >= average_reward_buf:
                                self.agent.save(address=self.model_address)
                                average_reward_buf = average_reward
                            if self.eval_func:
                                self.eval_func(self)
                    self.writer.add_scalar('episode_reward_step', episode_reward, global_step=self.step_num)
                    self.writer.add_scalar('exploration_rate_step', self.exploration_rate, global_step=self.step_num)

                    break



class SACAlgorithm(BaseAlgorithm):
    def learn(self, num_train_step, actor_update_freq, reward_scale=1):
        self.actor_update_freq = actor_update_freq
        self.agent.functor_dict['actor'].train()
        self.agent.functor_dict['critic'].train()
        step = 0
        episode_num = 0
        average_reward_buf = - 1e6
        while step <= (num_train_step):

            state = self.env.reset()
            episode_reward = 0
            episode_step = 0
            while True:

                if self.render:
                    self.env.render()

                if step < self.start_steps:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.agent.sample_action(state)

                next_state, reward, done, _ = self.env.step(action)

                reward = reward_scale * reward

                episode_step+=1
                if self.max_episode_steps:
                    if episode_step == self.max_episode_steps:
                        done = True

                done_value = 0 if done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                self.data_collection_dict['replay_buffer'].store(state, action, reward, next_state, done_value)

                state = next_state.copy()

                episode_reward += reward

                if step >= self.min_update_step and step % self.update_step == 0:
                    for _ in range(self.update_step):
                        batch = self.data_collection_dict['replay_buffer'].sample_batch(self.batch_size)

                        self.updator_dict['critic_update'](self.agent, obs=batch['state'], action=batch['action'],
                                                           reward=batch['reward'], next_obs=batch['next_state'],
                                                           not_done=batch['done'], gamma=self.gamma)
                        self.updator_dict['actor_and_alpha_update'](self.agent, obs=batch['state'],
                                                                    target_entropy=-self.agent.functor_dict[
                                                                        'critic'].action_dim)

                        self.updator_dict['soft_update'](self.agent.functor_dict['critic_target'].Q1,
                                                         self.agent.functor_dict['critic'].Q1,
                                                         polyak=self.polyak)
                        self.updator_dict['soft_update'](self.agent.functor_dict['critic_target'].Q2,
                                                         self.agent.functor_dict['critic'].Q2,
                                                         polyak=self.polyak)

                step += 1
                # if step >= self.min_update_step and step % self.save_interval == 0:
                #     self.agent.save(address=self.model_address)
                if done:
                    episode_num += 1
                    self.writer.add_scalar('episode_reward', episode_reward, global_step=step)
                    if self.save_mode == 'eval':
                        if step >= self.min_update_step and episode_num % self.eval_freq == 0:
                            average_reward = self.eval_performance(num_episode=self.eval_num_episode, step=step)
                            if average_reward > average_reward_buf:
                                self.agent.save(address=self.model_address)
                            average_reward_buf = average_reward
                    break


class MAAlgorithm:
    def __init__(
            self,
            agent_class_list,
            critic_update: None,
            actor_update: None,
            data_collection_list: [],
            env,
            train: None,
            soft_update: None,
            actor_list: [],
            critic_list: [],
            discriminator_list: [Any] = None,
            demonstrate_agent_list: [Any] = None,
            discriminator_update: [Any] = None,
            polyak: float = 0.9,
            optimizer_actor: [torch.optim.Adam] = torch.optim.Adam,
            optimizer_critic: [torch.optim.Adam] = torch.optim.Adam,
            optimizer_discriminator: [torch.optim.Adam] = torch.optim.Adam,
            agent_args: Dict[str, Any] = None,
            data_collection_args_dict: Dict[str, Dict[str, Any]] = None,
            lr_actor: float = 1e-2,
            lr_critic: float = 1e-2,
            lr_discriminator: float = 1e-3,
            buffer_size: int = 1e5,
            learning_rate: float = 4e-4,
            gamma: float = 0.95,
            batch_size: int = 512,
            tensorboard_log: str = None,
            render: bool = False,
            action_noise: float = 0.1,
            min_update_step: int = 1000,
            update_step: int = 100,
            start_steps: int = 10000,
            save_step: int = 2000,
    ):
        self.env = env
        self.agent_list = []
        self.data_collection_list = []
        self.data_collection_args_dict = {}
        self.writer = SummaryWriter(tensorboard_log)

        if data_collection_args_dict is None:

            for agent_name in self.env.possible_agents:
                self.data_collection_args_dict[agent_name] = {}

        else:

            self.data_collection_args_dict = data_collection_args_dict

        if agent_args:

            for agent_class, actor, critic, discriminator, in zip(agent_class_list, actor_list, critic_list,
                                                                  discriminator_list):
                self.agent_list.append(agent_class(Actor=actor,
                                                   Critic=critic,
                                                   Discriminator=discriminator,
                                                   optimizer_actor=optimizer_actor,
                                                   optimizer_critic=optimizer_critic,
                                                   optimizer_discriminator=optimizer_discriminator,
                                                   lr_actor=lr_actor,
                                                   lr_critic=lr_critic,
                                                   lr_discriminator=lr_discriminator,
                                                   **agent_args,
                                                   ))
        else:
            for agent_name, agent_class, actor, critic, data_collection, discriminator in zip(self.env.possible_agents,
                                                                                              agent_class_list,
                                                                                              actor_list, critic_list,
                                                                                              data_collection_list,
                                                                                              discriminator_list,
                                                                                              ):
                self.agent_list.append(agent_class(Actor=actor,
                                                   Critic=critic,
                                                   Discriminator=discriminator,
                                                   optimizer_actor=optimizer_actor,
                                                   optimizer_critic=optimizer_critic,
                                                   optimizer_discriminator=optimizer_discriminator,
                                                   lr_actor=lr_actor,
                                                   lr_critic=lr_critic,
                                                   lr_discriminator=lr_discriminator,
                                                   ))
                self.data_collection_list.append(
                    data_collection(self.env.observation_spaces[agent_name], self.env.action_spaces[agent_name],
                                    buffer_size, **(self.data_collection_args_dict[agent_name])))

        self.critic_update = critic_update

        self.actor_update = actor_update

        self.discriminator_update = discriminator_update

        self.train = train

        self.render = render

        self.action_noise = action_noise

        self.min_update_step = min_update_step

        self.update_step = update_step

        self.batch_size = batch_size

        self.gamma = gamma

        self.soft_update = soft_update

        self.polyak = polyak

        self.start_steps = start_steps

        self.save_step = save_step

        self.agent_name_mapping = dict(zip(self.env.possible_agents, list(range(len(self.env.possible_agents)))))

    def learn(self, num_train_step):
        """maddpg learning, update actor and critic through env"""
        for agent in self.agent_list:
            agent.actor.train()
            agent.critic.train()
        # self.agent.actor_target.train()
        # self.agent.critic_target.train()
        step = 0
        self.collect_flag = {}
        self.before_state_list = {}
        self.before_action_list = {}
        self.episode_reward_dict = {}
        while step <= (num_train_step):
            print(step)
            step += 1
            self.env.reset()

            # for agent in self.env.agents:
            #
            #     self.env.agent_selection = agent
            #
            #     self.before_state_list[agent] = state
            #
            #     self.before_action_list[agent] = 0
            for agent_name in self.env.agents:
                self.collect_flag[agent_name] = False
                self.episode_reward_dict[agent_name] = 0
            for agent_name in self.env.agent_iter():
                # print(agent_name)
                if self.render:
                    self.env.render()
                agent_num = self.agent_name_mapping[agent_name]
                state, before_reward, before_done, _ = self.env.last()
                # print('x_state',state)
                # self.next_state_list[agent] = state
                # total_reward += reward
                if step >= self.start_steps:
                    action = self.agent_list[agent_num].choose_action(state, self.action_noise)
                    # print(action)
                else:
                    action = self.env.action_spaces[agent_name].sample()
                # print(action)
                if before_done:
                    action = None

                # print(action)
                self.env.step(action)

                before_done_value = 0 if before_done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                if self.collect_flag[agent_name]:
                    # print('action', self.before_action_list[agent_name])
                    # print('reward', before_reward)
                    # print('state', state)
                    # print('before_state', self.before_state_list[agent_name])

                    self.data_collection_list[agent_num].store(self.before_state_list[agent_name],
                                                               self.before_action_list[agent_name], before_reward,
                                                               state, before_done_value)
                else:
                    self.collect_flag[agent_name] = True

                self.before_state_list[agent_name] = state

                self.before_action_list[agent_name] = action

                self.episode_reward_dict[agent_name] += before_reward

                if step >= self.min_update_step and step % self.update_step == 0:
                    for _ in range(self.update_step):
                        batch = self.data_collection_list[agent_num].sample_batch(
                            self.batch_size)  # random sample batch
                        self.critic_update(self.agent_list[agent_num], state=batch['state'], action=batch['action'],
                                           reward=batch['reward'], next_state=batch['next_state'],
                                           done_value=batch['done'], gamma=self.gamma)

                        self.actor_update(self.agent_list[agent_num], state=batch['state'], action=batch['action'],
                                          reward=batch['reward'], next_state=batch['next_state'], gamma=self.gamma)

                        self.soft_update(self.agent_list[agent_num].actor_target, self.agent_list[agent_num].actor,
                                         polyak=self.polyak)

                        self.soft_update(self.agent_list[agent_num].critic_target, self.agent_list[agent_num].critic,
                                         polyak=self.polyak)

                # step += 1

                if before_done:
                    # if step % 2000 == 0:
                    print(agent_name, self.episode_reward_dict[agent_name])
                if step >= self.min_update_step and step % self.save_step == 0:
                    self.agent_list[agent_num].save(actor_address='ac' + str(agent_num),
                                                    critic_address='cri' + str(agent_num))


class MAGAAlgorithms(MAAlgorithm):
    def __init__(
            self,
            agent_class_list,
            actor_update: None,
            discriminator_update: None,
            data_collection_list: [],
            env,
            train: None,
            lr_discriminator: None,
            actor_list: [],
            critic_list: [],
            discriminator_list: [],
            data_collection_args_dict: Dict[str, Dict[str, Any]] = None,
            critic_update=None,
            soft_update=None,
            polyak: float = 0.9,
            optimizer_actor: [torch.optim.Adam] = torch.optim.Adam,
            optimizer_critic: [torch.optim.Adam] = torch.optim.Adam,
            optimizer_discriminator: [torch.optim.Adam] = torch.optim.Adam,
            agent_args: Dict[str, Any] = None,
            lr_actor: float = 1e-2,
            lr_critic: float = 1e-2,
            buffer_size: int = 1e5,
            learning_rate: float = 4e-4,
            gamma: float = 0.95,
            batch_size: int = 512,
            tensorboard_log: str = None,
            render: bool = False,
            action_noise: float = 0.1,
            min_update_step: int = 1000,
            update_step: int = 100,
            start_steps: int = 10000,
    ):
        '''

        :param agent_class_list: agent class for every agent
        :param actor_update: update method for actor
        :param discriminator_update: update method for discriminator
        :param data_collection_list: replay buffer for every agent
        :param env: environment
        :param train: just flag no need
        :param lr_discriminator: learning rate for disc
        :param actor_list: actor object for every agent
        :param critic_list: critic object for every agent
        :param discriminator_list: discriminator object for every agent
        :param data_collection_args_dict: replay buffer args for every agent
        :param critic_update: update method for critic
        :param soft_update: no need
        :param polyak: no need
        :param optimizer_actor: optimizer
        :param optimizer_critic: ...
        :param optimizer_discriminator:...
        :param agent_args: no need
        :param lr_actor: learning rate for actor
        :param lr_critic: ...
        :param buffer_size: ...
        :param learning_rate: no need
        :param gamma: no need
        :param batch_size: ...
        :param tensorboard_log: no need
        :param render: ...
        :param action_noise: action noise for ddpg agent
        :param min_update_step: if step > min_update_step we start update
        :param update_step: amount of updating for every updating
        :param start_steps: no need
        '''
        super(MAGAAlgorithms, self).__init__(agent_class_list=agent_class_list,
                                             critic_update=critic_update,
                                             actor_update=actor_update,
                                             discriminator_update=discriminator_update,
                                             data_collection_list=data_collection_list,
                                             data_collection_args_dict=data_collection_args_dict,
                                             env=env,
                                             train=train,
                                             soft_update=soft_update,
                                             lr_discriminator=lr_discriminator,
                                             actor_list=actor_list,
                                             critic_list=critic_list,
                                             discriminator_list=discriminator_list,
                                             polyak=polyak,
                                             optimizer_actor=optimizer_actor,
                                             optimizer_critic=optimizer_critic,
                                             optimizer_discriminator=optimizer_discriminator,
                                             agent_args=agent_args,
                                             lr_actor=lr_actor,
                                             lr_critic=lr_critic,
                                             buffer_size=buffer_size,
                                             learning_rate=learning_rate,
                                             gamma=gamma,
                                             batch_size=batch_size,
                                             tensorboard_log=tensorboard_log,
                                             render=render,
                                             action_noise=action_noise,
                                             min_update_step=min_update_step,
                                             update_step=update_step,
                                             start_steps=start_steps,
                                             )

    def learn(self, num_train_step):
        '''
        training process.
        :param num_train_step:  total steps for training
        :return:
        '''
        for agent in self.agent_list:
            agent.actor.train()
            agent.discriminator.train()
        discriminator_flag = True
        step = 0
        self.collect_flag = {}
        self.before_state_list = {}
        self.before_action_list = {}
        self.episode_reward_dict = {}
        for agent_name in self.env.agents:
            self.episode_reward_dict[agent_name] = 0
        while step <= (num_train_step):
            # reset env

            self.env.reset()

            # for agent in self.env.agents:
            #
            #     self.env.agent_selection = agent
            #
            #     self.before_state_list[agent] = state
            #
            #     self.before_action_list[agent] = 0

            # init operation for training.
            for agent_name in self.env.agents:
                self.collect_flag[agent_name] = False
                self.writer.add_scalar('episode_reward' + str(agent_name), self.episode_reward_dict[agent_name],
                                       global_step=step * self.update_step)
                self.episode_reward_dict[agent_name] = 0
            # scan all agents and turn to next step automatically.
            for agent_name in self.env.agent_iter():
                # print(agent_name)
                # render or not
                if self.render:
                    self.env.render()
                # mapping the agent_name to number representing the agent order.
                agent_num = self.agent_name_mapping[agent_name]
                # get the s,r,done from last step.
                state, before_reward, before_done, _ = self.env.last()

                # print('x_state',state)
                # self.next_state_list[agent] = state
                # total_reward += reward

                # randomly decide acting as actor or demonstrator agent.
                rnd = np.random.random(1)
                if rnd > 0.5:
                    # acting as actor
                    action = self.agent_list[agent_num].choose_action(state, self.action_noise)
                    # this sample from actor
                    label = 0

                else:
                    # acting as demonstrator agent
                    action = self.data_collection_list[agent_num].demonstrator_agent.choose_action(state, 0)
                    # this sample from demonstrator.
                    # action = np.array([0,1])
                    label = 1

                if self.collect_flag[agent_name]:
                    # print('action', self.before_action_list[agent_name])
                    # print('reward', before_reward)
                    # print('state', state)
                    # print('before_state', self.before_state_list[agent_name])

                    # store experience from actor or demonstrator.
                    self.data_collection_list[agent_num].store(state,
                                                               action, label)

                else:
                    # just flag, dont care.
                    self.collect_flag[agent_name] = True
                # if one agent game over.
                if before_done:
                    action = None

                # print(action)

                # execute action for env
                self.env.step(action)

                # no need for gamail.
                before_done_value = 0 if before_done else 1

                self.before_state_list[agent_name] = state

                self.before_action_list[agent_name] = action

                self.episode_reward_dict[agent_name] += before_reward
                # if total steps > min_update_step and % update_step = 0
                # update the actor and discriminator.
                if step >= self.min_update_step and step % self.update_step == 0:
                    if discriminator_flag:
                        discriminator_flag = False
                        for i in range(self.update_step):
                            # sample data from replay buffer.
                            batch = self.data_collection_list[agent_num].sample_batch(
                                self.batch_size)  # random sample batch
                            # update actor
                            discriminator_rate = self.actor_update(self.agent_list[agent_num], state=batch['state'])

                            # discriminator_rate_list.append(discriminator_rate)
                            # self.discriminator_update(self.agent_list[agent_num], state=batch['state'],
                            #                          action=batch['action'],
                            #                          label=batch['label'])
                            self.writer.add_scalar('discriminator_rate' + str(agent_num), discriminator_rate,
                                                   global_step=step * self.update_step + i)

                    else:
                        discriminator_flag = True
                        for _ in range(self.update_step):
                            # sample data from replay buffer.
                            batch = self.data_collection_list[agent_num].sample_batch(
                                self.batch_size)  # random sample batch
                            # print(len(self.data_collection_list[agent_num]))
                            # update actor
                            # discriminator_rate = self.actor_update(self.agent_list[agent_num], state=batch['state'])
                            # discriminator_rate_list.append(discriminator_rate)
                            # update discriminator.
                            self.discriminator_update(self.agent_list[agent_num], state=batch['state'],
                                                      action=batch['action'],
                                                      label=batch['label'])

                step += 1

                # if before_done:
                #     # if step % 2000 == 0:
                #     # statistic the episode reward for every agent.
                #     print(agent_name, self.episode_reward_dict[agent_name])
