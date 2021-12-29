import copy
import os
from typing import Dict
from abc import ABC, abstractmethod
import gym
import numpy as np
import torch

from jueru.utils import scan_root
from jueru.weightinit import weight_init


class Agent(ABC):
    def __init__(
            self,
            functor_dict,
            optimizer_dict=None,
            lr_dict=None,
            init=True,
    ):

        # self.actor = Actor(input_dim = self.observation_space.n, output_dim = self.action_space.n)
        # self.critic = Critic(input_dim = self.observation_space.n, output_dim = self.action_space.n)
        # self.discriminator = Discriminator(input_dim = (self.observation_space.n + self.action_space.n))
        self.functor_dict = functor_dict
        self.lr_dict = lr_dict

        if not self.lr_dict:
            self.lr_dict = {}
            for functor_name, functor in self.functor_dict.items():
                self.lr_dict[functor_name] = 0
        if init:
            self.init()
        self.optimizer_dict = optimizer_dict
        if not self.optimizer_dict:
            self.optimizer_dict = {}
            for functor_name, functor in self.functor_dict.items():
                if not isinstance(self.functor_dict[functor_name], torch.Tensor):
                    self.optimizer_dict[functor_name] = torch.optim.Adam(params=functor.parameters(),
                                                                         lr=self.lr_dict[functor_name])
                else:
                    self.optimizer_dict[functor_name] = torch.optim.Adam(params=[functor],
                                                                         lr=self.lr_dict[functor_name])
        else:
            for functor_name, functor in self.functor_dict.items():
                self.optimizer_dict[functor_name] = self.optimizer_dict[functor_name](params=functor.parameters(),
                                                                                      lr=self.lr_dict[functor_name])

        for functor_name, functor in self.functor_dict.items():
            if 'target' in functor_name:
                self.functor_dict[functor_name] = copy.deepcopy(self.functor_dict[functor_name.split('_')[0]])
                for p in self.functor_dict[functor_name].parameters():
                    p.requires_grad = False

        # self.optimizer_discriminator = optimizer_critic(params = self.discriminator.parameters(), lr = lr_discriminator)

        # print(self.actor.limit_low)
        # print(self.actor.limit_high)


    def init(self):

        for functor_name, functor in self.functor_dict.items():

            if 'target' in functor_name:
                self.functor_dict[functor_name] = copy.deepcopy(self.functor_dict[functor_name.split('_')[0]])
            else:
                if not isinstance(self.functor_dict[functor_name], torch.Tensor):
                    self.functor_dict[functor_name].apply(weight_init)

    def save(self, address):
        os.makedirs(address, exist_ok=True)

        for functor_name, functor in self.functor_dict.items():
            torch.save(functor, os.path.join(address, functor_name))

        return

    @classmethod
    def load(cls, address):

        functor_dict = {}
        file_list, dir_list = scan_root(address)

        for file in file_list:

            functor_dict[os.path.split(file)[1]] = torch.load(file)

        agent = cls(functor_dict=functor_dict,
                    init=False,
                    )
        return agent

    @abstractmethod
    def predict(self, obs):
        """get action based on functor"""

class Sac_agent(Agent):

    def select_action(self, obs):
        with torch.no_grad():
            if isinstance(obs, Dict):
                for key in obs.keys():
                    obs[key] = torch.as_tensor(obs[key], dtype=torch.float32).unsqueeze(0)

                mu, _, _, _ = self.functor_dict['actor'](
                    obs, compute_pi=False, compute_log_pi=False
                )
            else:
                obs = torch.FloatTensor(obs)
                obs = obs.unsqueeze(0)
                mu, _, _, _ = self.functor_dict['actor'](
                    obs, compute_pi=False, compute_log_pi=False
                )

            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):

        with torch.no_grad():
            if isinstance(obs, Dict):
                for key in obs.keys():
                    obs[key] = torch.as_tensor(obs[key], dtype=torch.float32).unsqueeze(0)
                mu, pi, _, _ = self.functor_dict['actor'](obs, compute_log_pi=False)
            else:

                obs = torch.FloatTensor(obs)
                obs = obs.unsqueeze(0)
                mu, pi, _, _ = self.functor_dict['actor'](obs, compute_log_pi=False)
            print(pi)
            return pi.cpu().data.numpy().flatten()
    def predict(self, obs):
        return self.select_action(obs)

class DDPG_agent(Agent):

    def choose_action(self, observation, noise_scale):
        observation = observation.copy()
        with torch.no_grad():
            # print(observation)
            if isinstance(observation, Dict):
                for key in observation.keys():
                    observation[key] = torch.as_tensor(observation[key], dtype=torch.float32).unsqueeze(0)
                a = self.functor_dict['actor'](observation)
            else:
                a = self.functor_dict['actor'](torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))
            #a = self.functor_dict['actor'](observation)
            # print(a)
            a += noise_scale * np.random.randn(self.functor_dict['actor'].action_dim)
            a = a.squeeze(0).numpy()
        #print(self.functor_dict['actor'].limit_low)
        return np.clip(a, self.functor_dict['actor'].limit_low, self.functor_dict['actor'].limit_high).numpy()

    def predict(self, obs):
        return self.choose_action(obs, 0)

class DQN_agent(Agent):

    def choose_action_by_critic(self, observation):
        with torch.no_grad():
            a = np.argmax(self.functor_dict['critic'](torch.as_tensor(observation, dtype=torch.float32).reshape((1, -1))))

        return a

    def predict(self, obs):
        return self.choose_action_by_critic(obs).numpy()
