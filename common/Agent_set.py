import copy
import os
from typing import Dict

import gym
import numpy as np
import torch

from common.utils import scan_root
from common.weightinit import weight_init


class Agent:
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
        if init:
            self.init()
        self.optimizer_dict = optimizer_dict
        if not self.optimizer_dict:
            self.optimizer_dict = {}
            for functor_name, functor in self.functor_dict.items():
                self.optimizer_dict[functor_name] = torch.optim.Adam(params=functor.parameters(),
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

    def choose_action(self, observation, noise_scale):
        with torch.no_grad():
            # print(observation)
            a = self.functor_dict['actor'](torch.as_tensor(observation, dtype=torch.float32).reshape((1, -1)))
            # print(a)
            a += noise_scale * np.random.randn(self.functor_dict['actor'].action_dim)
            # print(a)
            # print(self.actor.limit_low)
        return np.clip(a, self.functor_dict['actor'].limit_low, self.functor_dict['actor'].limit_high)

    def choose_action_by_critic(self, observation):
        with torch.no_grad():
            a = np.argmax(self.functor_dict['critic'](torch.as_tensor(observation, dtype=torch.float32).reshape((1, -1))))

        return a

    def init(self):

        for functor_name, functor in self.functor_dict.items():

            if 'target' in functor_name:
                self.functor_dict[functor_name] = copy.deepcopy(self.functor_dict[functor_name.split('_')[0]])
            else:
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
