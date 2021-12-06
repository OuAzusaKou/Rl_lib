import copy

import gym
import numpy as np
import torch

from common.weightinit import weight_init


class Agent:
    def __init__(
            self,
            Actor=None,
            Critic=None,
            Discriminator=None,
            optimizer_actor=None,
            optimizer_critic=None,
            optimizer_discriminator=None,
            lr_actor: float = 1e-5,
            lr_critic: float = 1e-4,
            lr_discriminator: float = 1e-4,
            init=True,
    ):

        # self.actor = Actor(input_dim = self.observation_space.n, output_dim = self.action_space.n)
        # self.critic = Critic(input_dim = self.observation_space.n, output_dim = self.action_space.n)
        # self.discriminator = Discriminator(input_dim = (self.observation_space.n + self.action_space.n))
        self.actor = Actor
        self.critic = Critic
        self.discriminator = Discriminator
        if init:
            self.init()
        if optimizer_actor and self.actor:
            self.optimizer_actor = optimizer_actor(params=self.actor.parameters(), lr=lr_actor)
        if optimizer_critic and self.critic:
            self.optimizer_critic = optimizer_critic(params=self.critic.parameters(), lr=lr_critic)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        if optimizer_discriminator and self.discriminator:
            self.optimizer_discriminator = optimizer_discriminator(params=self.discriminator.parameters(),
                                                                   lr=lr_discriminator)
        if self.actor:
            for p in self.actor_target.parameters():
                p.requires_grad = False
        if self.critic:
            for p in self.critic_target.parameters():
                p.requires_grad = False

        # self.optimizer_discriminator = optimizer_critic(params = self.discriminator.parameters(), lr = lr_discriminator)

        # print(self.actor.limit_low)
        # print(self.actor.limit_high)

    def choose_action(self, observation, noise_scale):
        with torch.no_grad():
            # print(observation)
            a = self.actor(torch.as_tensor(observation, dtype=torch.float32).reshape((1, -1)))
            # print(a)
            a += noise_scale * np.random.randn(self.actor.action_dim)
            #print(a)
            # print(self.actor.limit_low)
        return np.clip(a, self.actor.limit_low, self.actor.limit_high)

    def choose_action_by_critic(self, observation):
        with torch.no_grad():

            a = np.argmax(self.critic(torch.as_tensor(observation, dtype=torch.float32).reshape((1, -1))))

        return a

    def init(self):
        if self.actor:
            self.actor.apply(weight_init)
        if self.critic:
            self.critic.apply(weight_init)
        if self.discriminator:
            self.discriminator.apply(weight_init)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

    def save(self, actor_address=None, critic_address=None, discriminator_address=None):
        if actor_address and self.actor:
            torch.save(self.actor, actor_address)
        if critic_address and self.critic:
            torch.save(self.critic, critic_address)
        if discriminator_address and self.discriminator:
            torch.load(self.discriminator, discriminator_address)
        return

    @classmethod
    def load(cls, actor_address=None, critic_address=None, discriminator_address=None):
        if actor_address:
            actor = torch.load(actor_address)
        else:
            actor = None
        if critic_address:
            critic = torch.load(critic_address)
        else:
            critic = None
        if discriminator_address:
            discriminator = torch.load(discriminator_address)
        else:
            discriminator = None
        agent = cls(Actor=actor,
                    Critic=critic,
                    Discriminator=discriminator,
                    init=False,
                    )
        return agent
