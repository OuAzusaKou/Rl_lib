import gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import spinup

env = gym.make("Pendulum-v1")

spinup.ddpg_pytorch(env_fn=env)