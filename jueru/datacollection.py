from collections import namedtuple
from random import random
from typing import Union

import numpy as np
#import stable_baselines3
#Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
#import spinup
import torch
from torch.utils.data import Dataset

from jueru.utils import get_obs_shape, get_action_dim
from gym import spaces


class Replay_buffer:
    '''single agent replay_buffer'''
    def __init__(self, env, size, device: Union[torch.device, str] = "auto"):
        self.obs_shape = get_obs_shape(env.observation_space)
        self.action_dim = get_action_dim(env.action_space)
        self.device = device
        self.state_buf = np.zeros((int(size), *self.obs_shape), dtype=env.observation_space.dtype)
        self.next_state_buf = np.zeros((int(size), *self.obs_shape), dtype=env.observation_space.dtype)

        self.action_buf = np.zeros((int(size), self.action_dim), dtype=env.action_space.dtype)
        self.reward_buf = np.zeros((int(size), 1), dtype=np.float32)
        self.done_buf = np.zeros((int(size), 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.state_buf[int(self.ptr)] = np.array(obs).copy()
        self.next_state_buf[int(self.ptr)] = np.array(next_obs).copy()
        self.action_buf[int(self.ptr)] = np.array(act).copy()
        self.reward_buf[int(self.ptr)] = np.array(rew).copy()
        self.done_buf[int(self.ptr)] = np.array(done).copy()
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):

        batch_size = min(batch_size, self.size)

        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.state_buf[idxs],
                     next_state=self.next_state_buf[idxs],
                     action=self.action_buf[idxs],
                     reward=self.reward_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v.copy(), dtype=torch.float32).to(self.device) for k,v in batch.items()}

    def __len__(self):
        return len(self.state_buf)

class Dict_Replay_buffer:
    '''single agent dict_replay_buffer'''
    def __init__(self, env, size, device: Union[torch.device, str] = "auto"):
        self.obs_shape = get_obs_shape(env.observation_space)
        self.action_dim = get_action_dim(env.action_space)
        self.device = device
        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"

        self.state_buf = {
            key: np.zeros((int(size), *_obs_shape)) for key, _obs_shape in
            self.obs_shape.items()
        }
        self.next_state_buf = {
            key: np.zeros((int(size), *_obs_shape)) for key, _obs_shape in
            self.obs_shape.items()
        }

        self.action_buf = np.zeros((int(size), self.action_dim), dtype=env.action_space.dtype)
        self.reward_buf = np.zeros((int(size), 1), dtype=np.float32)
        self.done_buf = np.zeros((int(size), 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        for key in self.state_buf.keys():
            self.state_buf[key][int(self.ptr)] = np.array(obs[key]).copy()
        for key in self.next_state_buf.keys():
            self.next_state_buf[key][int(self.ptr)] = np.array(next_obs[key]).copy()

        self.action_buf[int(self.ptr)] = np.array(act).copy()
        self.reward_buf[int(self.ptr)] = np.array(rew).copy()
        self.done_buf[int(self.ptr)] = np.array(done).copy()
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        #print(self.ptr)
        #print(self.size)

    def sample_batch(self, batch_size=32):

        batch_size = min(batch_size, self.size)

        idxs = np.random.randint(0, self.size, size=batch_size)
        state = {}
        next_state = {}
        for key, obs in self.state_buf.items():
            state[key] = torch.as_tensor(obs[idxs].copy(), dtype=torch.float32).to(self.device)
        for key, obs in self.next_state_buf.items():
            next_state[key] = torch.as_tensor(obs[idxs].copy(), dtype=torch.float32).to(self.device)
        batch = dict(state=state,
                     next_state=next_state,
                     action=torch.as_tensor(self.action_buf[idxs].copy(), dtype=torch.float32).to(self.device),
                     reward=torch.as_tensor(self.reward_buf[idxs].copy(), dtype=torch.float32).to(self.device),
                     done=torch.as_tensor(self.done_buf[idxs].copy(), dtype=torch.float32)).to(self.device)
        #print(batch)
        return batch

    def __len__(self):
        return len(self.state_buf)

class SignleReplay_buffer:
    '''repaly_buffer for multi-agent algorithms '''
    def __init__(self, observation_space, action_space, size):
        '''

        :param observation_space: env observation space
        :param action_space: env action_space
        :param size: buffer size
        '''
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.state_buf = np.zeros((int(size), *self.obs_shape), dtype=np.float32)
        self.next_state_buf = np.zeros((int(size), *self.obs_shape), dtype=np.float32)
        self.action_buf = np.zeros((int(size), self.action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((int(size), 1), dtype=np.float32)
        self.done_buf = np.zeros((int(size), 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
    def store(self, obs, act, rew, next_obs, done):
        '''
        store the experience to replay buffer
        :param obs: state
        :param act: action
        :param rew: reward
        :param next_obs: next_state
        :param done: done
        :return:
        '''
        self.state_buf[int(self.ptr)] = obs
        self.next_state_buf[int(self.ptr)] = next_obs
        self.action_buf[int(self.ptr)] = act
        self.reward_buf[int(self.ptr)] = rew
        self.done_buf[int(self.ptr)] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        '''
        sample
        :param batch_size:
        :return:
        '''
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.state_buf[idxs],
                     next_state=self.next_state_buf[idxs],
                     action=self.action_buf[idxs],
                     reward=self.reward_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def __len__(self):
        return len(self.state_buf)


class GAIL_DataSet:
    '''data collection(replay buffer) for GAIL or MAGAIL'''
    def __init__(self, observation_space, action_space, size, demonstrator_agent, demonstrator_traj = None):
        '''

        :param observation_space: env observation space
        :param action_space: env action_space
        :param size: buffer size
        :param demonstrator_agent: generator of the demonstrator (optional)
        :param demonstrator_traj: fixed generator (optional)
        '''
        self.demonstrator_agent = demonstrator_agent
        self.demonstrator_traj = demonstrator_traj
        self.state_buf = np.zeros((int(size), *observation_space.shape), dtype=np.float32)
        self.action_buf = np.zeros((int(size), *action_space.shape), dtype=np.float32)
        self.label_buf = np.zeros((int(size), 1), dtype=np.int32)

        self.ptr, self.size, self.max_size = 0, 0, size
    def store(self, obs, act , label):
        '''

        :param obs: state
        :param act: action
        :param label: label 0 for data from actor, 1 for data from demonstrator.
        :return:
        '''
        if (obs is None) or (act is None) or (label is None):
            return
        self.state_buf[int(self.ptr)] = obs
        self.action_buf[int(self.ptr)] = act
        self.label_buf[int(self.ptr)] = label

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

        return

    def __len__(self):
        return self.size

    def sample_batch(self, batch_size=32):
        '''

        :param batch_size:
        :return: a dict consist of {state: ,action:, label:,}
        for trainning
        '''
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.state_buf[idxs],
                     action=self.action_buf[idxs],
                     label=self.label_buf[idxs],
                     )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}




