from typing import List, Type

import gym
import torch
from torch import nn

from jueru.utils import get_flattened_obs_dim, get_action_dim, gaussian_logprob, squash


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
  Create a multi layer perceptron (MLP), which is
  a collection of fully-connected layers each followed by an activation function.

  :param input_dim: Dimension of the input vector
  :param output_dim:
  :param net_arch: Architecture of the neural net
      It represents the number of units per layer.
      The length of this list is the number of layers.
  :param activation_fn: The activation function
      to use after each layer.
  :param squash_output: Whether to squash the output using a Tanh
      activation function
  :return:
  """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class CombinedExtractor(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, feature_dim = 128):
        super(CombinedExtractor, self).__init__()
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                # extractors[key] = CNNfeature_extractor(observation_space=subspace, features_dim=feature_dim)
                extractors[key] = FlattenExtractor(observation_space=subspace)
                total_concat_size += feature_dim
            elif key == "target":
                # Run through a simple MLP
                # extractors[key] = nn.Linear(subspace.shape[0], 16)
                extractors[key] = FlattenExtractor(observation_space=subspace)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        # for key, extractor in self.extractors.items():
        #     encoded_tensor_list.append(extractor(observations[key]))
        encoded_tensor_list.append(self.extractors['image'](torch.FloatTensor(observations['image'])/255))
        encoded_tensor_list.append(self.extractors['target'](observations['target']))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


class FlattenExtractor(nn.Module):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super(FlattenExtractor, self).__init__()

        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # print(observations)
        return self.flatten(observations)


class CNNfeature_extractor(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CNNfeature_extractor, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        #print(n_input_channels)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=3, stride=1,
            #                    padding=0, dilation=1, return_indices=False, ceil_mode=False),
            # nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        return self.linear(self.cnn(torch.FloatTensor(observations)/255))


class MLPfeature_extractor(nn.Module):
    def __init__(self, observation_space: gym.spaces.Space, net_arch=[], features_dim: int = 256, ):
        super(MLPfeature_extractor, self).__init__()
        input_dim = get_flattened_obs_dim(observation_space)

        self.modules_list = create_mlp(input_dim=input_dim,
                                 net_arch=net_arch,
                                 output_dim= features_dim
                                 )
        #print(self.modules_list)
        self.linear = nn.Sequential(*self.modules_list)
        #self.linear = nn.Sequential(nn.Linear(observation_space.shape[0], features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # print(observations)
        # print(observations)
        feature = self.linear(observations)
        return feature


class ddpg_actor(nn.Module):
    def __init__(self, action_space, feature_extractor, feature_dim):
        super(ddpg_actor, self).__init__()
        self.feature_extractor = feature_extractor

        self.action_dim = get_action_dim(action_space)

        self.linear = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, self.action_dim), nn.Tanh())

        self.limit_high = torch.tensor(action_space.high)
        #print(self.limit_high)
        self.limit_low = torch.tensor(action_space.low)
        # print(self.limit_low)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        return (self.limit_high + self.limit_low) / 2 + (self.limit_high - self.limit_low) / 2 * self.linear(
            self.feature_extractor(observations))


class ddpg_critic(nn.Module):
    def __init__(self, action_space, feature_extractor, feature_dim):
        super(ddpg_critic, self).__init__()
        self.feature_extractor = feature_extractor
        self.action_dim = get_action_dim(action_space)
        self.linear = nn.Sequential(nn.Linear(feature_dim + self.action_dim, 256), nn.ReLU(), nn.Linear(256, 256),
                                    nn.ReLU(), nn.Linear(256, 1))

    def forward(self, observations: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([self.feature_extractor(observations), action], dim=-1))


class gail_discriminator(nn.Module):
    def __init__(self, action_space, feature_extractor, feature_dim):
        super(gail_discriminator, self).__init__()
        self.feature_extractor = feature_extractor

        self.linear = nn.Sequential(nn.Linear(feature_dim + action_space.shape[0], 256), nn.ReLU(), nn.Linear(256, 256),
                                    nn.ReLU(), nn.Linear(256, 2), nn.Softmax())

    def forward(self, observations: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([self.feature_extractor(observations), action], dim=-1))


class dqn_critic(nn.Module):
    def __init__(self, action_space, feature_extractor, feature_dim):
        super(dqn_critic, self).__init__()
        self.feature_extractor = feature_extractor

        self.action_dim = action_space.n

        self.linear = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, self.action_dim))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # print(observations)
        return self.linear(
            self.feature_extractor(observations))


class Sac_actor(nn.Module):
    """MLP actor network."""

    def __init__(
            self, action_space, hidden_dim, feature_extractor,
            feature_dim, log_std_min, log_std_max
    ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.action_dim = get_action_dim(action_space)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.action_dim)
        )

        self.outputs = dict()

    def forward(
            self, obs, compute_pi=True, compute_log_pi=True
    ):
        obs = self.feature_extractor(obs)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Sac_critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
            self, action_space, feature_extractor, hidden_dim,
            feature_dim
    ):
        super().__init__()

        # self.encoder = make_encoder(
        #     encoder_type, obs_shape, encoder_feature_dim, num_layers,
        #     num_filters, output_logits=True
        # )
        self.feature_extractor = feature_extractor
        self.action_dim = get_action_dim(action_space)

        self.Q1 = QFunction(
            feature_dim, self.action_dim, hidden_dim
        )
        self.Q2 = QFunction(
            feature_dim, self.action_dim, hidden_dim
        )

        self.outputs = dict()

    def forward(self, obs, action):
        # detach_encoder allows to stop gradient propogation to encoder
        with torch.no_grad():
            obs = self.feature_extractor(obs)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        # self.outputs['q1'] = q1
        # self.outputs['q2'] = q2

        return q1, q2
