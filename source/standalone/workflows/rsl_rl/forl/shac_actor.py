from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
# from forl.models import model_utils
from torch.distributions.normal import Normal


# class ActorDeterministicMLP(nn.Module):
#     def __init__(
#         self,
#         obs_dim: int,
#         action_dim: int,
#         units: List[int],
#         activation_class: Type = nn.ELU,
#         init_gain: float = 2.0**0.5,
#     ):
#         super().__init__()

#         self.layer_dims = [obs_dim] + units + [action_dim]

#         if isinstance(activation_class, str):
#             activation_class = eval(activation_class)
#         self.activation_class = activation_class

#         init_ = lambda m: model_utils.init(
#             m,
#             lambda x: nn.init.orthogonal_(x, init_gain),
#             lambda x: nn.init.constant_(x, 0),
#         )

#         modules = []
#         for i in range(len(self.layer_dims) - 1):
#             modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
#             if i < len(self.layer_dims) - 2:
#                 modules.append(self.activation_class())
#                 modules.append(nn.LayerNorm(self.layer_dims[i + 1]))

#         self.actor = nn.Sequential(*modules)

#         self.action_dim = action_dim
#         self.obs_dim = obs_dim

#     def forward(self, observations, deterministic=False):
#         return self.actor(observations)


class ActorStochasticMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        units: List[int],
        activation_class: Type = nn.ELU,
        init_gain: float = 1.0,
        init_logstd: float = -1.0,
        normalizer=None,
    ):
        super().__init__()

        self.layer_dims = [obs_dim] + units + [action_dim]

        if isinstance(activation_class, str):
            activation_class = eval(activation_class)
        self.activation_class = activation_class

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(self.activation_class())
                modules.append(nn.LayerNorm(self.layer_dims[i + 1]))
            else:
                modules.append(nn.Identity())

        self.mu_net = nn.Sequential(*modules)

        self.normalizer = normalizer

        self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32) * init_logstd)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        for param in self.parameters():
            param.data *= init_gain

    def get_logstd(self):
        return self.logstd

    # def forward(self, obs, deterministic=False):
    #     # print("Obeservations:")
    #     # print(obs)
    #     mu = self.mu_net(obs)
    #     # print(f"deterministic actions unnormalized: {mu}")
    #     # print("Mu:")
    #     # print(mu)

    #     if deterministic:
    #         return mu
    #     else:
    #         std = self.logstd.exp()
    #         # print("std:")
    #         # print(std)
    #         dist = Normal(mu, std)
    #         sample = dist.rsample()
    #         return sample
        
    def forward(self, obs):
        if self.normalizer is not None:
            mu = self.mu_net(self.normalizer.normalize(obs))
        else:
            mu = self.mu_net(obs)
        # print(f"deterministic actions unnormalized: {mu}")
        # print("Mu:")
        # print(mu)

        return mu


    def forward_with_dist(self, obs, deterministic=False):
        mu = self.mu_net(obs)
        std = self.logstd.exp()

        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std

    def evaluate_actions_log_probs(self, obs, actions):
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)

        return dist.log_prob(actions)
