import torch
from torch import nn


class BaseActionSampler(nn.Module):

    def __init__(self, context_net, environment_encoder, action_dimension, horizon, condition_on_cost=False):
        super().__init__()
        self.environment_encoder = environment_encoder
        self.context_net = context_net
        self.du = action_dimension
        self.H = horizon
        self.condition_on_cost = condition_on_cost

    def sample(self, start, goal, environment):
        raise NotImplementedError

    def likelihood(self, action_sequence, start, goal, environment):
        raise NotImplementedError

    def encode_environment(self, environment, z_environment=None, reconstruct=False):
        env_encoding = self.environment_encoder.encode(environment, z_env=z_environment, reconstruct=reconstruct)
        return env_encoding

    def condition(self, start, goal, environment, params=None, z_environment=None, reconstruct=False):
        env_encoding = self.encode_environment(environment, z_environment, reconstruct=reconstruct)
        key = 'h_environment'
        if 'h_environment' not in env_encoding.keys():
            key = 'z_environment'

        if self.condition_on_cost:
            cost_params = params
        else:
            cost_params = None
        context = self.context_net(start, goal, env_encoding[key], params=cost_params)

        context_encoding = {
            'context': context
        }
        context_encoding.update(env_encoding)
        return context_encoding

    def get_environment_ood_score(self, environment):
        return self.environment_encoder.get_ood_score(environment)
