import torch
from torch import nn
from .base import BaseActionSampler
from torch.distributions.normal import Normal

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal, DiagonalNormal, StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, \
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform, AffineCouplingTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
from nflows.transforms.lu import LULinear
from nflows.nn.nets import ResidualNet
from nflows.utils import torchutils

def build_nvp_flow(flow_dim, context_dim, flow_length, use_autoregressive=False):
    def create_transform_net(in_features, out_features):
        net = ResidualNet(in_features, out_features,
                          hidden_features=64,
                          context_features=context_dim,
                          use_batch_norm=True)
        return net

    base_dist = StandardNormal(shape=[flow_dim])
    transforms = []
    for _ in range(flow_length):
        if use_autoregressive:
            # transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=flow_dim,
            #                                                                          context_features=context_dim,
            #                                                                          hidden_features=64,
            #                                                                          tails='linear',
            #                                                                          tail_bound=4))

            transforms.append(MaskedAffineAutoregressiveTransform(features=flow_dim,
                                                                  context_features=context_dim,
                                                                  hidden_features=64)
                              )
            transforms.append(ReversePermutation(features=flow_dim))
        else:
            # mask = torchutils.create_mid_split_binary_mask(flow_dim)
            mask = torchutils.create_random_binary_mask(flow_dim)
            # transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask=mask,
            #                                                              transform_net_create_fn=create_transform_net,
            #                                                              tails='linear', tail_bound=100))
            #
            # transforms.append(LULinear(features=flow_dim))
            transforms.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=create_transform_net))
            # transforms.append(RandomPermutation(features=flow_dim))

    return Flow(CompositeTransform(transforms), base_dist)


class FlowActionSampler(BaseActionSampler):

    def __init__(self, context_net, environment_encoder, action_dimension, horizon, flow_length, flow_type='nvp',
                 condition_on_cost=False):
        super().__init__(context_net, environment_encoder, action_dimension, horizon, condition_on_cost)
        flow_dim = self.du * self.H
        if flow_type == 'ffjord':
            self.flow = build_ffjord(flow_dim, self.context_net.context_dim, 1)
        elif flow_type == 'otflow':
            self.flow = OTFlow(flow_dim, 64, 2, self.context_net.context_dim)
        else:
            self.flow = build_nvp_flow(flow_dim, self.context_net.context_dim, flow_length, False)
        self.flow_type = flow_type
        self.register_buffer('prior_mu', torch.tensor(0.0))
        self.register_buffer('prior_scale', torch.tensor(1.0))

    def sample(self, start, goal, environment, cost_params=None,
               N=1, z_environment=None, reconstruct=False, z_only=False):
        B, _ = start.shape
        context_net_out = self.condition(start, goal, environment, cost_params, z_environment, reconstruct=reconstruct)
        context = context_net_out['context']
        B, H = context.shape
        u = self.flow.sample(num_samples=N, context=context)
        context_n_samples = context.unsqueeze(1).repeat(1, N, 1).reshape(N * B, -1)
        log_qu = self.flow.log_prob(u.reshape(B * N, -1).detach(), context=context_n_samples)

        context_net_out['Wj'] = None
        context_net_out['reg'] = None

        # we will replace nans with zeros
        u[torch.isnan(u)] = 0.0
        u[torch.isinf(u)] = 0.0
        # clip u so not super large
        u = torch.clamp(u, -100, 100)

        return u.reshape(B, N, self.H, self.du), log_qu.reshape(B, N), context_net_out

    def likelihood(self, u, start, goal, environment, cost_params=None, z_environment=None, reconstruct=False):
        # assume that we may have multiple samples per environments, i.e. u is shape N x B x H x du
        B, N, H, du = u.shape
        assert H == self.H
        assert du == self.du
        context_dict = self.condition(start, goal, environment, params=cost_params,
                                      z_environment=z_environment, reconstruct=reconstruct)
        context = context_dict['context']
        context_n_samples = context.unsqueeze(1).repeat(1, N, 1).reshape(N * B, -1)

        # out = self.flow(u.reshape(N * B, H * du), logpx=log_qu, context=context_n_samples, reverse=True)
        log_qu = self.flow.log_prob(u.reshape(N * B, H * du), context=context_n_samples)

        if self.training and self.flow_type == 'ffjord':
            context_dict['reg'] = self.flow.chain[0].regularization_states[1:]
        else:
            context_dict['reg'] = None

        return log_qu.reshape(B, N), context_dict

    def sample_w_peturbation(self, start, goal, environment, cost_params=None,
                             N=1, sigma=1, z_environment=None,
                             reconstruct_env=False):
        B, _ = start.shape
        with torch.no_grad():
            u, _, context_dict = self.sample(start, goal, environment, cost_params,
                                             z_environment=z_environment, N=N, reconstruct=False,
                                             z_only=True)
        # u = u.detach()
        # Note that we are sampling from a peturbed version of the qU, not qU itself.
        # We denote the peturbed distribution pU
        # We need prior weights which are qU / pU
        # pU is defined by p(U|U') Normal
        # and U' is distributed qU' with the flow
        # p_peturbation = Normal(loc=self.mean, scale=self.scale*sigma)
        peturbed_u = u + sigma * torch.randn_like(u)  # p_peturbation.sample(sample_shape=u.shape)
        log_qu, context_dict = self.likelihood(peturbed_u, start, goal, environment,
                                               cost_params=cost_params,
                                               z_environment=z_environment,
                                               reconstruct=reconstruct_env)

        if False:
            epsilon = peturbed_u.transpose(0, 1).reshape(B,
                                                         1, N, -1) - u.transpose(0, 1).reshape(B, N, 1, -1)
            # estimate pU via expectations
            log_pu = p_peturbation.log_prob(epsilon).sum(dim=-1)
            log_pu -= torch.max(log_pu)
            p_u = log_pu.exp().mean(dim=1).transpose(0, 1)
            p_u /= torch.sum(p_u, dim=0)
            q_u = (log_qu - torch.max(log_qu)).exp()
            q_u /= torch.sum(q_u, dim=0)
            sample_weights = q_u / (p_u + 1e-15)
            sample_weights /= torch.sum(sample_weights, dim=0) + 1e-6
            context_dict['Wj'] = sample_weights
        else:
            context_dict['Wj'] = None
            context_dict['reg'] = None

        return peturbed_u, log_qu, context_dict

    def reconstruct(self, z, start, goal, environment, cost_params=None, z_environment=None, reconstruct_env=False,
                    z_only=False):
        B, N, dz = z.shape
        assert dz == self.H * self.du
        context_net_out = self.condition(start, goal, environment, cost_params,
                                         z_environment,
                                         reconstruct=reconstruct_env)
        context = context_net_out['context']
        B, H = context.shape
        # Unfortunately we have to duplicate the context for each sample - there appears to be no way around this
        context_n_samples = context.unsqueeze(1).repeat(1, N, 1).reshape(B * N, H)
        prior = Normal(self.prior_mu, self.prior_scale)
        log_qz = torch.zeros(B * N, device=z.device)
        u,_ = self.flow._transform.inverse(z.reshape(N * B, -1), context=context_n_samples)
        
        # u, delta_log_qu = out[:2]
        if self.training and self.flow_type == 'ffjord':
            context_net_out['reg'] = self.flow.chain[0].regularization_states[1:]
        else:
            context_net_out['reg'] = None
        context_net_out['Wj'] = None
        log_qu = prior.log_prob(z.reshape(N * B, -1)).sum(dim=1)  # + delta_log_qu
        return u.reshape(B, N, self.H, self.du), log_qu.reshape(B, N), context_net_out

    def forward(self, start, goal, environment, cost_params=None, N=1, sigma=None, reconstruct=False,
                z_environment=None):
        if sigma is None:
            return self.sample(start, goal, environment, cost_params=cost_params, N=N, reconstruct=reconstruct,
                               z_environment=z_environment)
        return self.sample_w_peturbation(start, goal, environment, cost_params=cost_params,
                                         N=N, sigma=sigma, reconstruct_env=reconstruct,
                                         z_environment=z_environment)

    def transform_to_noise(self, u, start, goal, environment, cost_params=None, z_environment=None, reconstruct=False):
        # assume that we may have multiple samples per environments, i.e. u is shape N x B x H x du
        B, N, H, du = u.shape
        assert H == self.H
        assert du == self.du
        context_dict = self.condition(start, goal, environment, params=cost_params,
                                      z_environment=z_environment, reconstruct=reconstruct)
        context = context_dict['context']
        context_n_samples = context.unsqueeze(1).repeat(1, N, 1).reshape(N * B, -1)

        z = self.flow.transform_to_noise(u.reshape(N * B, -1), context_n_samples)
        
        return z.reshape(B, N, -1), context_dict
