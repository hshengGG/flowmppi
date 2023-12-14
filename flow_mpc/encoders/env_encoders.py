import torch
from torch import nn
from torch.nn import functional as F
from flow_mpc.encoders.vae import VAE
from torch.distributions.normal import Normal


class BaseEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def encode(self, environment):
        raise NotImplementedError

    def get_ood_score(self, environment):
        raise NotImplementedError

    def reconstruct(self, z_env):
        raise NotImplementedError


class Encoder(nn.Module):

    def __init__(self, context_dim, z_env_dim):
        super(self).__init__()
        self.environment_image_embedding_size = 512
        # convolutons for environments image
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.fc_image_reduction = nn.Linear(2048, z_env_dim)
        self.act_fn = F.relu
        self.context_dim = context_dim

    def encode(self, environment):
        h_env = self.act_fn(self.conv1(environment))
        h_env = self.act_fn(self.conv2(h_env))
        h_env = self.act_fn(self.conv3(h_env)).reshape(-1, 2048)
        h_env = self.act_fn(self.fc_image_reduction(h_env))

        return {
            'h_environment', h_env
        }


class EnsembleEncoder(nn.Module):

    def __init__(self, context_dim, z_env_dim, num_ensembles):
        super().__init__()

        for _ in range(num_ensembles):
            self.nets.append(ConditioningNetwork(context_dim))

        self.nets = nn.ModuleList(self.nets)

    def encode(self, environment):
        h_env = []
        for net in self.nets:
            h_env.append(net(start, goal, environment))

        h_env = torch.stack(context, dim=0)

        raise NotImplementedError

    def get_ood_score(self, environment):
        raise NotImplementedError


class VAEEncoder(nn.Module):

    def __init__(self, context_dim, z_env_dim, voxels=False, flow_prior=None):
        super().__init__()
        self.z_env_dim = z_env_dim
        self.vae = VAE(z_env_dim, flow_prior=flow_prior, voxels=voxels)
        self.dummy_param = nn.Parameter(torch.empty(0))

    def encode(self, environment, z_env=None, reconstruct=True):
        if z_env is not None:
            return {
                'z_environment': z_env,
            }
        out = {}
        B = environment.shape[0]
        # Actually return variational lower bound to log_p_env
        if reconstruct:
            env_hat, z_env, latent_mu, latent_sigma = self.vae(environment)
            sq_diff = -(env_hat - environment) ** 2
            kl_term = self.vae.get_kl_divergence(z_env, latent_mu, latent_sigma)
            log_p_env = sq_diff.view(B, -1).sum(dim=1) - kl_term
            out['log_p_env'] = log_p_env
        else:
            z_env, _, _ = self.vae.encode(environment)
            out['log_p_env'] = None

        out['z_environment'] = z_env
        return out

    def reconstruct(self, z_env, N=1):
        # if N > 1:
        #    raise NotImplementedError("can't sample N>1 for vae")
        B = z_env.shape[0]
        # environment = self.vae.decoder(z_env)
        environment = self.vae.decoder(z_env.repeat(1, N, 1).reshape(B * N, -1))

        out = {
            'environments': environment.reshape(N, B, *environment.shape[1:]),
        }
        return out

    def sample(self, N=1):
        latent = self.vae.prior.sample(N).to(device=self.dummy_param.device)
        return {
            'environments': self.vae.decoder(latent)
        }
