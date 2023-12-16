import torch
from torch import nn
from flow_mpc.models.utils import PointGoalFcn, CollisionFcn

import time


class GenerativeModel(nn.Module):

    def __init__(self, dynamics, sigma, state_dim, control_dim):
        super().__init__()
        self.dynamics = dynamics
        self.dynamics_sigma = 0.00
        self.collision_fn = CollisionFcn()
        self.dx = state_dim
        self.du = control_dim
        self.combined_cost = False
        self.default_control_sigma = sigma
        self.isotropic_prior = True

    def get_smooth_prior_covariance(self, lengthscale, sigma, horizon):
        """
        lengthscale is (B x 1) parameter of lengthscale (could make different lengthscale for different control dim?)
        """
        B = lengthscale.shape[0]
        device = lengthscale.device
        t = torch.arange(0, horizon).reshape(-1).to(device=device)
        t_diff = t.unsqueeze(0) - t.unsqueeze(1)

        covar = torch.exp(
            -t_diff.unsqueeze(0)**2 / lengthscale.reshape(-1, 1, 1))  # B x T x T -> want to make it T x d x t x d
        covar = covar.reshape(B, horizon, horizon, 1, 1)
        covar = covar * torch.eye(self.du, device=device).reshape(1, 1, 1, self.du, self.du)
        covar = sigma.reshape(-1, 1, 1, 1, 1) * covar
        covar = covar.permute(0, 1, 3, 2, 4).reshape(B, horizon * self.du, horizon * self.du)
        return covar

    def get_prior(self, lengthscale, sigma, horizon):
        B = lengthscale.shape[0]
        if self.isotropic_prior:
            eye = torch.eye(horizon*self.du, horizon*self.du, device=sigma.device).expand(B, -1, -1)
            covar = sigma.reshape(B, 1, 1) * eye
        else:
            covar = self.get_prior_covariance(lengthscale, sigma, horizon)
        mu = torch.zeros(B, horizon * self.du, device=lengthscale.device)
        return torch.distributions.MultivariateNormal(mu, covariance_matrix=covar)

    def goal_log_likelihood(self, state, goal, vel_penalty=None):
        raise NotImplementedError

    def collision_log_likelihood(self, state, sdf, sdf_grad):
        raise NotImplementedError

    def forward(self, start, goal, sdf, sdf_grad, action_sequence, params=None, compute_costs=True,
                X=None):
        B, N, dx = start.shape
        _, _, H, du = action_sequence.shape
        assert dx == self.dx
        assert du == self.du

        if params is not None:
            control_sigma, control_lengthscale, vel_penalty = torch.chunk(params, chunks=3, dim=1)
        else:
            control_lengthscale = 1*torch.ones(B, device=action_sequence.device)
            control_sigma = self.default_control_sigma * torch.ones(B, device=action_sequence.device)
            vel_penalty = None

        # for numerical reasons let's constraint the actions
        action_sequence = torch.clamp(action_sequence, min=-1000, max=1000)
        if compute_costs:
            prior = self.get_prior(control_lengthscale, control_sigma, H)
            action_logprob = prior.log_prob(action_sequence.reshape(B, N, H*du).permute(1, 0, 2)).permute(1, 0)
        #prior = torch.distributions.Normal(0.0, 1.0)
        #action_logprob = prior.log_prob(action_sequence).sum(dim=[-2, -1])
        goal_logprob = 0
        collision_logprob = 0
        x_sequence = []
        x = start
        start = time.time()
        duration = 0
        start_time=time.time()
        for t in range(H):
            action = action_sequence[:, :, t]
            if X is None:
                x_mu = self.dynamics(x.reshape(-1, dx), action.reshape(-1, du)).reshape(B, N, dx)
                x = x_mu + self.dynamics_sigma * torch.randn_like(x_mu)
                x_sequence.append(x_mu)
            else:
                x = X[:, :, t]

            if compute_costs:
                if not self.combined_cost:
                    goal_logprob += self.goal_log_likelihood(x, goal, vel_penalty)
                    # collision_logprob += self.collision_log_likelihood(x, sdf, sdf_grad)
        end_time = time.time()
        print("Time elapsed for Horizon: %f" % (end_time-start_time))
        if X is None:
            X = torch.stack(x_sequence, dim=2)  # B x N x T x dx
        if compute_costs:
            if self.combined_cost:
                cost_logprob = self.total_logpcost(X, goal, sdf, sdf_grad, vel_penalty)
            else:
                goal_logprob += 10 * self.goal_log_likelihood(x, goal, vel_penalty)
                collision_logprob = self.collision_log_likelihood(X.reshape(B, -1, dx),
                                                                  sdf, sdf_grad).reshape(B, N, H).sum(dim=2)

                cost_logprob = collision_logprob + goal_logprob

            cost_logprob = torch.where(
                torch.logical_or(torch.isnan(cost_logprob), torch.isinf(cost_logprob)),
                -1e7 * torch.ones_like(cost_logprob),
                cost_logprob
            )
        else:
            cost_logprob, action_logprob = None, None
        # if N == 1:
        #    print(collision_logprob, goal_logprob)
        return cost_logprob, action_logprob, X
