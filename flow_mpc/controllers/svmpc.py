import torch

from gpytorch.kernels import RBFKernel
import torch.distributions as dist

from torch.optim import Adam
from torch.functorch import grad


def bw_median(x: torch.Tensor, y: torch.Tensor = None, bw_scale: float = 1.0, tol: float = 1.0e-5) -> torch.Tensor:
    if y is None:
        y = x.detach().clone()
    pairwise_dists = squared_distance(x, y).detach()
    h = torch.median(pairwise_dists)
    # TODO: double check which is correct
    # h = torch.sqrt(0.5 * h / torch.tensor(x.shape[0] + 1)).log()
    h = torch.sqrt(0.5 * h) / torch.tensor(x.shape[0] + 1.0).log()
    return bw_scale * h.clamp_min_(tol)


def squared_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Computes squared distance matrix between two arrays of row vectors.
    Code originally from:
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(
        x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
    ).add_(x1_norm)
    return res.clamp(min=0)  # avoid negative distances due to num precision


class MoG:
    def __init__(self, means, sigma, weights=None, td="cpu"):
        if weights is None:
            weights = torch.ones(means.shape[0], device=td) / means.shape[0]
        self.means = means.detach()
        mix_d = torch.distributions.Categorical(weights)

        comp_d = torch.distributions.Independent(
            torch.distributions.Normal(self.means, sigma * torch.ones(means.shape, device=td)), 2
        )
        self.mixture = torch.distributions.MixtureSameFamily(mix_d, comp_d)
        self.grad_log_prob = grad(self.log_prob)

    def sample(self, n=None):
        return self.mixture.sample((n,)) if n is not None else self.mixture.sample()

    def log_prob(self, x):
        return self.mixture.log_prob(x)


def d_log_qU(U_samples, U_mean, U_sigma):
    return (U_samples - U_mean) / U_sigma ** 2


class FactorizedKernel:
    """
        Factorized kernel for effective use on trajectories
        this assumes action sequence obeys following cnditional independence structure
        p(u) = p(u1)p(u2|u1)p(u3|u2)...

    """

    def __init__(self, horizon, du, device):
        self.H = horizon
        self.du = du
        # Based on this factorization we will have 2H - 1 factors
        self.kernel = RBFKernel().to(device=device)
        self.lengthscale = 1

    def __call__(self, U):
        """
            Note -- U must have requires grad set to true
            U will be N x H x du tensor --
        """
        ## U will be N x H x d
        self.kernel.lengthscale = self.lengthscale

        # Get set of factors
        U_reshaped = U.transpose(1, 0)  # reshape to be H x N x du

        # first factor based on unary potentials
        Kxx_unary = self.kernel(U_reshaped, U_reshaped.clone().detach()).evaluate()

        # next we'll do this offset
        Kxx_pairwise = self.kernel(U_reshaped[:-1], U_reshaped[1:]).evaluate()

        # Sum up the kernels -- divide by horizon
        Kxx = (Kxx_unary.sum(dim=0) + Kxx_pairwise.sum(dim=0)) / self.H

        grad_k = torch.autograd.grad(Kxx.sum(), U)[0]

        return Kxx, -grad_k


class SVMPC:

    def __init__(self, cost, dx, du, horizon, num_particles, samples_per_particle,
                 lr=0.01, lambda_=1.0, sigma=1.0, iters=1,
                 control_constraints=None,
                 device='cuda:0', action_transform=None, flow=False):
        """

            SVMPC
            optionally uses flow to represent posterior q(U) --
            samples maintained in U space. q(U) is mixture of Gaussians, in Flow space

        """
        self.dx = dx
        self.du = du
        self.H = horizon
        self.cost = cost
        self.control_constraints = control_constraints
        self.device = device
        self.M = num_particles
        self.N = samples_per_particle
        self.lr = lr
        # self.kernel = RBFKernel().to(device=device)
        self.kernel = FactorizedKernel(horizon=horizon, du=du, device=device)
        self.lambda_ = lambda_
        self.sigma = sigma
        self.iters = iters
        self.warmed_up = False
        self.action_transform = action_transform
        self.flow = flow

        # sample initial actions
        self.U = torch.randn(self.M, self.H, self.du, device=self.device)
        self.U.requires_grad = True

        self.weights = torch.ones(self.M, device=self.device) / self.M
        self.prior = MoG(weights=self.weights, means=self.U, sigma=self.sigma, td=self.device)

        self.optim = Adam([self.U], lr=self.lr)
        self.reset()

    @property
    def best_K_U(self):
        return self.U.detach()

    def step(self, state):
        self.U.requires_grad = True
        self.optim = Adam([self.U], lr=self.lr)
        if self.warmed_up:
            for _ in range(self.iters):
                self.update_particles(state)
        else:
            for _ in range(25):
                self.update_particles(state)
            self.warmed_up = True

        with torch.no_grad():
            # compute costs
            costs = self.cost(state, self.U)

            # Update weights
            self.weights = torch.softmax(-costs / self.lambda_, dim=0)
            out_U = self.U[torch.argmax(self.weights)].clone()

            # shift actions & psterior
            self.U = torch.roll(self.U, -1, dims=1)
            self.U[:, -1] = 0 * self.sigma * torch.randn(self.M, self.du, device=self.device)

            # Update prior with weights
        self.prior = MoG(weights=self.weights, means=self.U, sigma=self.sigma, td=self.device)

        return out_U.detach()

    def update_particles(self, state):
        # first N actions from each mixture
        with torch.no_grad():
            noise = torch.randn(self.N, self.M, self.H, self.du, device=self.device)
            U_samples = self.sigma * noise + self.U.unsqueeze(0)

            # Evaluate cost of action samples - evaluate each action set N times
            costs = self.cost(state, U_samples.reshape(-1, self.H, self.du)).reshape(self.N, self.M)
            weights = torch.softmax(-costs / self.lambda_, dim=0)

            bw = bw_median(self.U.flatten(1, -1), self.U.flatten(1, -1), 1)
            self.kernel.lengthscale = bw

        Kxx, grad_k = self.kernel(self.U)
        prior_ll = self.prior.log_prob(self.U)
        grad_prior = torch.autograd.grad(prior_ll.sum(), self.U)[0]

        # u_prior_grad =
        with torch.no_grad():
            grad_lik = d_log_qU(U_samples, self.U.unsqueeze(0), self.sigma)
            grad_lik = (weights.reshape(self.N, self.M, 1, 1) * grad_lik).sum(dim=0)

            if grad_prior is not None:
                grad_lik = grad_lik + grad_prior

            phi = grad_k + torch.tensordot(Kxx, grad_lik, 1) / self.M
            # self.U = self.U + self.lr * phi
            self.optim.zero_grad()

            self.U.grad = -phi
            # torch.nn.utils.clip_grad_norm_(self.U, 1)
            self.optim.step()

    def reset(self):
        # sample initial actions
        self.U = torch.randn_like(self.U)
        self.U.requires_grad = True
        self.warmed_up = False
        self.weights = torch.ones(self.M, device=self.device) / self.M
        self.prior = MoG(weights=self.weights, means=self.U, sigma=self.sigma, td=self.device)
