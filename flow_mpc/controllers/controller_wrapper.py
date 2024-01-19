import torch
from torch import optim
import numpy as np
from flow_mpc.controllers.mppi import MPPI
#from flow_mpc.controllers.svmpc import SVMPC
from flow_mpc.controllers.icem import ICEM

from flow_mpc.trainer import SVIMPC_LossFcn

import matplotlib.pyplot as plt
from flow_mpc.visualisation import add_trajectory_to_axis
import matplotlib
import time
import csv

#matplotlib.use('tkAgg')


class MPCController:

    def __init__(self, generative_model,
                 horizon,
                 action_dim,
                 state_dim,
                 config,
                 control_constraints=None,
                 N=10,
                 device='cuda:0',
                 action_sampler=None,
                 sample_nominal=False,
                 use_true_grad=False,
                 use_vae=True
                 ):

        project_sample_fraction = 2
        # only project if there is an action sampler
        if action_sampler is not None:
            self.project = config['project']
        else:
            self.project = False

        self.use_true_grad = use_true_grad
        self.sample_nominal = sample_nominal
        if sample_nominal and action_sampler is None:
            raise ValueError('Must provide action sampler for sampling nominal MPPI trajectory')

        self.action_sampler = action_sampler
        self.generative_model = generative_model

        flow = False
        if action_sampler is not None and not sample_nominal:
            flow = True
        Nsamples = N
        N_project_samples = 0
        if 'mppi' in config['name']:
            if self.project:
                N_project_samples = N // project_sample_fraction
            Nsamples = (N - N_project_samples) // config['iters']

            self.controller = MPPI(self.cost, state_dim, action_dim, horizon, Nsamples, config['lambda'],
                                   config['sigma'], control_constraints=control_constraints, device=device,
                                   action_transform=self.action_transform, flow=flow, iterations=config['iters'])
        
        elif 'icem' in config['name']:
            if self.project:
                N_project_samples = N // project_sample_fraction
            Nsamples = (N - N_project_samples) // config['iters']
            K = int(config['elite_fraction'] * Nsamples)
            self.controller = ICEM(self.cost, state_dim, action_dim, horizon, Nsamples, K,
                                   alpha=config['momentum'], noise_beta=config['noise_param'],
                                   sigma=config['sigma'], elites_keep_fraction=config['kept_elites'],
                                   iterations=config['iters'], control_constraints=None, device=device,
                                   action_transform=self.action_transform, flow=flow)
        
        self.N = N_project_samples
        self.dx = state_dim
        self.du = action_dim
        self.H = horizon
        self.device = device
        self.sdf = torch.zeros((1, 64, 64), device=device)
        self.goal = torch.zeros((1, 3), device=device)
        self.N = N // project_sample_fraction
        self.use_vae = use_vae
        self.cost_params = None
    '''
        elif 'svmpc' in config['name']:
            if flow:
                raise NotImplementedError()

            samples_per_particle = N // (config['particles'] * config['iters'])
            self.controller = SVMPC(self.cost, state_dim, action_dim, horizon=horizon,
                                    num_particles=config['particles'],
                                    samples_per_particle=samples_per_particle,
                                    lambda_=config['lambda'], sigma=config['sigma'], lr=config['lr'],
                                    iters=config['iters'], device=device,
                                    action_transform=None, flow=False)
    '''

    def cost(self, x, U):
        N, H, du = U.shape

        log_pCost, log_pU, X = self.generative_model(
            x.unsqueeze(0).repeat(1, N, 1),
            self.goal.unsqueeze(0).repeat(1, N, 1),
            self.sdf,
            None, U.unsqueeze(0), self.cost_params)

        if False:
            fig, ax = plt.subplots()
            add_trajectory_to_axis(ax, x[0].detach().cpu().numpy(),
                                   self.goal[0].detach().cpu().numpy(),
                                   X[0].detach().cpu().numpy(),
                                   self.raw_sdf)
            plt.show()

        return -(log_pCost + log_pU).squeeze(0)

    def action_transform(self, x, Z, reverse=False, return_logprob=False):
        if self.action_sampler is None or self.sample_nominal:
            return Z
        with torch.no_grad():
            N = Z.shape[0]
            if reverse:
                U, _ = self.action_sampler.transform_to_noise(Z.reshape(1, -1, self.H, self.du),
                                                              x.reshape(1, self.dx),
                                                              self.goal[0].reshape(1, -1),
                                                              cost_params=self.cost_params,
                                                              environment=None,
                                                              z_environment=self.z_env[0].unsqueeze(0)
                                                              )

            else:
                U, logpU, _ = self.action_sampler.reconstruct(Z.reshape(1, N, self.H * self.du),
                                                              x.reshape(1, self.dx),
                                                              self.goal[0].reshape(1, -1),
                                                              cost_params=self.cost_params,
                                                              environment=None,
                                                              z_environment=self.z_env[0].unsqueeze(0)
                                                              )

            if return_logprob:
                return U.reshape(N, self.H, self.du), logpU.reshape(-1)

        return U.reshape(N, self.H, self.du)

    def update_environment(self, sdf, sdf_grad=None):
        self.sdf = torch.from_numpy(sdf).to(device=self.device).unsqueeze(0).unsqueeze(1)
        self.raw_sdf = sdf
        self.normalised_sdf = torch.where(self.sdf < 0, self.sdf / 1000.0,
                                          self.sdf)  # + 0.02 * torch.randn_like(self.sdf)

        # self.sdf *= 2
        # torch.where(self.sdf < 0, self.sdf, self.sdf)
        # self.sdf = torch.where(self.sdf < 0, -1e4, 0.0)
        self.sdf = self.normalised_sdf
        # if self.action_sampler is not None:
        #    self.action_sampler.update_environment(sdf, sdf_grad)
        if self.action_sampler is not None:
            with torch.no_grad():
                _, self.z_env, self.z_mu, self.z_sigma = self.action_sampler.environment_encoder.vae(
                    self.normalised_sdf[0].unsqueeze(0))
                imagined_sdf = self.action_sampler.environment_encoder.reconstruct(self.z_env, N=1)
                imagined_sdf = imagined_sdf['environments']
                self.imagined_sdf = imagined_sdf.cpu().numpy()[0, 0, 0]
                self.og_z_env = self.z_env.clone()

        if sdf_grad is not None:
            self.sdf_grad = torch.from_numpy(sdf_grad).to(device=self.device).unsqueeze(0)
        else:
            self.sdf_grad = torch.empty(1, *self.sdf.shape[2:], len(self.sdf.shape[2:]))

    def update_goal(self, goal):
        self.goal = torch.from_numpy(goal).to(device=self.device).reshape(1, -1).float()

    def update_cost_params(self, cost_params):
        self.cost_params = cost_params
        if self.cost_params is not None:
            self.cost_params = self.cost_params.reshape(1, -1)

    def step(self, state, project=False):
        
        step_start = time.time()
        tstate = torch.from_numpy(state).to(device=self.device).reshape(1, -1).float()
        project_time = 0
        diff = 0
        vae_time = 0
        flow_time = 0
        loss_time = 0
        backward_time = 0
        optimiser_time = 0
        project_total = 0
        if project:
            start_time=time.time()
            vae_time, flow_time, loss_time, backward_time, optimiser_time, project_total = self.project_imagined_environment(state, 1)
            end_time = time.time()
            project_time = end_time-start_time
            # self.random_shooting_best_env(state)
        with torch.no_grad():
            #print("---------------")
            #print(f"slef_nominal is {self.sample_nominal}")
            if self.sample_nominal:
                N = 5
                U, _, _ = self.action_sampler.sample(
                    tstate,
                    self.goal[0].reshape(1, -1),
                    environment=None,
                    N=N - 1,
                    z_environment=self.z_env[0].unsqueeze(0)
                )

                U = torch.cat((U.reshape(-1, self.H, self.du), self.controller.U.reshape(1, self.H, self.du)), dim=0)
                log_pCost, log_pU, _ = self.generative_model(
                    tstate[-N:],
                    self.goal[-N:],
                    self.sdf[-N:],
                    None, U)
                log_likelihood = log_pCost + log_pU

                u_idx = torch.argmax(log_likelihood)
                # u_idx = torch.randperm(self.N)[0]

                u_sampled = U[u_idx]
                # if log_likelihood[u_idx] < log_likelihood[-1]:
                #    u_sampled = U[-1]

                self.controller.U = u_sampled

        tstate = torch.from_numpy(state).to(device=self.device).reshape(1, -1).float()

        # Step will sometimes fail due to some stupid numerics (bad samples) -- this is a hack but it seems to be rare
        # enough that it is OK
        U, forward_NF_time, reverse_NF_time, cost_time = self.controller.step(tstate)
        # return first action from sequence, and entire action sequence

        step_end = time.time()
        
        print("--------------------------------------------------------")
        forward_NF_percent = forward_NF_time/(step_end - step_start) * 100
        print(f"Time for forward NF is {forward_NF_time}; The percentage is {forward_NF_percent}%")
        reverse_NF_percent = reverse_NF_time/(step_end - step_start) * 100
        print(f"Time for reverse NF is {reverse_NF_time}; The percentage is {reverse_NF_percent}%")
        cost_percent = cost_time/(step_end - step_start) * 100
        print(f"Time for Cost function is {cost_time}; The percentage is {cost_percent}%")
        project_percent = project_time/(step_end - step_start) * 100
        print(f"Time for projection function is {project_time}; The percentage is {project_percent}%")
        
        '''
	diff_percent = diff/project_time * 100
        print(f"Time for part of projection function is {diff}; The percentage is {diff_percent}%")
        '''
        step_total = step_end-step_start
        other_code_time = step_end - step_start - project_time - forward_NF_time - reverse_NF_time - cost_time
        other_percent = other_code_time/(step_end - step_start) * 100
        print(f"Time for other code in step function is {other_code_time}; The percentage is {other_percent}%")
        
        
        return U[0].detach().cpu().numpy(), self.controller.best_K_U.detach().cpu().numpy(), U.detach().cpu().numpy(), forward_NF_time, reverse_NF_time, cost_time, project_time, vae_time, flow_time, loss_time, backward_time, optimiser_time, step_total, project_total

    def random_shooting_best_env(self, state):
        num_envs = 100
        samples_per_env = 10
        z_env_dim = self.z_env.shape[-1]

        with torch.no_grad():
            starts = torch.from_numpy(state).reshape(1, -1).repeat(num_envs, 1).to(device=self.device).float()
            goals = self.goal.reshape(1, -1).repeat(num_envs, 1)

            # Sample a bunch of different environments
            z_env = self.action_sampler.environment_encoder.vae.prior.sample(sample_shape=(num_envs, z_env_dim))
            z_env[0] = self.z_env

            # Sample a bunch of trajectories for each goal
            U, log_qU, context_dict = self.action_sampler.sample(starts, goals, environment=None,
                                                                 z_environment=z_env, N=samples_per_env)

            # MPPI like thing for environment

            # Roll all envs together
            U = U.reshape(-1, self.H, self.du)

            # Evaluate costs            self.z_env = z_env[torch.argmin(costs)].unsqueeze(0)

            costs = self.cost(starts[0].unsqueeze(0), U).reshape(num_envs, samples_per_env).mean(dim=1)
            weights = torch.softmax(-costs / 100, dim=0)
            self.z_env = torch.sum(weights.reshape(-1, 1) * z_env, dim=0, keepdim=True)

    def project_imagined_environment(self, state, num_iters, name=None):
        total_start = time.time()
        loss_fn = SVIMPC_LossFcn(self.generative_model, False, use_grad=self.use_true_grad)
        lr = 1e-2
        
        z_env = torch.nn.Parameter(self.z_env[0].unsqueeze(0).clone())
        z_env.requires_grad = True
        optimiser = optim.Adam(
            [{'params': z_env}],
            lr=lr
        )

        # TODO Instead we will randomly sample starts and goals in the environments that are not in collision
        # tstate = torch.from_numpy(state).to(device=self.device).reshape(1, -1).float()
        if num_iters > 10:
            num_planning_problems = 10
            num_samples = 100
        else:
            num_planning_problems = 1
            num_samples = self.N

        # Randomly sample starts and goals
        INVALID_VALUE = -99
        states = INVALID_VALUE * torch.ones(num_planning_problems, self.dx, device=self.device)
        goals = INVALID_VALUE * torch.ones(num_planning_problems, self.goal.shape[-1], device=self.device)

        if num_planning_problems > 1:
            while (states == INVALID_VALUE).sum() > 0 or (goals == INVALID_VALUE).sum() > 0:
                prospective_states = -1.8 + 3.6 * torch.rand(num_planning_problems, 4, device=self.device)
                prospective_goals = -1.8 + 3.6 * torch.rand(num_planning_problems, 4, device=self.device)
                g = torch.clamp(64 * (prospective_goals[:, :2] + 2) / 4, min=0, max=63).long()
                s = torch.clamp(64 * (prospective_states[:, :2] + 2) / 4, min=0, max=63).long()
                goals = torch.where(index_sdf(self.sdf[0].repeat(num_planning_problems, 1, 1, 1), g) > -1e-3,
                                    prospective_goals, goals)
                states = torch.where(index_sdf(self.sdf[0].repeat(num_planning_problems, 1, 1, 1), s) > -1e-3,
                                     prospective_states,
                                     states)

            goals[:, self.dx // 2:] = 0.0
            states[:, self.dx // 2:] = torch.randn_like(states[:, self.dx // 2:])
        goals[0] = self.goal[0]
        states[0] = torch.from_numpy(state).to(device=self.device).float()
        # visualise_starts_and_goals(states.detach().cpu().numpy(), goals.detach().cpu().numpy(),
        #                           self.sdf.detach().cpu().numpy()[0, 0])

        if name is not None:
            U, log_qU, context_dict = self.action_sampler.sample(states[0].unsqueeze(0), goals[0].unsqueeze(0),
                                                                 cost_params=self.cost_params,
                                                                 environment=self.normalised_sdf[0].unsqueeze(0),
                                                                 N=100)
            _, _, X = self.generative_model(
                states[0].repeat(100, 1).unsqueeze(0),
                goals[0].repeat(100, 1).unsqueeze(0),
                self.sdf[0].unsqueeze(0),
                None, U)

            visualise_trajectories(states[0].detach().unsqueeze(0).cpu().numpy(),
                                   goals[0].detach().unsqueeze(0).cpu().numpy(),
                                   X.detach().cpu().unsqueeze(0).numpy(), self.raw_sdf, f'{name}_before.png')
        alpha, beta, kappa = 1.0, 1.0, np.prod(self.sdf.shape[1:]) / z_env.shape[1]
        sigma = 1
        #iter_start_time = time.time()
        #print(f"iteration count: {num_iters}")
        for iter in range(num_iters):
            action_start = time.time()
            U, log_qU, context_dict = self.action_sampler(states,
                                                          goals,
                                                          environment=None,
                                                          z_environment=z_env,
                                                          cost_params=self.cost_params,
                                                          N=num_samples,
                                                          sigma=sigma)
            action_end = time.time()
            vae_start = time.time()
            log_p_env = self.action_sampler.environment_encoder.vae.prior.log_prob(z_env).sum(dim=1)
            vae_end = time.time()
            # if self.project_use_reg:
            loss_start = time.time()
            loss_dict, _ = loss_fn.compute_loss(U, log_qU, states, goals,
                                                self.sdf[0],
                                                self.sdf_grad[0], self.cost_params,
                                                log_p_env, None,
                                                alpha=alpha, beta=beta, kappa=kappa, normalize=True)
            loss_end = time.time()
            backward_start = time.time()
            loss_dict['total_loss'].backward()
            backward_end = time.time()
            # else:
            # loss = -kappa * log_p_env.sum() / np.prod(self.sdf.shape[1:])
            # loss.backward()
            # print(log_p_env)
            # print(log_p_env)
            optimiser.step()
            optimiser.zero_grad()
            total_end = time.time()
            total_time = total_end - total_start
            flow_time = action_end - action_start
            vae_time = vae_end - vae_start
            loss_time = loss_end - loss_start
            backward_time = backward_end - backward_start
            optimiser_time = total_time - flow_time - vae_time - loss_time - backward_time
            print("--------------------------------------------------------")
            vae_percent = vae_time/total_time * 100
            print(f"Time for VAE is {vae_time}; The percentage is {vae_percent}%")
            flow_percent = flow_time/total_time * 100
            print(f"Time for flow is {flow_time}; The percentage is {flow_percent}%")
            loss_percent = loss_time/total_time * 100
            print(f"Time for loss is {loss_time}; The percentage is {loss_percent}%")
            backward_percent = backward_time/total_time * 100
            print(f"Time for backward is {backward_time}; The percentage is {backward_percent}%")
            opt_percent = optimiser_time/total_time * 100
            print(f"Time for opt is {optimiser_time}; The percentage is {opt_percent}%")
            grad_percent = loss_fn.grad_time/total_time * 100
            print(f"Time for grad_t is {loss_fn.grad_time}; The percentage is {grad_percent}%")
            forward_percent = self.action_sampler.forward_time/flow_time * 100
            print(f"Time for forward/flow is {self.action_sampler.forward_time}; The percentage is {forward_percent}%")
            logqu_sample_percent = self.action_sampler.logqu_sample_time/flow_time * 100
            print(f"Time for logqu_sample/flow is {self.action_sampler.logqu_sample_time}; The percentage is {logqu_sample_percent}%")
            logqu_like_percent = self.action_sampler.logqu_like_time/flow_time * 100
            print(f"Time for logqu_likelihood/flow is {self.action_sampler.logqu_like_time}; The percentage is {logqu_like_percent}%")
            horizon_percent = loss_fn.horizon_time/loss_fn.grad_time * 100
            print(f"Time for horizon/grad is {loss_fn.horizon_time}; The percentage is {horizon_percent}%")
            print(f"Total Time is {total_time}")

            


        #iter_end_time = time.time()
        #diff_time = iter_end_time - iter_start_time
        with torch.no_grad():
            imagined_sdf = self.action_sampler.environment_encoder.reconstruct(z_env, N=1)
            imagined_sdf = imagined_sdf['environments']
            imagined_sdf = imagined_sdf.cpu().numpy()[0, 0, 0]

        self.imagined_sdf = imagined_sdf
        self.z_env = z_env

        if name is not None:
            U, log_qU, context_dict = self.action_sampler.sample(states[0].unsqueeze(0), goals[0].unsqueeze(0),
                                                                 environment=None,
                                                                 z_environment=z_env,
                                                                 cost_params=self.cost_params,
                                                                 N=100)
            _, _, X = self.generative_model.posterior_log_likelihood(
                states[0].repeat(100, 1).unsqueeze(0),
                goals[0].repeat(100, 1).unsqueeze(0),
                self.sdf[0].unsqueeze(0),
                None, U)

            visualise_trajectories(states[0].detach().unsqueeze(0).cpu().numpy(),
                                   goals[0].detach().unsqueeze(0).cpu().numpy(),
                                   X.detach().cpu().unsqueeze(0).numpy(), self.raw_sdf, f'{name}_after.png')
        return vae_time, flow_time, loss_time, backward_time, optimiser_time, total_time
        # print('Deviation from original z', torch.linalg.norm(self.z_env - self.og_z_env))

    def load_model(self, model_path):
        self.action_sampler.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)

    def reset(self):
        self.controller.reset()


def visualise_starts_and_goals(starts, goals, sdf):
    import matplotlib.pyplot as plt
    from cv2 import resize
    import numpy as np
    #fig, ax = plt.subplots(1, 1)
    big_sdf = resize(sdf, (256, 256))
    goals = np.clip(256 * (goals[:, :2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)
    starts = np.clip(256 * (starts[:, :2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

    ax.imshow(big_sdf[::-1])
    '''
    for goal, start in zip(goals, starts):
        plt.plot(goal[0], 255 - goal[1], marker='o', color="red", linewidth=2)
        plt.plot(start[0], 255 - start[1], marker='o', color="green", linewidth=2)
        plt.plot([start[0], goal[0]], [255 - start[1], 255 - goal[1]], color='b', alpha=0.5)

    plt.show()
    '''

def visualise_trajectories(starts, goals, trajectories, sdf, name):
    import matplotlib.pyplot as plt
    import numpy as np
    from cv2 import resize
    #fig, ax = plt.subplots(1, 1)
    big_sdf = resize(sdf, (256, 256))
    goals = np.clip(256 * (goals[:, :2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)
    starts = np.clip(256 * (starts[:, :2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)
    '''
    ax.imshow(big_sdf[::-1])
    for goal, start, trajs in zip(goals, starts, trajectories):
        plt.plot(goal[0], 255 - goal[1], marker='x', color="red", linewidth=2)
        plt.plot(start[0], 255 - start[1], marker='o', color="green", linewidth=2)
        # plt.plot([start[0], goal[0]], [255 - start[1], 255 - goal[1]], color='b', alpha=0.5)
        positions = trajs[:, :, :2]
        positions_idx = np.clip(256 * (positions + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

        for i in range(len(positions_idx)):
            ax.plot(positions_idx[i, :, 0], 255 - positions_idx[i, :, 1], linewidth=1, alpha=0.5)

    fig.savefig(name)
    '''

def index_sdf(sdf, indices):
    device = sdf.device
    N = sdf.shape[0]
    indexy = torch.arange(64, device=device, dtype=torch.long).reshape(1, 64, 1).repeat(N, 1, 1)
    indexy[:, :, 0] = indices[:, 0].reshape(-1, 1).repeat(1, 64)
    y_indexed_sdf = sdf.view(-1, 64, 64).gather(2, indexy).squeeze(2)
    return y_indexed_sdf.gather(1, indices[:, 1].reshape(-1, 1))
'''
with open('../perf.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow()
'''
