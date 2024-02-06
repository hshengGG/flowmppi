import torch
from torch import nn
from flow_mpc.models.utils import CollisionFcn
from flow_mpc.models.generative_model import GenerativeModel
#import race_car.forward



class DoubleIntegratorDynamics(nn.Module):

    def __init__(self, dim=2):
        super(DoubleIntegratorDynamics, self).__init__()
        dt = 0.05
        self.dt = dt

        if dim == 2:
            # Add viscous damping to A matrix
            self.register_buffer('A', torch.tensor([[1.0, 0.0, dt, 0.0],
                                                    [0.0, 1.0, 0.0, dt],
                                                    [0.0, 0.0, 0.95, 0.0],
                                                    [0.0, 0.0, 0.0, 0.95]]))

            self.register_buffer('B', torch.tensor([[0.0, 0.0],
                                                    [0.0, 0.0],
                                                    [dt, 0.0],
                                                    [0.0, dt]]))
        elif dim == 3:
            # Add viscous damping to A matrix
            self.register_buffer('A', torch.tensor([[1.0, 0.0, 0.0, dt, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                                                    [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                                                    [0.0, 0.0, 0.0, 0.95, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.95, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.95]
                                                    ]))

            self.register_buffer('B', torch.tensor([[0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0],
                                                    [dt, 0.0, 0.0],
                                                    [0.0, dt, 0.0],
                                                    [0.0, 0.0, dt]
                                                    ]))
        else:
            raise ValueError('dim must either be 2 or 3')

    def forward(self, state, control): #changed from action to control
        #return race_car.forward(state,control)
        #return self.batched_dynamics(state, control)
        ''' unroll state '''
        C = 1200
        mu = 0.55
        M = 3
        I_z = 50
        F_z = 100
        a = 10
        b = 4
        state = torch.cat((state, torch.rand((256,5), device = 'cuda:0')), 1)
        
        x, y, psi, beta, v_x, v_y, r, delta, F_x = torch.chunk(state, chunks=9, dim=-1)

        delta_des, F_xdes = torch.chunk(control, chunks=2, dim=-1)

        # Trigonometric fcns on all the angles needed for dynamics
        F = beta + a*r/v_x
        R = beta - b*r/v_x
        
        alpha_F = torch.atan2(F, torch.rand(F.size(), device = 'cuda:0'))

        alpha_R = torch.atan2(R, torch.rand(R.size(), device = 'cuda:0'))
        ksi = torch.sqrt((mu**2*F_z - F_x**2)/(mu*F_z))
        gamma_val = 3*ksi*F_z*torch.sign(alpha_F)

        gamma = torch.abs(torch.atan2(gamma_val, torch.rand(gamma_val.size(), device = 'cuda:0')))

        frand = torch.rand(alpha_F.size(), device = 'cuda:0')
        yF_compare = torch.ge(alpha_F, gamma)
        F_yF = yF_compare * (-mu*ksi*F_z*torch.sign(alpha_R)) + ~yF_compare * (-C*torch.tan(alpha_F) + (C**2*torch.atan2(alpha_F, frand)**3)/(3*ksi*mu*F_z*torch.abs(torch.atan2(alpha_F, frand))) - (C*torch.atan2(alpha_F, frand))**3/(27*(mu*ksi*F_z)**2))

        yR_compare = torch.ge(alpha_R, gamma)
        F_yR = yR_compare*(-mu*ksi*F_z*torch.sign(alpha_R)) + ~yR_compare * (-C*torch.tan(alpha_R) + (C**2*torch.atan2(alpha_R, frand)**3)/(3*ksi*mu*F_z*torch.abs(torch.atan2(alpha_R, frand)) - (C*torch.atan2(alpha_R, frand))**3/(27*(mu*ksi*F_z)**2)))

        x_dot = v_x*torch.cos(psi) - v_y*torch.sin(psi)
        y_dot = v_x*torch.sin(psi) + v_y*torch.cos(psi)
        psi_dot = r
        beta_dot = (F_yF + F_yR)/(M*v_x) - r
        v_x_dot = (F_x - F_yF*torch.sin(delta))/M + r*v_x*beta
        v_y_dot = (F_yF + F_yR)/M - r*v_x
        r_dot = (a*F_yF - b*F_yR)/I_z
        delta_dot = 10*(delta - delta_des)
        F_x_dot = 10*(F_x - F_xdes)

        dstate = torch.cat((x_dot, y_dot, psi_dot, beta_dot, v_x_dot, v_y_dot, r_dot, delta_dot, F_x_dot), dim=-1)

        return (state + dstate * self.dt)[:, 0:4]
       
       
       
        ''' unroll state '''
        '''
        g = -9.81
        m = 1
        Ix, Iy, Iz = 0.5, 0.1, 0.3
        K = 5
        #print(control.shape)
        #print(state)
        #print(state.shape)
        state = torch.cat((state, torch.rand((256,8), device = 'cuda:0')), 1)
        #print(state)
        control =  torch.cat((control, torch.rand((256,2), device = 'cuda:0')),1)
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = torch.chunk(state, chunks=12, dim=-1)

        u1, u2, u3, u4 = torch.chunk(control, chunks=4, dim=-1)

        # Trigonometric fcns on all the angles needed for dynamics
        cphi = torch.cos(phi)
        ctheta = torch.cos(theta)
        cpsi = torch.cos(psi)

        sphi = torch.sin(phi)
        stheta = torch.sin(theta)
        spsi = torch.sin(psi)

        ttheta = torch.tan(theta)

        x_ddot = -(sphi * spsi + cpsi * cphi * stheta) * K * u1 / m
        y_ddot = - (cpsi * sphi - cphi * spsi * stheta) * K * u1 / m
        z_ddot = g - (cphi * ctheta) * K * u1 / m

        p_dot = ((Iy - Iz) * q * r + K * u2) / Ix
        q_dot = ((Iz - Ix) * p * r + K * u3) / Iy
        r_dot = ((Ix - Iy) * p * q + K * u4) / Iz
        
        # velocities
        psi_dot = q * sphi / ctheta + r * cphi / ctheta
        theta_dot = q * cphi - r * sphi
        phi_dot = p + q * sphi * ttheta + r * cphi * ttheta

        dstate = torch.cat((x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot,
                            x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot), dim=-1)
        #print((state + dstate * self.dt).shape)
        #ans = (state + dstate * self.dt)[:,0:4]
        #print(ans)
        #print(ans.shape)
        return (state + dstate * self.dt)[:, 0:4]
        '''
    def batched_dynamics(self, state, action):
        u = action  # torch.clamp(action, min=-10, max=10)
        
        return (self.A @ state.unsqueeze(2) + self.B @ u.unsqueeze(2)).squeeze()


class DoubleIntegratorModel(GenerativeModel):

    def __init__(self, world_dim=2):
        assert world_dim == 2 or world_dim == 3
        self.dworld = world_dim
        dynamics = DoubleIntegratorDynamics(dim=world_dim)
        super().__init__(dynamics=dynamics, sigma=1, state_dim=2 * world_dim, control_dim=world_dim)

    @staticmethod
    def state_to_configuration(state):
        # Converts a full state to a position for checking the SDF
        config, _ = torch.chunk(state, dim=-1, chunks=2)
        return config

    @staticmethod
    def state_to_vel(state):
        _, vel = torch.chunk(state, dim=-1, chunks=2)
        return vel

    def goal_log_likelihood(self, state, goal, vel_penalty=None):
        state_config = self.state_to_configuration(state)
        goal_config = self.state_to_configuration(goal)
        vel = self.state_to_vel(state)
        ll = -10 * torch.linalg.norm(state_config - goal_config, dim=-1)
        if vel_penalty is not None:
            vel_cost = vel_penalty.reshape(-1, 1) * torch.linalg.norm(vel)
            ll -= vel_cost
        return ll

    def collision_log_likelihood(self, state, sdf, sdf_grad):
        config = self.state_to_configuration(state)
        sdf_val = self.collision_fn.apply(config, sdf, sdf_grad)
        return 1e4 * (torch.clamp(sdf_val, max=0.1, min=None) - 0.1)


def trajectory_kernel(X, goals):
    # We do kind of a time convoluted traj kernel
    B, T, N, d = X.shape
    lengthscale = 0.7
    squared_distance = sq_diff(X, X)  # will return batch_size x traj_length x num_samples x num_samples
    squared_distance2 = sq_diff(X[:, :-1], X[:, 1:])
    squared_distance3 = sq_diff(X[:, :-2], X[:, 2:])
    distance_to_goal = 1e-5 + torch.linalg.norm(X.reshape(-1, d) - goals[:, :2].repeat(1, T, 1).reshape(-1, 2),
                                                dim=1).reshape(B, T, N, 1).repeat(1, 1, 1, N)
    weighting = 0.5 * (distance_to_goal + distance_to_goal.permute(0, 1, 3, 2))
    weighting2 = 0.5 * (distance_to_goal[:, :-1] + distance_to_goal[:, 1:].permute(0, 1, 3, 2))
    weighting3 = 0.5 * (distance_to_goal[:, :-2] + distance_to_goal[:, 2:].permute(0, 1, 3, 2))

    K = torch.exp(-squared_distance / (0.5 * weighting)).sum(dim=1) + \
        torch.exp(-squared_distance2 / (0.10 * weighting2)).sum(dim=1) + \
        torch.exp(-squared_distance3 / (0.05 * weighting3)).sum(dim=1)
    return K.mean(dim=[1, 2])


def sq_diff(x, y):
    ''' Expects trajectories batch_size x time x num_samples x state_dim'''
    # G = torch.einsum('btij, btjk->btik', (x, y.transpose(2, 3)))
    # diagonals = torch.einsum('btii->bti', G)
    # D = diagonals.unsqueeze(-2) + diagonals.unsqueeze(-1) - 2 * G
    B, T, N, d = x.shape
    D = torch.cdist(x.reshape(B * T, N, d), y.reshape(B * T, N, d), p=2).reshape(B, T, N, N)
    return D
