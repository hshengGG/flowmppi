import torch
import numpy as np
from torch import nn
from flow_mpc.models.generative_model import GenerativeModel

from flow_mpc.models.pytorch_transforms import euler_angles_to_matrix
from flow_mpc.models.utils import PointGoalFcn, CollisionFcn


def forward(self, state, control):
    ''' unroll state '''
    C = 1200
    mu = 0.55
    M = 3
    I_z = 50
    F_z = 100
    a = 10
    b = 4
    x, y, psi, beta, v_x, v_y, r, delta, F_x = torch.chunk(state, chunks=9, dim=-1)

    delta_des, F_xdes = torch.chunk(control, chunks=2, dim=-1)

    # Trigonometric fcns on all the angles needed for dynamics
    alpha_F = torch.atan2(beta + a*r/v_x)
    alpha_R = torch.atan2(beta - b*r/v_x)
    ksi = torch.sqrt((mu**2*F_z - F_x**2)/(mu*F_z))
    gamma = torch.abs(torch.atan2(3*ksi*F_z*torch.sign(alpha)))
    
    if alpha_F >= gamma:
        F_yF = -mu*ksi*F_z*torch.sign(alpha_F)
    else:
        F_yF = -C*torch.tan(alpha_F) + (C**2*torch.atan2(alpha_F)**3)/(3*ksi*mu*F_z*torch.abs(torch.atan2(alpha_F))) - (C*torch.atan2(alpha_F))**3/(27*(mu*ksi*F_z)**2) 
    
    if alpha_R >= gamma:
        F_yR = -mu*ksi*F_z*torch.sign(alpha_R)
    else:
        F_yR = -C*torch.tan(alpha_R) + (C**2*torch.atan2(alpha_R)**3)/(3*ksi*mu*F_z*torch.abs(torch.atan2(alpha_R))) - (C*torch.atan2(alpha_R))**3/(27*(mu*ksi*F_z)**2) 

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

    return state + dstate * self.dt


