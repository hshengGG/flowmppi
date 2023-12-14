import torch


def gen_cost_params(B, config):
    prior_sigma = config['min_sigma'] * 10 ** (torch.rand(B, 1).to(device=config['device']))
    prior_lengthscale = config['min_lengthscale'] * 10 ** (2 * torch.rand(B, 1).to(device=config['device']))
    vel_penalty = config['min_vel_penalty'] * 10 ** (2 * torch.rand(B, 1).to(device=config['device']))
    cost_params = torch.cat((prior_sigma, prior_lengthscale, vel_penalty), dim=1)
    return cost_params


def hyperparam_schedule(epoch, max_epochs, min_param, max_param, schedule='linear'):
    """
    hyperparam scheduler,
    assumes we are going from max_param -> min_param  as epochs progress
    """
    if schedule == 'linear':
        return min_param + (max_param - min_param) * (1.0 - epoch / max_epochs)
    elif schedule == 'quadratic':
        return min_param + (max_param - min_param) * (1.0 - epoch / max_epochs) ** 2
    elif schedule == 'inverse_linear':
        inverse_max_param = 1.0 / max_param
        inverse_min_param = 1.0 / min_param
        inverse_param = inverse_max_param + (inverse_min_param - inverse_max_param) * (epoch / max_epochs)
        return 1.0 / inverse_param
