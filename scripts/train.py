import pathlib
import argparse
from torch import optim
from torch import nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset

# flow_mpc imports
from flow_mpc.action_samplers.flow import FlowActionSampler
from flow_mpc.encoders import ConditioningNetwork, VAEEncoder
from flow_mpc.trainer import SVIMPC_LossFcn, Trainer
from flow_mpc.planning_dataset import PlanningProblemDataset, dataset_builder
from flow_mpc.models import DoubleIntegratorModel, QuadcopterModel
from flow_mpc.visualisation import *
from flow_mpc.utils import gen_cost_params, hyperparam_schedule
FLOW_MPC_ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    return args


EPSILON = 1e-6


def train_model(planning_network, generative_model,
                train_loader, test_loader, config):
    # Logging stuff
    writer = SummaryWriter(f'{FLOW_MPC_ROOT}/runs/{config["env"]}_{config["name"]}')

    # Loss and optimisation stuff
    train_loss_fn = SVIMPC_LossFcn(generative_model, repel_trajectories=False,
                                   use_grad=False, supervised=False)
    test_loss_fn = SVIMPC_LossFcn(generative_model, repel_trajectories=False,
                                  use_grad=True, supervised=False)

    # Define a training class
    trainer = Trainer(planning_network, train_loss_fn)
    tester = Trainer(planning_network, test_loss_fn)

    min_loss = 1e8
    trained_vae = False
    if trained_vae:
        for param in trainer.planning_network.environment_encoder.vae.parameters():
            param.requires_grad = False
        trainer.planning_network.environment_encoder.vae.eval()

    optimiser = optim.Adam(planning_network.parameters(), lr=config['lr'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, 0.9)

    for epoch in range(config['epochs']):
        trainer.train()
        epoch_losses = {}
        epoch_metadata = {}

        # ability to turn off VAE training
        if epoch > config['vae_training_epochs']:
            if not trained_vae:
                for param in trainer.planning_network.environment_encoder.vae.parameters():
                    param.requires_grad = False
                optimiser = optim.Adam(trainer.parameters(), lr=config['lr'], weight_decay=0)
                trained_vae = True

        if trained_vae:
            trainer.planning_network.environment_encoder.vae.eval()

        alpha = hyperparam_schedule(epoch, config['epochs'], config['min_alpha'],
                                    config['max_alpha'], config['alpha_schedule'])
        beta = hyperparam_schedule(epoch, config['epochs'], config['min_beta'],
                                    config['max_beta'], config['beta_schedule'])

        kappa = 0 if trained_vae else 5
        if config['train_with_noise']:
            sigma = 1.0 - epoch / config['epochs']
        else:
            sigma = None
        for batch_no, (starts, goals, sdf, sdf_grad, U) in enumerate(train_loader):
            # Send data to device
            starts = starts.to(device=config['device'])
            goals = goals.to(device=config['device'])
            sdf = sdf.to(device=config['device'])
            sdf_grad = sdf_grad.to(device=config['device']) if sdf_grad is not None else None
            U = None

            if config['randomize_cost_params']:
                B = starts.shape[0]
                cost_params = gen_cost_params(B, config)
            else:
                cost_params = None

            plot = batch_no == 0 if generative_model.dworld == 2 else False
            loss_dict = trainer(starts, goals, sdf, sdf_grad, U, cost_params,
                                config['samples_per_env'], alpha, beta, kappa, sigma=sigma, plot=plot,
                                reconstruct=not trained_vae, normalize=config['normalize_costs'])

            # store losses for this epoch
            for key, item in loss_dict.items():
                if key not in epoch_losses.keys():
                    epoch_losses[key] = item.mean() / len(train_loader)
                else:
                    epoch_losses[key] += item.mean() / len(train_loader)

            # TODO at the moment I can't work out how to get the metadata out from the parallel module
            # I need to figure this out
            if isinstance(trainer, nn.DataParallel):
                metadata = {}
            else:
                metadata = trainer.metadata

            # Store metadata for sending to tensorboard
            for key, item in metadata.items():
                if key not in epoch_metadata.keys():
                    epoch_metadata[key] = item
                else:
                    if epoch_metadata[key]['type'] == 'figure':
                        continue

                    epoch_metadata[key]['data'] = np.concatenate((epoch_metadata[key]['data'],
                                                                  metadata[key]['data']), axis=0)

            loss_dict['total_loss'].mean().backward()
            torch.nn.utils.clip_grad_norm_(planning_network.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad()

        if (epoch % (config['epochs'] // 20)) == 0:
            scheduler.step()

        planning_network.eval()
        test_loss = test(tester, test_loader, config, do_reconstruction=not trained_vae)
        if (epoch % config['print_epochs']) == 0 or epoch == config['epochs'] - 1:
            print('###############')
            print(f'Epoch: {epoch}')
            for key, items in epoch_losses.items():
                print(f'{key}  Train: {epoch_losses[key]}   Test: {test_loss[key]}')

            if test_loss['total_loss'] < min_loss:
                min_loss = test_loss['total_loss']
                torch.save(planning_network.state_dict(), f'{FLOW_MPC_ROOT}/data/{config["env"]}_{config["name"]}/best_model')

        if (epoch % config["print_epochs"]) == 0 or epoch == (config["epochs"] - 1):
            visualise(planning_network, generative_model, test_loader, config, epoch, 'test')

            visualise(planning_network, generative_model, train_loader, config, epoch, 'train')

        # Send losses to tensorboard
        for key, item in epoch_losses.items():
            writer.add_scalar(f'train/{key}', item, epoch)
            writer.add_scalar(f'test/{key}', test_loss[key], epoch)

        # Send metadata to tensorboard
        for key, item in epoch_metadata.items():
            if item['type'] == 'histogram':
                writer.add_histogram(f'train/{key}', item['data'].reshape(-1), epoch)
            if item['type'] == 'figure':
                writer.add_figure(f'train/{key}', item['data'], epoch)
        plt.close('all')


def test(tester, test_loader, config, do_reconstruction=True):
    tester.eval()
    total_loss = {}

    alpha = 1.0
    beta = 1.0
    kappa = 0

    with torch.no_grad():
        for starts, goals, sdf, sdf_grad, U in test_loader:

            # Send data to device
            starts = starts.to(device=config['device'])
            goals = goals.to(device=config['device'])
            sdf = sdf.to(device=config['device'])
            sdf_grad = sdf_grad.to(device=config['device'])
            U = U.to(device=config['device']) if U is not None else None
            U = None
            if config['randomize_cost_params']:
                B = starts.shape[0]
                cost_params = gen_cost_params(B, config)
            else:
                cost_params = None

            loss_dict = tester(starts, goals, sdf, sdf_grad, U, cost_params,
                               128, alpha, beta, kappa, reconstruct=do_reconstruction)

            # store losses for this epoch
            for key, item in loss_dict.items():
                if key not in total_loss.keys():
                    total_loss[key] = item / len(test_loader)
                else:
                    total_loss[key] += item / len(test_loader)

    return total_loss


def visualise(planning_network, generative_model, test_loader, config, epoch, prefix):
    fdir = f'{FLOW_MPC_ROOT}/figures/{config["env"]}_{config["name"]}'

    # If the world is 2d it is much easier for us to plot stuff
    if generative_model.dworld == 2:
        if isinstance(test_loader.dataset, ConcatDataset):
            # just take from first dataset
            starts, goals, sdf, _, _ = test_loader.dataset.datasets[0][:16]
        else:
            starts, goals, sdf, _, _ = test_loader.dataset[:16]

        plot_trajectories(planning_network, generative_model, starts, goals,
                          sdf, f'{fdir}/{prefix}_trajectories_{epoch}', config)
        plot_sdf_samples(planning_network, sdf[:8], f'{fdir}/{prefix}_sdf_samples_{epoch}', config)

    save_for_visualisation(planning_network, generative_model, test_loader, config, epoch, prefix)


def save_for_visualisation(planning_network, generative_model, test_loader, config, epoch, prefix):
    fdir = f'{FLOW_MPC_ROOT}/figures/{config["env"]}_{config["name"]}/data'

    if isinstance(test_loader.dataset, ConcatDataset):
        # just take from first dataset
        starts, goals, sdf, _, _ = test_loader.dataset.datasets[0][:16]
    else:
        starts, goals, sdf, _, _ = test_loader.dataset[:16]

    data = {}
    data['starts'] = starts
    data['goals'] = goals
    data['sdf'] = sdf[:, 0]

    # Send data to GPU
    starts = starts.to(device=config['device'])
    goals = goals.to(device=config['device'])
    sdf = sdf.to(device=config['device'])
    if config['randomize_cost_params']:
        B = starts.shape[0]
        cost_params = gen_cost_params(B, config)
    else:
        cost_params = None
    with torch.no_grad():
        # Generate trajectories for visualisation
        U, _, context_dict = planning_network(starts, goals, sdf, cost_params, N=config['samples_per_vis'])
        _, _, trajectory = generative_model(
            starts.unsqueeze(1).repeat(1, config['samples_per_vis'], 1),
            goals.unsqueeze(1).repeat(1, config['samples_per_vis'], 1),
            sdf,
            None,
            U
        )

        if isinstance(planning_network, nn.DataParallel):
            encoder = planning_network.module.environment_encoder
        else:
            encoder = planning_network.environment_encoder

        if 'h_environment' in context_dict.keys():
            key = 'h_environment'
        else:
            key = 'z_environment'

        reconstructed = encoder.reconstruct(context_dict[key])[
            'environments'].squeeze(0)

    trajectory = trajectory.reshape(16, config['samples_per_vis'], config['horizon'], -1).cpu().numpy()

    data['U'] = U.cpu().numpy()
    data['X'] = trajectory
    data['sdf_recon'] = reconstructed.cpu().numpy()

    np.savez(f'{fdir}/vis_{epoch}_{prefix}', **data)


if __name__ == '__main__':
    args = parse_arguments()
    import yaml
    config = yaml.safe_load(pathlib.Path(f'{FLOW_MPC_ROOT}/config/training/{args.config}').read_text())

    # Make sure relevant folders exist
    pathlib.Path(f'{FLOW_MPC_ROOT}/data/{config["env"]}_{config["name"]}').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{FLOW_MPC_ROOT}/figures/{config["env"]}_{config["name"]}/data').mkdir(parents=True, exist_ok=True)

    # Get datasets
    if len(config['training_data']) == 1:
        train_dataset = PlanningProblemDataset(f'{FLOW_MPC_ROOT}/data/{config["training_data"][0]}', False, False)
    else:
        train_dataset = dataset_builder([f'{FLOW_MPC_ROOT}/data/{fname}' for fname in config['training_data']],
                                        False, no_load=False)
    test_dataset = PlanningProblemDataset(f'{FLOW_MPC_ROOT}/data/{config["test_data"]}', False, False)

    # Data loaders
    train_sampler = RandomSampler(train_dataset)
    test_sampler = RandomSampler(test_dataset)

    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              batch_size=config["batch_size"], num_workers=8, drop_last=False, pin_memory=False)
    test_loader = DataLoader(test_dataset, sampler=test_sampler,
                             batch_size=len(test_dataset), num_workers=1, drop_last=False, pin_memory=False)

    flow_prior = None
    if config['vae_flow_prior']:
        flow_prior = config['flow_type']

    # Environment encoder
    encoder = VAEEncoder(context_dim=config['context_dim'], z_env_dim=config['z_env_dim'],
                         voxels=config['voxels'], flow_prior=flow_prior)

    # Control sampler
    planning_network = FlowActionSampler(
        context_net=ConditioningNetwork(context_dim=config['context_dim'], z_env_dim=config['z_env_dim'],
                                        state_dim=config['state_dim'], goal_dim=config['goal_dim'],
                                        param_dim=config['param_dim']),
        environment_encoder=encoder,
        action_dimension=config['control_dim'], horizon=config['horizon'], flow_length=config['num_flows'],
        flow_type=config['flow_type'], condition_on_cost=config['condition_on_cost_params'],
    ).to(device=config['device'])

    if config['env'] =='double_integrator':
        generative_model = DoubleIntegratorModel(world_dim=config['world_dim']).to(device=config['device'])
    elif config['env'] == 'quadrotor':
        generative_model = QuadcopterModel(world_dim=config['world_dim'],
                                           dt=config['dt'],
                                           kinematic=config['kinematic']).to(device=config['device'])
    elif config['env'] == 'victor':
        generative_model = VictorModel(dt=config['dt']).to(device=args.device)


    # Train
    train_model(planning_network, generative_model, train_loader, test_loader, config)
    torch.save(planning_network.state_dict(), f'{FLOW_MPC_ROOT}/data/{config["env"]}_{config["name"]}/final_model')
