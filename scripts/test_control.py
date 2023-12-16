import torch
import pathlib
import argparse
import numpy as np
# annoying numpy deprecation issue
np.bool = bool
np.float = float
from cv2 import resize
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from flow_mpc.environments import DoubleIntegratorEnv, QuadcopterEnv, QuadcopterDynamicEnv
from flow_mpc.models import DoubleIntegratorModel, QuadcopterModel
from flow_mpc.controllers import MPCController
from flow_mpc.encoders import ConditioningNetwork, VAEEncoder
from flow_mpc.action_samplers import FlowActionSampler
from flow_mpc.utils import gen_cost_params
import csv

FLOW_MPC_ROOT = pathlib.Path(__file__).resolve().parents[1]

np.random.seed(1)
import time


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    return args


def create_animation(start, goal, sdf, trajectories, planned_trajectories, projected_sdfs, env_no, controller, env_type,
                     name,
                     costs=None):
    import matplotlib.pylab as pl

    # costs -= np.max(costs, axis=1, keepdims=True)
    # costs /= np.min(costs, axis=1, keepdims=True)
    # if projected_sdfs is not None:
    #    projected_sdfs = np.where(projected_sdfs < 0, 1000 * projected_sdfs, projected_sdfs)
    big_sdf = resize(sdf, (256, 256))
    # big_sdf = np.where(big_sdf < 0, 255 * np.ones_like(big_sdf), np.zeros_like(big_sdf))
    goal = np.clip(256 * (goal[:2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)
    start = np.clip(256 * (start[:2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

    frames = [{'t': t} for t in np.arange(1, trajectories.shape[0], 2)]

    if projected_sdfs is None:
        fig, ax = plt.subplots(1, 1)
        ax.grid(False)
        ax.set_axis_off()
    else:
        fig, axes = plt.subplots(1, 2)
        ax = axes[0]
        ax2 = axes[1]
        ax.grid(False)
        ax2.grid(False)
        ax.set_axis_off()
        ax2.set_axis_off()

    positions = trajectories[:, :2]
    positions_idx = np.clip(256 * (positions + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

    planned_positions = planned_trajectories[:, :, :, :2]
    planned_positions_idx = np.clip(256 * (planned_positions + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

    ax.imshow(big_sdf[::-1])
    alpha = min(0.1 + 4.0 / planned_positions.shape[1], 1)

    def animate(frames):

        def plot_on_sdf(sdf, ax):
            ax.lines = []
            ax.patches = []

            ims = [ax.imshow(sdf[::-1]),
                   ax.plot(goal[0], 255 - goal[1], marker='o', color="red", linewidth=4)[0],
                   ax.plot(start[0], 255 - start[1], marker='o', color="green", linewidth=4)[0]]

            ims.extend(
                ax.plot(positions_idx[:frames['t'], 0], 255 - positions_idx[:frames['t'], 1], linewidth=2, color='b'))
            for i in range(planned_positions.shape[1]):
                color = 'k'  # plt.cm.jet(costs[frames['t'], i])
                ims.extend(
                    ax.plot(planned_positions_idx[frames['t'], i, :, 0],
                            255 - planned_positions_idx[frames['t'], i, :, 1],
                            linewidth=2,
                            color=color, alpha=alpha))
            return ims

        ims = plot_on_sdf(big_sdf, ax)

        if projected_sdfs is not None:
            projected_big_sdf = resize(projected_sdfs[frames['t']], (256, 256))
            ims.extend(plot_on_sdf(projected_big_sdf, ax2))

        # plt.savefig(f'{FLOW_MPC_ROOT}/figures/{name}/control_test/env_{env_no}_{controller}_{env_type}_frame_{frames["t"]}.png')
        return ims

    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=10, blit=True)
    ani.save(f'{FLOW_MPC_ROOT}/figures/{name}/control_test/{env_type}/env_{env_no}_{controller}.gif', writer='pillow',
             fps=4)
    plt.close()

    return ani


def save_visualisation_data(start, goal, sdf, trajectories, planning_trajectories, env_no, controller, env_type, name):
    fname = f'{FLOW_MPC_ROOT}/figures/{name}/control_test/{controller}/{env_type}/env_{env_no}_vis.npz'
    data = {}
    data['sdf'] = sdf
    data['starts'] = start
    data['goals'] = goal
    data['X'] = trajectories
    data['planned_X'] = planning_trajectories
    np.savez(fname, **data)


def visualise_trajectory(start, goal, trajectories, sdf, env_no, controller, env_type, name):
    fig, ax = plt.subplots(1, 1)
    big_sdf = resize(sdf, (256, 256))
    goal = np.clip(256 * (goal[:2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)
    start = np.clip(256 * (start[:2] + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

    positions = trajectories[:, :, :2]
    positions_idx = np.clip(256 * (positions + 2) / 4, a_min=0, a_max=255).astype(np.uint64)

    ax.imshow(big_sdf[::-1])
    for i in range(len(positions_idx)):
        ax.plot(positions_idx[i, :, 0], 255 - positions_idx[i, :, 1], linewidth=2)
    plt.plot(goal[0], 255 - goal[1], marker='o', color="red", linewidth=4)
    plt.plot(start[0], 255 - start[1], marker='o', color="green", linewidth=4)
    # plt.show()
    plt.savefig(f'{FLOW_MPC_ROOT}/figures/{name}/control_test/{env_type}/env_{env_no}_{controller}.png')
    plt.close()


def test_controller(env, controller, T=50):
    total_start = time.time()
    state_history = []
    control_history = []
    state = env.state

    state_history.append(state)
    planned_control_sequences = []
    collision_failure = False
    projected_envs = []
    project_imag_time = 0
    # Let's project upfront a little bit just to get started
    if controller.project:
        start_time = time.time()
        controller.project_imagined_environment(state, 10)
        end_time = time.time()
        project_imag_time = end_time - start_time
       
    total_time = 0.0
    count = 0
    cost = 0.0
    forward_NF_times = []
    reverse_NF_times = []
    cost_times = []
    project_times = []
    vae_times = []
    flow_times = []
    loss_times = []
    backward_times =[]
    optimiser_times = []
    step_totals = [] 
    project_totals = []
    for i in range(T):
        print(f"episode {i}")
        episode_start = time.time()
        if not collision_failure:
            count += 1
            stime = time.time()
            new_control, control_sequence, unneeded, forward_NF_time, reverse_NF_time, cost_time, project_time, vae_time, flow_time, loss_time, backward_time, optimiser_time, step_total, project_total = controller.step(state, project=controller.project)
            forward_NF_times.append(forward_NF_time)
            reverse_NF_times.append(reverse_NF_time)
            cost_times.append(cost_time)
            project_times.append(project_time)
            vae_times.append(vae_time)
            flow_times.append(flow_time)
            loss_times.append(loss_time)
            backward_times.append(backward_time)
            optimiser_times.append(optimiser_time)
            step_totals.append(step_total)
            project_totals.append(project_total)
            torch.cuda.synchronize()
            etime = time.time()
            
            total_time += etime - stime
            controller_step = etime - stime
            new_state, collision = env.step(new_control)
            collision_failure = collision_failure or collision
        if controller.action_sampler is not None:
            projected_envs.append(controller.imagined_sdf)

        if not collision_failure:
            state = new_state.copy()
            control = new_control.copy()
        else:
            control *= 0
        episode_end = time.time()
        ratio  = controller_step/(episode_end - episode_start)*100
        print("-----------------------------------------------------------------------")
        print(f"controller takes:{controller_step}; Percentage of episode is {ratio}%")
        cost += env.cost(vel_penalty=controller.cost_params[0, -1])
        state_history.append(state.copy())
        control_history.append(control.copy())
        planned_control_sequences.append(control_sequence.copy())

    state_history = np.asarray(state_history)
    control_history = np.asarray(control_history)
    planned_control_sequences = np.asarray(planned_control_sequences)
    failure = collision_failure or not env.at_goal()
    print("-------------forward array----------------")
    print(forward_NF_times)
    if len(projected_envs) > 0:
        projected_envs = np.asarray(projected_envs)
    else:
        projected_envs = None
    total_end = time.time()
    project_imag_percentage = project_imag_time/(total_end-total_start)
    print(f"the time for projecting imagined env is {project_imag_time}; percent is {project_imag_percentage}")
    with open('perf_double_integrator.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["One episode breakdown"])
        writer.writerow(step_totals)
        writer.writerow(forward_NF_times)
        writer.writerow(reverse_NF_times)
        writer.writerow(cost_times)
        writer.writerow(project_times)
        
        writer.writerow(["Projection part breakdown"])
        writer.writerow(project_totals)
        writer.writerow(vae_times)
        writer.writerow(flow_times)
        writer.writerow(loss_times)
        writer.writerow(backward_times)
        writer.writerow(optimiser_times)
   
    return -cost, state_history, control_history, planned_control_sequences, failure, projected_envs, total_time / count


if __name__ == '__main__':
    args = parse_arguments()
    import yaml

    config = yaml.safe_load(pathlib.Path(f'{FLOW_MPC_ROOT}/config/testing/{args.config}').read_text())

    pathlib.Path(f'{FLOW_MPC_ROOT}/figures/{config["name"]}/control_test/{config["obstacle_type"]}').mkdir(parents=True,
                                                                                                           exist_ok=True
                                                                                                           )
    # Setup environment and cost models
    if 'double_integrator' in config['env']:
        env = DoubleIntegratorEnv(world_dim=config["world_dim"], world_type=config["obstacle_type"])
        generative_model = DoubleIntegratorModel(world_dim=config["world_dim"]).to(device=config["device"])
    elif 'quadrotor' in config['env']:
        if config['kinematic']:
            env = QuadcopterEnv(world_dim=config['world_dim'], world_type=config['obstacle_type'], dt=config['dt'])
        else:
            env = QuadcopterDynamicEnv(world_dim=config['world_dim'], world_type=config['obstacle_type'],
                                       dt=config['dt'])
        generative_model = QuadcopterModel(world_dim=config['world_dim'], dt=config['dt'],
                                           kinematic=config['kinematic']).to(device=config['device'])
    else:
        raise ValueError('Invalid env specified')

    # Load flow
    flow_prior = None
    if config["vae_flow_prior"]:
        flow_prior = config["flow_type"]

    encoder = VAEEncoder(context_dim=config["context_dim"], z_env_dim=config["z_env_dim"],
                         voxels=config["voxels"], flow_prior=flow_prior)

    # Define action sampler and model of environment / dynamics
    num_flows = 16
    flow_sampler = FlowActionSampler(
        context_net=ConditioningNetwork(context_dim=config["context_dim"], z_env_dim=config["z_env_dim"],
                                        state_dim=config["state_dim"],
                                        goal_dim=config["goal_dim"],
                                        param_dim=config["param_dim"]),
        environment_encoder=encoder,
        action_dimension=config["control_dim"], horizon=config["horizon"], flow_length=config["num_flows"],
        condition_on_cost=config['condition_on_cost_params']
    ).to(device=config["device"])

    flow_sampler.eval()
    import copy

    # instantiate results dictionary
    test_results = {}
    for controller_name in config['controllers'].keys():
        test_results[controller_name] = {}
        test_results[controller_name]['total_costs'] = []
        test_results[controller_name]['likelihood'] = []
        test_results[controller_name]['prior'] = []
        test_results[controller_name]['success'] = []
        test_results[controller_name]['smoothness'] = []
        test_results[controller_name]['compute_time'] = []

        # make sure folders exist
        pathlib.Path(f'{FLOW_MPC_ROOT}/figures/{config["name"]}/control_test/{controller_name}/{str(env)}').mkdir(
            parents=True,
            exist_ok=True)

    # Instantiate controllers        
    controllers = {}
    for controller_name, controller_config in config['controllers'].items():

        if 'flow' in controller_name:
            action_sampler = flow_sampler
        else:
            action_sampler = None

        if 'no_reg' in controller_name:
            project_use_reg = False
        elif 'no_ood' in controller_name:
            project_use_ood = False

        # add name to config
        controller_config['name'] = controller_name
        # make controller
        controller = MPCController(generative_model, config['horizon'], config['control_dim'], config['state_dim'],
                                   controller_config, N=config['sample_budget'], device=config['device'],
                                   action_sampler=action_sampler)

        if 'flow' in controller_name:
            print("----------loading flow model------------")
            controller.load_model(f'{FLOW_MPC_ROOT}/data/{config["name"]}')

        controllers[controller_name] = controller

    flow_images = []
    mppi_images = []
    cost_params = None
    print(f"number of env : {config['num_envs']}")
    for num_env in range(config['num_envs']):
        env.reset(zero_velocity=True)
        # start from zero velocity
        # env.start[2:] *= 0

        goal = env.goal
        sdf, sdf_grad = env.get_sdf()

        # randomize cost parameters
        if config['randomize_cost_params']:
            cost_params = gen_cost_params(config)
        else:
            cost_params = torch.tensor([
                generative_model.default_control_sigma ** 2, 0.01, 0.1
            ]).to(device=config['device'])

        for name, controller in controllers.items():
            env.state = env.start.copy()
            controller.reset()
            controller.update_goal(goal)
            controller.update_environment(sdf, sdf_grad)
            controller.update_cost_params(cost_params)

            likelihood, states, controls, planned_controls, failure, projected_sdfs, comp_time = test_controller(env,
                                                                                                                 controller,
                                                                                                                 config[
                                                                                                                     "episode_length"])
            '''
            if controller.project and config['world_dim'] == 2:
                # controller.project_imagined_environment(env.start.copy())
                fig, axes = plt.subplots(1, 2)
                normalised_sdf = np.where(sdf < 0, sdf, sdf)
                axes[0].imshow(normalised_sdf[::-1])
                axes[1].imshow(controller.imagined_sdf[::-1])
                plt.savefig(
                    f'{FLOW_MPC_ROOT}/figures/{config["name"]}/control_test/{config["obstacle_type"]}/projected_env_{num_env}.png')
                plt.close()
            '''

            def get_planned_trajectory(states, goal, sdf, planned_controls):
                N, T, _ = planned_controls.shape
                start = torch.from_numpy(states).reshape(-1, 1, config['state_dim'])
                start = start.repeat(1, N // config["episode_length"], 1).reshape(-1, 1,
                                                                                  config['state_dim']).float().to(
                    device=config['device'])
                goal = torch.from_numpy(goal).reshape(1, 1, config['state_dim']).repeat(N, 1, 1).float().to(
                    device=config['device'])
                controls = torch.from_numpy(planned_controls).reshape(N, 1, T, config['control_dim']).float().to(
                    device=config['device'])
                # sdf = torch.from_numpy(sdf).unsqueeze(0).unsqueeze(0)
                # sdf = torch.repeat_interleave(sdf, N, dim=1).float().to(device=config['device'])
                with torch.no_grad():
                    _, _, planned_trajectories = generative_model(start, goal, None, None, controls,
                                                                  compute_costs=False)

                return planned_trajectories.squeeze(1).reshape(config["episode_length"], -1, config['horizon'],
                                                               config['state_dim']).cpu().numpy()


            def get_cost(start, goal, sdf, controls, states, cost_params=None):
                T = controls.shape[-2]
                start = torch.from_numpy(start).reshape(-1, 1, config['state_dim']).float().to(device=config['device'])
                goal = torch.from_numpy(goal).reshape(-1, 1, config['state_dim']).float().to(device=config['device'])
                controls = torch.from_numpy(controls).reshape(-1, 1, T, config['control_dim']).float().to(
                    device=config['device'])
                sdf = torch.from_numpy(sdf).unsqueeze(0).unsqueeze(0).float().to(device=config['device'])
                states = torch.from_numpy(states).reshape(1, 1, -1, config['state_dim']).float(
                ).to(device=config['device'])[:, :, 1:]
                if cost_params is not None:
                    cost_params = cost_params.reshape(-1, 3).repeat(sdf.shape[0])
                log_pCost, log_pU, _ = generative_model(start, goal, sdf, None,
                                                        controls, cost_params, X=states)
                return log_pCost.item(), log_pU.item()


            likelihood, prior = get_cost(env.start, env.goal, sdf, controls, states)
            cost = (likelihood + prior)
            if cost != cost:
                exit()

            test_results[name]['total_costs'].append(cost)
            test_results[name]['prior'].append(prior)
            test_results[name]['likelihood'].append(likelihood)
            test_results[name]['success'].append(0.0 if failure else 1.0)
            test_results[name]['smoothness'].append(np.sum(np.diff(controls, axis=0) ** 2) / config['num_envs'])
            test_results[name]['compute_time'].append(comp_time)
            planned_trajectories = get_planned_trajectory(states[:-1],
                                                          goal, sdf,
                                                          planned_controls.reshape(-1, config['horizon'],
                                                                                   config[
                                                                                       'control_dim']))

            # can only create animation if 2D
            #if config['world_dim'] == 2:
                #create_animation(env.start.copy(), goal.copy(), sdf, states.reshape(-1, config['state_dim']),
                 #                planned_trajectories, projected_sdfs, num_env, name, config['obstacle_type'],
                  #               config['name'],
                   #              None)

            # Save trial data
            #save_visualisation_data(env.start.copy(), goal.copy(), sdf, states.reshape(-1, env.state_dim),
              #                      planned_trajectories, num_env, name, str(env), config['name'])

        print(f'### {num_env} ###')
        for controller, results in test_results.items():
            print(f'Average cost for {controller}: {np.mean(results["total_costs"])}')
            print(f'Average success for {controller}: {np.mean(results["success"])}')
            print(f'Average smoothness for {controller}: {np.mean(results["smoothness"])}')
            print(f'Average compute time for {controller}: {np.mean(results["compute_time"])}')

        # save results
        import pickle

        with open(
                f'{FLOW_MPC_ROOT}/figures/{config["name"]}/control_test/results_{str(env)}_{config["sample_budget"]}.pkl',
                'wb') as f:
            pickle.dump(test_results, f)
