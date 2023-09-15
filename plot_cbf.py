import torch
import os
import argparse
import matplotlib.pyplot as plt
import shutil
import numpy as np

from torch_geometric.data import Data
from tqdm import tqdm

from gcbf.trainer.utils import set_seed, read_settings, plot_cbf_contour
from gcbf.env import make_env
from gcbf.algo import make_algo


def plot_cbf(args):
    # set random seed
    set_seed(args.seed)

    # set up testing device
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if use_cuda else 'cpu')

    # load training settings
    try:
        settings = read_settings(args.path)
    except TypeError:
        raise TypeError('Cannot find configuration file in the path')

    # make environment
    env = make_env(
        env=settings['env'] if args.env is None else args.env,
        num_agents=settings['num_agents'] if args.num_agents is None else args.num_agents,
        device=device
    )
    params = env.default_params
    params['area_size'] = args.area_size
    params['num_obs'] = args.obs
    env = make_env(
        env=settings['env'] if args.env is None else args.env,
        num_agents=settings['num_agents'] if args.num_agents is None else args.num_agents,
        device=device,
        params=params,
        max_neighbors=12 if settings['algo'] == 'macbf' else None
    )
    env.test()

    # build algorithm
    algo = make_algo(
        algo=settings['algo'],
        env=env,
        num_agents=settings['num_agents'] if args.num_agents is None else args.num_agents,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        action_dim=env.action_dim,
        device=device,
        hyperparams=settings['hyper_params']
    )
    model_path = os.path.join(args.path, 'models')
    if args.iter is not None:
        # load the controller at given iteration
        algo.load(os.path.join(model_path, f'step_{args.iter}'))
    else:
        # load the last controller
        controller_name = os.listdir(model_path)
        controller_name = [i for i in controller_name if 'step' in i]
        controller_id = sorted([int(i.split('step_')[1].split('.')[0]) for i in controller_name])
        algo.load(os.path.join(model_path, f'step_{controller_id[-1]}'))
    fig_path = os.path.join(args.path, 'figs')

    # mkdir for the video and the figures
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    fig_path = os.path.join(fig_path, f'agent_{args.agent}')
    if os.path.exists(fig_path):
        shutil.rmtree(fig_path)
    os.mkdir(fig_path)

    # simulate the environment and plot the CBFs
    for i_epi in range(args.epi):
        set_seed(np.random.randint(100000))
        data = env.reset()
        t = 0
        os.mkdir(os.path.join(fig_path, f'epi_{i_epi}'))
        pbar = tqdm()
        while True:
            data.update(Data(u_ref=env.u_ref(data)))
            action = algo.apply(data)
            pbar.update(1)

            if hasattr(algo, 'cbf'):
                ax = plot_cbf_contour(
                    algo.cbf, data, env, args.agent, args.x_dim, args.y_dim, True)
                plt.savefig(os.path.join(os.path.join(fig_path, f'epi_{i_epi}'), f'{t}.pdf'))
                plt.close()
            else:
                raise KeyError('The algorithm must has a CBF function')
            next_data, reward, done, _ = env.step(action)
            data = next_data
            t += 1
            if done:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--obs', type=int, default=0)
    parser.add_argument('--area-size', type=float, required=True)
    parser.add_argument('-n', '--num-agents', type=int, default=None)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--iter', type=int, default=None)
    parser.add_argument('--epi', type=int, default=5)
    parser.add_argument('--agent', type=int, default=0)
    parser.add_argument('--x-dim', type=int, default=0)
    parser.add_argument('--y-dim', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)

    # default
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', default=False)

    args = parser.parse_args()
    plot_cbf(args)
