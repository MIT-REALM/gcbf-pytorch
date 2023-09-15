import argparse
import torch
import os

from gcbf.env import make_env
from gcbf.algo import make_algo
from gcbf.trainer import Trainer
from gcbf.trainer.utils import set_seed, init_logger, read_params


def train(args):
    # set random seed
    set_seed(args.seed)

    # set up training device
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'> Training with {device}')

    # make environment
    env = make_env(args.env, args.num_agents, device)
    params = env.default_params
    if args.area_size is not None:
        params['area_size'] = args.area_size
    if args.obs is not None:
        params['num_obs'] = args.obs
    env = make_env(args.env, args.num_agents, device, params=params,
                   max_neighbors=12 if args.algo == 'macbf' else None)
    env.train()
    env_test = make_env(args.env, args.num_agents, device, params=params,
                        max_neighbors=12 if args.algo == 'macbf' else None)
    env_test.train()

    # set training params
    params = read_params(args.env, args.algo)
    if params is None or args.cus:
        params = {  # set up custom hyper-parameters
            'alpha': 1.0,
            'eps': 0.02,
            'inner_iter': 10,
            'loss_action_coef': 0.001 if args.action_coef is None else args.action_coef,
            'loss_unsafe_coef': 1.0,
            'loss_safe_coef': 1.0,
            'loss_h_dot_coef': 0.2 if args.h_dot_coef is None else args.h_dot_coef
        }
        print('> Using custom hyper-parameters')
    else:
        print('> Using pre-defined hyper-parameters')

    # set up logger
    log_path = init_logger(
        args.log_path, args.env, args.algo, args.seed, vars(args), hyper_params=params
    )

    # build algorithm
    algo = make_algo(
        args.algo, env, args.num_agents, env.node_dim, env.edge_dim,
        env.action_dim, device, args.batch_size, hyperparams=params
    )

    # set up trainer
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_path,
    )

    # start training
    trainer.train(args.steps, eval_interval=args.steps // 10, eval_epi=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('-n', '--num-agents', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)

    # custom
    parser.add_argument('--area-size', type=float, default=None)
    parser.add_argument('--obs', type=int, default=0)
    parser.add_argument('--algo', type=str, default='gcbf')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cus', action='store_true', default=False)
    parser.add_argument('--h-dot-coef', type=float, default=None)
    parser.add_argument('--action-coef', type=float, default=None)

    # default
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--log-path', type=str, default='./logs')
    parser.add_argument('--batch-size', type=int, default=512)

    args = parser.parse_args()
    train(args)
