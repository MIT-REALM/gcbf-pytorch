import torch
import os
import cv2
import numpy as np
import argparse
import time

from PIL import Image
from scipy.io import savemat

from gcbf.trainer.utils import set_seed, read_settings, eval_ctrl_epi
from gcbf.env import make_env
from gcbf.algo import make_algo


def test(args):
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
        settings = {'algo': 'nominal', 'num_agents': args.num_agents}

    # make environment
    env = make_env(
        env=settings['env'] if args.env is None else args.env,
        num_agents=settings['num_agents'] if args.num_agents is None else args.num_agents,
        device=device,
        max_neighbors=12 if settings['algo'] == 'macbf' else None
    )
    params = env.default_params
    if args.area_size is not None:
        params['area_size'] = args.area_size
    if args.obs is not None:
        params['num_obs'] = args.obs
    if args.sense_radius is not None:
        params['comm_radius'] = args.sense_radius
    env = make_env(
        env=settings['env'] if args.env is None else args.env,
        num_agents=settings['num_agents'] if args.num_agents is None else args.num_agents,
        device=device,
        params=params,
        max_neighbors=12 if settings['algo'] == 'macbf' else None
    )
    # env.demo(3)
    if args.demo is None:
        env.test()
    else:
        env.demo(args.demo)

    # build algorithm
    algo = make_algo(
        algo=settings['algo'],
        env=env,
        num_agents=settings['num_agents'] if args.num_agents is None else args.num_agents,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        action_dim=env.action_dim,
        device=device,
        hyperparams=settings['hyper_params'] if 'hyper_params' in settings.keys() else None
    )
    if args.path is None:
        assert args.env is not None and args.num_agents is not None
        # evaluate the nominal controller
        args.path = f'./logs/{args.env}'
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        if not os.path.exists(args.path):
            os.mkdir(args.path)
        if not os.path.exists(os.path.join(args.path, 'nominal')):
            os.mkdir(os.path.join(args.path, 'nominal'))
        video_path = os.path.join(args.path, 'nominal', 'videos')
    else:
        # evaluate the learned controller
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
        video_path = os.path.join(args.path, 'videos')

    # mkdir for the video and the figures
    if not args.no_video and not os.path.exists(video_path):
        os.mkdir(video_path)

    # evaluate policy
    def apply(data):
        return algo.apply(data, rand=args.rand)
    start_time = time.time()
    results = []
    for i in range(args.epi):
        print(f'epi: {i}')
        results.append(
            eval_ctrl_epi(apply, env, np.random.randint(100000), not args.no_video, plot_edge=not args.no_edge)
        )
    rewards, lengths, video, info = zip(*results)
    video = sum(video, ())

    # calculate safe rate and goal reaching rate
    safe_rates = []
    reach_rates = []
    success_rates = []
    n_traj = 0
    for i in info:
        if 'safe' in i.keys():
            safe_rates.append(float(i['safe']))
            n_traj += 1
        if 'reach' in i.keys():
            reach_rates.append(float(i['reach']))
        if 'success' in i.keys():
            success_rates.append(float(i['success']))

    # output trajectory
    if args.write_traj is not None:
        if args.write_traj == 'mat':
            if not os.path.exists(os.path.join(args.path, 'trajs')):
                os.mkdir(os.path.join(args.path, 'trajs'))
            for i, i_info in enumerate(info):
                savemat(
                    os.path.join(
                        args.path,
                        'trajs',
                        f'demo{args.demo}_seed{args.seed}_agent{env.num_agents}_size_{args.area_size}_'
                        f'safe{np.mean(safe_rates)}_reach{np.mean(reach_rates)}_success{np.mean(success_rates)}_reward{np.mean(rewards):.2f}_traj{i}.mat',
                    ),
                    {'states': i_info['states'].cpu().numpy()}
                )

    # make video
    if not args.no_video:
        print(f'> Making video...')
        out = cv2.VideoWriter(
            os.path.join(
                video_path,
                f'demo{args.demo}_seed{args.seed}_agent{env.num_agents}_size_{args.area_size}_'
                f'safe{np.mean(safe_rates)}_reach{np.mean(reach_rates)}_success{np.mean(success_rates)}_reward{np.mean(rewards):.2f}.mp4'
            ),
            cv2.VideoWriter_fourcc(*'mp4v'),
            25,
            (video[-1].shape[1], video[-1].shape[0])
        )

        # release the video
        for i_fig, fig in enumerate(video):
            if fig.shape[2] == 4:
                img_pil = Image.fromarray(fig, 'RGBA')
                img_rgb = img_pil.convert('RGB')
                img_rgb.save(f'{video_path}/tmp.png')
                fig = cv2.imread(f'{video_path}/tmp.png')
            out.write(np.uint8(fig))
        out.release()
        if os.path.exists(f'{video_path}/tmp.png'):
            os.remove(f'{video_path}/tmp.png')

    # print evaluation results
    verbose = f'average reward: {np.mean(rewards):.2f}, average length: {np.mean(lengths):.2f}'
    if n_traj > 0:
        verbose += f', safe rate: {np.mean(safe_rates)} +/- {np.std(safe_rates)}, ' \
                   f'reach rate: {np.mean(reach_rates)} +/- {np.std(reach_rates)}, ' \
                   f'success rate: {np.mean(success_rates)} +/- {np.std(success_rates)}'
    print(verbose)
    with open(os.path.join(args.path, 'test_log.csv'), "a") as f:
        f.write(f'{env.num_agents},{args.obs},{args.epi},{args.area_size},{np.mean(safe_rates)},{np.std(safe_rates)},'
                f'{np.mean(reach_rates)},{np.std(reach_rates)},{np.mean(success_rates)},{np.std(success_rates)}\n')
    print(f'> Done in {time.time() - start_time:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--obs', type=int, default=None)
    parser.add_argument('--sense-radius', type=float, default=None)
    parser.add_argument('--area-size', type=float, default=None)
    parser.add_argument('-n', '--num-agents', type=int, default=None)
    parser.add_argument('--demo', type=int, default=None)
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--iter', type=int, default=None)
    parser.add_argument('--epi', type=int, default=5)
    parser.add_argument('--no-video', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--no-edge', action='store_true', default=False)
    parser.add_argument('--write_traj', type=str, default=None)
    parser.add_argument('--rand', type=float, default=30)

    # default
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', default=False)

    args = parser.parse_args()
    test(args)
