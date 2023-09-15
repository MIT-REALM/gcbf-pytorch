import os
import numpy as np
import torch

from typing import Tuple
from time import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch_geometric.data import Data

from gcbf.env import MultiAgentEnv
from gcbf.algo import Algorithm


class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: Algorithm,
            log_dir: str
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.log_dir = log_dir

        # make dir for the models
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.model_dir = os.path.join(log_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        # set up log writer
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)

    def train(self, steps: int, eval_interval: int, eval_epi: int):
        """
        Start training

        Parameters
        ----------
        steps: int,
            number of training steps
        eval_interval: int,
            interval of steps between evaluations
        eval_epi: int,
            number of episodes for evaluation
        """
        # record start time
        start_time = time()

        # reset the environment
        data = self.env.reset()

        verbose = None
        for step in tqdm(range(1, steps + 1), ncols=80):
            data.update(Data(u_ref=self.env.u_ref(data)))
            action = self.algo.step(data, prob=1 - (step - 1) / steps)
            next_data, reward, done, info = self.env.step(action)
            next_data.update(Data(u_ref=self.env.u_ref(next_data)))
            self.algo.post_step(data, action, reward, done, next_data)
            if done:
                data = self.env.reset()
            else:
                data = next_data

            # update the algorithm
            if self.algo.is_update(step):
                verbose = self.algo.update(step, self.writer)

            # evaluate the algorithm
            if step % eval_interval == 0:
                if eval_epi > 0:
                    if eval_epi > 0:
                        reward, eval_info = self.eval(step, eval_epi)
                        eval_verbose = f'step: {step}, time: {time() - start_time:.0f}s, reward: {reward:.2f}'
                        if len(eval_info.keys()) > 0:
                            for key in eval_info.keys():
                                eval_verbose += f', {key}: {eval_info[key]}'
                        tqdm.write(eval_verbose)
                if verbose is not None:
                    verbose_update = f'step: {step}'
                    for key in verbose.keys():
                        verbose_update += f', {key}: {verbose[key]:.3f}'
                    tqdm.write(verbose_update)
                self.algo.save(os.path.join(self.model_dir, f'step_{step}'))
                self.algo._env = self.env

        print(f'> Done in {time() - start_time:.0f} seconds')

    def eval(self, step: int, eval_epi: int) -> Tuple[float, dict]:
        """
        Evaluate the current model

        Parameters
        ----------
        step: int,
            current training step
        eval_epi: int,
            number of episodes for evaluation

        Returns
        -------
        reward: float
            average episode reward
        info: dict
            other information
        """
        rewards = []
        safe_rate = []
        reach = 0
        self.algo._env = self.env_test
        for i_epi in range(eval_epi):
            safe_agent = torch.ones(self.env_test.num_agents).bool()
            data = self.env_test.reset()
            epi_reward = 0.
            safe = True
            while True:
                data.update(Data(u_ref=self.env_test.u_ref(data)))
                action = self.algo.apply(data)
                data, reward, done, info = self.env_test.step(action)
                epi_reward += np.mean(reward)
                if 'collision' in info.keys():
                    safe_agent[info['collision']] = False
                if 'reach' in info.keys():
                    reach = info['reach']
                if done:
                    break

            rewards.append(epi_reward)
            safe_rate.append(safe_agent.sum().item() / self.env_test.num_agents)

        self.writer.add_scalar('test/reward', np.mean(rewards), step)
        self.writer.add_scalar('test/safe_rate', np.mean(safe_rate), step)

        return np.mean(rewards).item(), {'safe': round(float(np.mean(safe_rate)), 2),
                                         'reach': round(float(torch.mean(reach.float())), 2)}
