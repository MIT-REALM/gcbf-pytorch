import numpy as np
import torch

from torch import Tensor
from abc import ABC, abstractmethod, abstractproperty
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from typing import Optional

from gcbf.env import MultiAgentEnv


class Algorithm(ABC):

    def __init__(
            self,
            env: MultiAgentEnv,
            num_agents: int,
            node_dim: int,
            edge_dim: int,
            action_dim: int,
            device: torch.device
    ):
        super(Algorithm, self).__init__()
        self._env = env
        self._num_agents = num_agents
        self._node_dim = node_dim
        self._edge_dim = edge_dim
        self._action_dim = action_dim
        self._device = device
        self.params = {}

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def node_dim(self) -> int:
        return self._node_dim

    @property
    def edge_dim(self) -> int:
        return self._edge_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    @torch.no_grad()
    def act(self, data: Data) -> Tensor:
        """
        Get actions using the current controller without gradient.
        Note: remember to use torch.no_grad() to disable the gradients.

        Parameters
        ----------
        data: Data,
            data of the current multi-agent map or batched data using Batch.from_data_list()

        Returns
        -------
        action: (bs, n, action_dim)
            actions of all the agents
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def step(self, data: Data, prob: float) -> Tensor:
        """
        Do one step forward in training. Different from act() by doing necessary steps for training.

        Parameters
        ----------
        data: Data,
            data of the current multi-agent map or batched data using Batch.from_data_list()
        prob: float,
            probability to choose the nominal controller

        Returns
        -------
        action: (bs, n, action_dim),
            actions of all the agents
        """
        pass

    def post_step(self, data: Data, action: Tensor, reward: float, done: bool, next_data: Data):
        pass

    def sample(self, data: Data, prob: float = 0.01) -> Tensor:
        """
        Get actions using the current controller with noise

        Parameters
        ----------
        data: Data,
            data of the current multi-agent map or batched data using Batch.from_data_list()
        prob: float,
            probability of adding noise

        Returns
        -------
        action: (bs, n, action_dim),
            actions of all the agents
        """
        actions = self.act(data)
        action_lim = self._env.action_lim
        if np.random.uniform() < prob:
            noise = torch.randn_like(actions) * 0.3 * (action_lim[1] - action_lim[0])
            actions += noise
        return actions

    @abstractmethod
    def is_update(self, step: int) -> bool:
        """
        Judge if the model is ready to be updated

        Parameters
        ----------
        step: int,
            current training step

        Returns
        -------
        update: bool,
            time to update or not
        """
        pass

    @abstractmethod
    def update(self, step: int, writer: SummaryWriter = None) -> dict:
        """
        Update the models

        Parameters
        ----------
        step: int,
            current training step
        writer: SummaryWriter,
            writer for the logs
        """
        pass

    @abstractmethod
    def save(self, save_dir: str):
        """
        Save models in save_dir

        Parameters
        ----------
        save_dir: str,
            folder to save the models
        """
        pass

    @abstractmethod
    def load(self, load_dir: str):
        """
        Load the models from load_dir

        Parameters
        ----------
        load_dir: str,
            folder to load the models
        """
        pass

    def apply(self, data: Data, rand: Optional[float] = 30) -> Tensor:
        """
        Apply the agent in the environment during test

        Parameters
        ----------
        data: Data,
            data of the current multi-agent map or batched data using Batch.from_data_list()
        rand: float,
            coefficient of noise in action gradient

        Returns
        -------
        action: (bs, n, action_dim)
            actions of all the agents
        """
        pass
