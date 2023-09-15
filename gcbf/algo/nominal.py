import torch

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from typing import Optional

from gcbf.env import MultiAgentEnv
from gcbf.controller import NominalController

from .base import Algorithm


class Nominal(Algorithm):
    def __init__(
            self,
            env: MultiAgentEnv,
            num_agents: int,
            node_dim: int,
            edge_dim: int,
            action_dim: int,
            device: torch.device,
    ):
        super(Nominal, self).__init__(
            env=env,
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            device=device
        )
        self.actor = NominalController(
            num_agents=self.num_agents,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            action_dim=self.action_dim
        ).to(device)

    def step(self, data: Data, prob: float) -> Tensor:
        raise NotImplementedError

    def is_update(self, step: int) -> bool:
        raise NotImplementedError

    def update(self, step: int, writer: SummaryWriter = None):
        raise NotImplementedError

    def save(self, save_dir: str):
        raise NotImplementedError

    def load(self, load_dir: str):
        raise NotImplementedError

    def act(self, data: Data) -> Tensor:
        with torch.no_grad():
            return self.actor(data)

    def apply(self, data: Data, rand: Optional[float] = 30) -> Tensor:
        return self.act(data)
