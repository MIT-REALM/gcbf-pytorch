import torch.nn as nn

from abc import ABC, abstractmethod
from torch_geometric.data import Data
from torch import Tensor


class MultiAgentController(nn.Module, ABC):

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, action_dim: int):
        super(MultiAgentController, self).__init__()
        self._node_dim = node_dim
        self._edge_dim = edge_dim
        self._action_dim = action_dim
        self._num_agents = num_agents

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

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        """
        Get the control actions for the input states.

        Parameters
        ----------
        data: Data
            batched data using Batch.from_data_list().

        Returns
        -------
        actions: (bs x n, action_dim)
            control actions for all agents
        """
        pass
