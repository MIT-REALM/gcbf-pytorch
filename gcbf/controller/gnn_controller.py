import torch

from torch_geometric.nn import Sequential
from torch import Tensor
from torch_geometric.data import Data

from gcbf.nn.gnn import ControllerGNNLayer
from gcbf.nn.mlp import MLP

from .base import MultiAgentController


class GNNController(MultiAgentController):

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, phi_dim: int, action_dim: int):
        super(GNNController, self).__init__(
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim
        )

        self.feat_transformer = Sequential('x, edge_attr, edge_index', [
            (ControllerGNNLayer(node_dim=node_dim, edge_dim=edge_dim, output_dim=1024, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
        ])
        self.feat_2_action = MLP(in_channels=1024+action_dim, out_channels=action_dim, hidden_layers=(512, 128, 32))

    def forward(self, data: Data) -> Tensor:
        """
        Get the control actions for the input states.

        Parameters
        ----------
        data: Data,
            batched data using Batch.from_data_list().

        Returns
        -------
        actions: (bs x n, action_dim)
            control actions for all agents
        """
        x = self.feat_transformer(data.x, data.edge_attr, data.edge_index)
        if hasattr(data, 'agent_mask'):
            x = x[data.agent_mask]
        actions = self.feat_2_action(torch.cat([x, data.u_ref], dim=1))

        return actions
