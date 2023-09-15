import torch

from torch_geometric.nn import Sequential
from torch import Tensor
from torch_geometric.data import Data

from gcbf.nn.gnn import MACBFControllerLayer
from gcbf.nn.mlp import MLP

from .base import MultiAgentController


class MACBFController(MultiAgentController):

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int, phi_dim: int, action_dim: int):
        super(MACBFController, self).__init__(
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim
        )

        self.net = Sequential('x, edge_attr, edge_index', [
            (MACBFControllerLayer(node_dim=node_dim, edge_dim=edge_dim, output_dim=action_dim, phi_dim=phi_dim),
             'x, edge_attr, edge_index -> x'),
        ])
        self.feat_2_action = MLP(in_channels=2*action_dim, out_channels=action_dim, hidden_layers=(512, 128, 32))

    def forward(self, data: Data) -> Tensor:
        """
        Get the control actions for the input states.

        Parameters
        ----------
        data: Data,
            reconstructed data using top k neighbors.

        Returns
        -------
        actions: (bs x n, action_dim)
            control actions for all agents
        """
        x = self.net(data.x, data.edge_attr, data.edge_index)
        if hasattr(data, 'agent_mask'):
            x = x[data.agent_mask]
        actions = self.feat_2_action(torch.cat([x, data.u_ref], dim=1))

        return actions
