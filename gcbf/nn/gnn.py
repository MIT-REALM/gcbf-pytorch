import torch

from typing import Optional, Tuple
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch import Tensor, cat
from torch_sparse import SparseTensor
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torch_geometric.data import Data
from torch_geometric.utils import softmax

from .mlp import MLP


class CBFGNNLayer(MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int):
        super(CBFGNNLayer, self).__init__(aggr=AttentionalAggregation(
            gate_nn=MLP(in_channels=phi_dim, out_channels=1, hidden_layers=(128, 128), limit_lip=False)
        ))
        self.phi = MLP(
            in_channels=2 * node_dim + edge_dim, out_channels=phi_dim, hidden_layers=(2048, 2048), limit_lip=True
        )
        self.gamma = MLP(
            in_channels=phi_dim + node_dim, out_channels=output_dim, hidden_layers=(2048, 2048), limit_lip=True
        )

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor = None, edge_attr: Tensor = None) -> Tensor:
        info_ij = cat([x_i, x_j, edge_attr], dim=1)
        return self.phi(info_ij)

    def update(self, aggr_out: Tensor, x: Tensor = None) -> Tensor:
        gamma_input = cat([aggr_out, x], dim=1)
        return self.gamma(gamma_input)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    def edge_update(self) -> Tensor:
        raise NotImplementedError

    def attention(self, data: Data):
        kwargs = {'x': data.x, 'edge_attr': data.edge_attr}
        size = self._check_input(data.edge_index, None)
        coll_dict = self._collect(self._user_args, data.edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        out = self.message(**msg_kwargs)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        gate = self.aggr_module.gate_nn(out)
        attention = softmax(gate, aggr_kwargs['index'], aggr_kwargs['ptr'], aggr_kwargs['dim_size'], dim=-2)
        return attention


class ControllerGNNLayer(MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int):
        super(ControllerGNNLayer, self).__init__(aggr=AttentionalAggregation(
            gate_nn=MLP(in_channels=phi_dim, out_channels=1, hidden_layers=(128, 128))))
        self.phi = MLP(in_channels=2 * node_dim + edge_dim, out_channels=phi_dim, hidden_layers=(2048, 2048))
        self.gamma = MLP(in_channels=phi_dim + node_dim, out_channels=output_dim, hidden_layers=(2048, 2048))

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor = None, edge_attr: Tensor = None) -> Tensor:
        info_ij = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.phi(info_ij)

    def update(self, aggr_out: Tensor, x: Tensor = None) -> Tensor:
        gamma_input = torch.cat([aggr_out, x], dim=1)
        return self.gamma(gamma_input)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    def edge_update(self) -> Tensor:
        raise NotImplementedError


class CBFNetLayer(MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int):
        super(CBFNetLayer, self).__init__(aggr=None)
        self.phi = MLP(
            in_channels=2 * node_dim + edge_dim, out_channels=output_dim, hidden_layers=(64, 128, 64), limit_lip=False
        )

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor = None, edge_attr: Tensor = None) -> Tensor:
        info_ij = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.phi(info_ij)

    def update(self, aggr_out: Tensor, x: Tensor = None) -> Tensor:
        return aggr_out

    def propagate(self, edge_index: Tensor, size: Optional[Tuple[int, int]] = None, **kwargs) -> Tensor:
        size = self._check_input(edge_index, None)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        out = self.message(**msg_kwargs)
        return out

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    def edge_update(self) -> Tensor:
        raise NotImplementedError


class MACBFControllerLayer(MessagePassing):

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int):
        super(MACBFControllerLayer, self).__init__(aggr='max')
        self.phi = MLP(in_channels=2 * node_dim + edge_dim, out_channels=phi_dim, hidden_layers=(64,))
        self.gamma = MLP(in_channels=phi_dim, out_channels=output_dim, hidden_layers=(64, 128, 64))

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, x_i: Tensor = None, edge_attr: Tensor = None) -> Tensor:
        info_ij = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.phi(info_ij)

    def update(self, aggr_out: Tensor, x: Tensor = None) -> Tensor:
        return self.gamma(aggr_out)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        raise NotImplementedError

    def edge_update(self) -> Tensor:
        raise NotImplementedError
