import torch.nn as nn
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.nn import Sequential
from torch import Tensor
from torch.optim import Adam
from typing import Optional

from gcbf.nn import CBFNetLayer
from gcbf.controller import MACBFController
from gcbf.env import MultiAgentEnv

from .buffer import Buffer
from .gcbf import GCBF


class CBFNet(nn.Module):

    def __init__(self, num_agents: int, node_dim: int, edge_dim: int):
        super(CBFNet, self).__init__()
        self._num_agents = num_agents
        self._top_k = 12
        self.net = Sequential('x, edge_attr, edge_index', [
            (CBFNetLayer(node_dim=node_dim, edge_dim=edge_dim, output_dim=1),
             'x, edge_attr, edge_index -> x'),
        ])

    def forward(self, data: Data) -> Tensor:
        """
        Get the pari-wise CBF value for the input states.

        Parameters
        ----------
        data: Data,
            reconstructed data using top k neighbors

        Returns
        -------
        h: Tensor (bs x n, k)
            pair-wise CBF values for all agents
        """
        h = self.net(data.x, data.edge_attr, data.edge_index)
        # if hasattr(data, 'agent_mask'):
        #     h = h[data.agent_mask]
        return h


class MACBF(GCBF):

    def __init__(
            self,
            env: MultiAgentEnv,
            num_agents: int,
            node_dim: int,
            edge_dim: int,
            action_dim: int,
            device: torch.device,
            batch_size: int = 500,
            params: Optional[dict] = None
    ):
        super(MACBF, self).__init__(
            env=env,
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            device=device
        )

        # models
        self.cbf = CBFNet(num_agents=num_agents, node_dim=node_dim, edge_dim=edge_dim).to(device)
        self.actor = MACBFController(
            num_agents=num_agents,
            node_dim=node_dim,
            edge_dim=edge_dim,
            phi_dim=128,
            action_dim=action_dim
        ).to(device)

        # optimizer
        self.optim_cbf = Adam(self.cbf.parameters(), lr=3e-4)
        self.optim_actor = Adam(self.actor.parameters(), lr=1e-3)

        # buffer to store data used in training
        self.buffer = Buffer()  # buffer for current episode
        self.memory = Buffer()  # replay buffer
        self.batch_size = batch_size

        # hyperparams
        if params is None:
            self.params = {  # default hyper-parameters
                'alpha': 1.0,
                'eps': 0.02,
                'inner_iter': 10,
                'loss_action_coef': 0.001,
                'loss_unsafe_coef': 1.,
                'loss_safe_coef': 1.,
                'loss_h_dot_coef': 0.1,
            }
        else:
            self.params = params

    @torch.no_grad()
    def step(self, data: Data, prob: float) -> Tensor:
        action = self.actor(data)
        prob = max(prob, 0.5)
        if np.random.rand() < prob:
            action = torch.zeros_like(action)
        is_safe = True
        if torch.any(self._env.unsafe_mask(data)):
            is_safe = not is_safe

        self.buffer.append(data, is_safe)

        return action

    def update(self, step: int, writer: SummaryWriter = None) -> dict:
        seg_len = 3  # pls use odd number
        acc_safe = torch.zeros(1, dtype=torch.float)
        acc_unsafe = torch.zeros(1, dtype=torch.float)
        acc_h_dot = torch.zeros(1, dtype=torch.float)
        for i_inner in range(self.params['inner_iter']):
            # sample from the current buffer and the memory
            if self.memory.size == 0:
                graph_list = self.buffer.sample(self.batch_size // 5, seg_len)
            else:
                curr_graphs = self.buffer.sample(self.batch_size // 10, seg_len, True)
                prev_graphs = self.memory.sample(self.batch_size // 5 - self.batch_size // 10, seg_len, True)
                graph_list = curr_graphs + prev_graphs

            # get CBF values and the control inputs
            graphs = Batch.from_data_list(graph_list)
            graphs.edge_attr.requires_grad = True
            h = self.cbf(graphs)
            actions = self.actor(graphs)

            # calculate loss
            eps = self.params['eps']

            # unsafe region h(x) < 0
            unsafe_mask = self._env.unsafe_mask(graphs, return_edge=True)
            h_unsafe = h[unsafe_mask]
            if h_unsafe.numel():
                max_val_unsafe = torch.relu(h_unsafe + eps)
                loss_unsafe = torch.mean(max_val_unsafe)
                acc_unsafe = torch.mean(torch.less(h_unsafe, 0).type_as(h_unsafe))

            else:
                loss_unsafe = torch.tensor(0.0).type_as(h_unsafe)
                acc_unsafe = torch.tensor(1.0).type_as(h_unsafe)

            # safe region h(x) > 0
            safe_mask = self._env.safe_mask(graphs, return_edge=True)
            h_safe = h[safe_mask]
            if h_safe.numel():
                max_val_safe = torch.relu(-h_safe + eps)
                loss_safe = torch.mean(max_val_safe)
                acc_safe = torch.mean(torch.greater_equal(h_safe, 0).type_as(h_safe))

            else:
                loss_safe = torch.tensor(0.0).type_as(h_unsafe)
                acc_safe = torch.tensor(1.0).type_as(h_unsafe)

            # derivative loss h_dot + \alpha h > 0
            graphs_next = self._env.forward_graph(graphs, actions)
            h_next = self.cbf(graphs_next)
            h_dot = (h_next - h) / self._env.dt
            max_val_h_dot = torch.relu((-h_dot - self.params['alpha'] * h + eps))
            loss_h_dot = torch.mean(max_val_h_dot)
            acc_h_dot = torch.mean(torch.greater_equal((h_dot + self.params['alpha'] * h), 0).type_as(h_dot))

            # action loss
            loss_action = torch.mean(torch.square(actions).sum(dim=1))

            # backpropagation
            loss = (self.params['loss_unsafe_coef'] * loss_unsafe +
                    self.params['loss_safe_coef'] * loss_safe +
                    self.params['loss_h_dot_coef'] * loss_h_dot +
                    self.params['loss_action_coef'] * loss_action)

            self.optim_cbf.zero_grad(set_to_none=True)
            self.optim_actor.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cbf.parameters(), 1e-3)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1e-3)
            self.optim_cbf.step()
            self.optim_actor.step()

            # save loss
            writer.add_scalar('loss/unsafe', loss_unsafe.item(), step * self.params['inner_iter'] + i_inner)
            writer.add_scalar('loss/safe', loss_safe.item(), step * self.params['inner_iter'] + i_inner)
            writer.add_scalar('loss/derivative', loss_h_dot.item(), step * self.params['inner_iter'] + i_inner)
            writer.add_scalar('loss/action', loss_action.item(), step * self.params['inner_iter'] + i_inner)

            # save accuracy
            writer.add_scalar('acc/unsafe', acc_unsafe.item(), step * self.params['inner_iter'] + i_inner)
            writer.add_scalar('acc/safe', acc_safe.item(), step * self.params['inner_iter'] + i_inner)
            writer.add_scalar('acc/derivative', acc_h_dot.item(), step * self.params['inner_iter'] + i_inner)

        # merge the current buffer to the memory
        self.memory.merge(self.buffer)
        self.buffer.clear()

        return {
            'acc/safe': acc_safe.item(),
            'acc/unsafe': acc_unsafe.item(),
            'acc/derivative': acc_h_dot.item(),
        }

    def apply(self, data: Data, rand: Optional[float] = 0) -> Tensor:
        lr = 1  # previously 0.1
        h = self.cbf(data).detach()
        action = self.actor(data).detach()
        nominal = torch.zeros_like(action)

        optim = Adam((action,), lr=lr)

        # consider the satisfaction of h_dot condition
        i_iter = 0
        max_iter = 30
        while True:
            data_next = self._env.forward_graph(data, action)
            h_next = self.cbf(data_next)
            h_dot = (h_next - h) / self._env.dt
            max_val_h_dot = torch.relu((-h_dot - self.params['alpha'] * h))
            loss_h_dot = torch.mean(max_val_h_dot)
            if loss_h_dot <= 0 or i_iter > max_iter:
                break
            else:
                optim.zero_grad(set_to_none=True)
                loss_h_dot.backward()
                optim.step()

                i_iter += 1

        return action
