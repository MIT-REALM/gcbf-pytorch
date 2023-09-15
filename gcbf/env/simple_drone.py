import numpy as np
import torch
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import os

from typing import Tuple, Optional, Union, List
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_geometric.utils import mask_to_index, index_to_mask
from cvxpy import Variable, Expression

from .utils import lqr, plot_graph_3d, create_cuboid, create_point_cloud
from .base import MultiAgentEnv


class SimpleDrone(MultiAgentEnv):
    """
    demo 2: limit the maximum travelling distance
    """

    def __init__(
            self,
            num_agents: int,
            device: torch.device,
            dt: float = 0.03,
            params: dict = None,
            max_neighbors: Optional[int] = None
    ):
        super(SimpleDrone, self).__init__(num_agents, device, dt, params, max_neighbors)

        # parameters for the reference controller
        self._K = None
        self._goal = None

        # obstacles
        self._num_obs = self._params['num_obs']
        self._obs = None  # obstacle point cloud
        self._obs_vertices = None  # obstacle vertices

        # parameters for plotting
        self._xyz_min = np.array([0, 0, 0])
        self._xyz_max = np.ones(3) * self._params['area_size']

    @property
    def state_dim(self) -> int:
        return 6  # [x, y, z, vx, vy, vz]

    @property
    def node_dim(self) -> int:
        return 4

    @property
    def edge_dim(self) -> int:
        return 6

    @property
    def action_dim(self) -> int:
        return 3  # [ax, ay, az]

    @property
    def max_episode_steps(self) -> int:
        if self._mode == 'train':
            return 500
        else:
            return 2000

    @property
    def default_params(self) -> dict:
        return {
            'area_size': 2.,
            'speed_limit': 0.6,  # maximum speed
            'drone_radius': 0.05,
            'comm_radius': 0.5,
            'dist2goal': 0.02,
            'obs_point_r': 0.05,
            'obs_len_max': 0.5,
            'max_distance': 4.0,
            'num_obs': 4,
        }

    @property
    def _A(self) -> Tensor:
        A = torch.zeros(6, 6, dtype=torch.float, device=self.device)
        A[0, 3] = 1.
        A[1, 4] = 1.
        A[2, 5] = 1.
        A[3, 3] = -1.1
        A[4, 4] = -1.1
        A[5, 5] = -6.
        return A

    @property
    def _B(self) -> Tensor:
        B = torch.zeros(6, 3, dtype=torch.float, device=self.device)
        B[3, 0] = 1.1
        B[4, 1] = 1.1
        B[5, 2] = 6.
        return B

    def dynamics(self, data: Data, u: Union[Tensor, Expression]) -> Union[Tensor, Expression]:
        if isinstance(u, Expression):
            A = self._A.cpu().numpy()
            B = self._B.cpu().numpy()
            xdot = data.states.cpu().detach().numpy() @ A.T + u @ B.T
        else:
            agent_mask = data.agent_mask
            xdot = data.states @ self._A.t()
            xdot[~agent_mask] = 0.
            xdot[agent_mask] += u @ self._B.t()
            if data.states.shape[0] == self.num_agents + self._obs.shape[0]:
                reach = torch.less(torch.norm(data.states[agent_mask, :3] - self._goal[:, :3], dim=1),
                                   self._params['dist2goal'])
                xdot[agent_mask] *= torch.logical_not(reach).unsqueeze(1).repeat(1, self.state_dim)
                return xdot
            else:
                return xdot
        return xdot

    def reset(self) -> Data:
        self._t = 0
        states = torch.zeros(self.num_agents, 6, device=self.device)
        goals = torch.zeros(self.num_agents, 6, device=self.device)

        if self._mode == 'train' or self._mode == 'test':
            # generate obstacles
            i = 0
            obs_pos = torch.zeros(self.num_agents, 3, device=self.device)
            while i < self.num_agents:
                obs_pos[i] = torch.rand(3, device=self.device) * self._params['area_size']
                i += 1
            self._obs = torch.zeros(obs_pos.shape[0], self.state_dim, device=self.device)
            self._obs[:, :3] = obs_pos

            # generate agents
            i = 0
            while i < self.num_agents:
                candidate = torch.rand(1, 3).type_as(states) * self._params['area_size']
                dist_min = torch.norm(states[:, :3] - candidate, dim=1).min()
                if dist_min <= self._params['drone_radius'] * 4:
                    continue
                dist_min = torch.norm(obs_pos - candidate, dim=1).min()
                if dist_min <= self._params['drone_radius'] * 2 + self._params['obs_point_r'] * 2:
                    continue
                states[i, :3] = candidate
                i += 1

            # generate goals
            i = 0
            while i < self.num_agents:
                if self._mode == 'demo_2':
                    candidate = (torch.rand(3, device=self.device) * 2 - 1) * self._params['max_distance'] \
                                + states[i, :3]
                    if (candidate > self._params['area_size']).any() or (candidate < 0).any():
                        continue
                else:
                    candidate = torch.rand(1, 3).type_as(states) * self._params['area_size']
                dist_min = torch.norm(goals[:, :3] - candidate, dim=1).min()
                if dist_min <= self._params['drone_radius'] * 4:
                    continue
                dist_min = torch.norm(obs_pos - candidate, dim=1).min()
                if dist_min <= self._params['drone_radius'] * 2 + self._params['obs_point_r'] * 2:
                    continue
                goals[i, :3] = candidate
                i += 1

            # build graph
            data = Data(
                x=torch.cat(
                    (torch.zeros(self.num_agents, self.node_dim), torch.ones(self._obs.shape[0], self.node_dim)),
                    dim=0
                ).type_as(states),
                pos=torch.cat((states[:, :3], obs_pos), dim=0).type_as(states),
                states=torch.cat((states, self._obs), dim=0).type_as(states),
                agent_mask=index_to_mask(torch.arange(self.num_agents, device=self.device),
                                         size=states.shape[0] + self._obs.shape[0])
            )
        else:
            raise NotImplementedError

        # record goals
        self._goal = goals

        data = self.add_communication_links(data)
        self._data = data

        return data

    def step(self, action: Tensor) -> Tuple[Data, float, bool, dict]:
        self._t += 1

        # calculate next state using dynamics
        reward_action = -torch.norm(action, dim=1) * 0.001
        action = action + self.u_ref(self._data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        prev_reach = torch.less(
            torch.norm(self.data.states[self.data.agent_mask, :3] - self._goal[:, :3], dim=1),
            self._params['dist2goal'])
        with torch.no_grad():
            state = self.forward(self._data, action)

        if self._mode == 'train' or self._mode == 'test' or self._mode == 'demo_2':
            data = Data(
                x=torch.cat(
                    (torch.zeros(self.num_agents, self.node_dim), torch.ones(self._obs.shape[0], self.node_dim)), dim=0
                ).type_as(state),
                pos=state[:, :3],
                states=state,
                agent_mask=self._data.agent_mask
            )
        else:
            raise NotImplementedError
        self._data = self.add_communication_links(data)

        # the episode ends when reaching max_episode_steps or all the agents reach the goal
        time_up = self._t >= self.max_episode_steps
        reach = torch.less(
            torch.norm(self.data.states[self.data.agent_mask, :3] - self._goal[:, :3], dim=1),
            self._params['dist2goal'])
        done = time_up or reach.all()

        # reward function
        reward_step = -0.01
        reward_collision = -self.collision_mask(data).int()
        reward_reach = (reach.int() - prev_reach.int()) * 10
        reward = reward_reach + reward_collision + reward_step + reward_action

        safe = float(1.0 - self.collision_mask(data).sum() / self.num_agents)
        collision_agent = torch.where(self.collision_mask(data) > 0)[0]
        return self.data, reward.cpu().detach().numpy(), done, {'safe': safe, 'reach': reach,
                                                                'collision': collision_agent}

    def forward_graph(self, data: Data, action: Tensor) -> Data:
        # calculate next state using dynamics
        action = action + self.u_ref(data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        state = self.forward(data, action)

        # construct the graph of the next step, retaining the connection
        data_next = Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=self.edge_attr(state, data.edge_index),
            pos=state[:, :3],
            states=state,
            agent_mask=data.agent_mask
        )

        return data_next

    def render(self, traj: Optional[Tuple[Data, ...]] = None, return_ax: bool = False, plot_edge: bool = True, ax=None
               ) -> Union[Tuple[np.array, ...], np.array]:
        return_tuple = True
        gif = []

        if traj is None:
            return_tuple = False
            if self._mode == 'train' or self._mode == 'test' or self._mode == 'demo_2':
                data = self.data
                traj = (data,)
            else:
                raise NotImplementedError

        if self._mode == 'train' or self._mode == 'test' or self._mode == 'demo_2':
            r = self._params['drone_radius']

            for data in traj:
                # 3D plot
                fig = plt.figure(figsize=(10, 10), dpi=80)
                ax = fig.add_subplot(projection='3d')

                # plot the drones and the communication network
                plot_graph_3d(ax, data, radius=r, color='#FF8C00', with_label=True,
                              plot_edge=plot_edge, alpha=0.3)

                # plot the goals
                goal_data = Data(pos=self._goal[:, :3])
                plot_graph_3d(ax, goal_data, radius=r, color='#3CB371', with_label=True,
                              plot_edge=False, alpha=0.3)

                # texts
                fontsize = 14
                collision_text = ax.text2D(0., 0.97, "", transform=ax.transAxes, fontsize=fontsize)
                unsafe_index = mask_to_index(self.unsafe_mask(data))
                collision_text.set_text(f'Collision: {unsafe_index.cpu().detach().numpy()}')

                # set axis limit
                ax.set_xlim(self._xyz_min[0], self._xyz_max[0])
                ax.set_ylim(self._xyz_min[1], self._xyz_max[1])
                ax.set_zlim(self._xyz_min[2], self._xyz_max[2])
                ax.set_aspect('equal')
                # plt.axis('off')

                if return_ax:
                    return ax

                # convert to numpy array
                fig.canvas.draw()
                fig_np = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
                gif.append(fig_np)
                plt.close()

        if return_tuple:
            return tuple(gif)
        else:
            return gif[0]

    def edge_attr(self, state: Tensor, edge_index: Tensor) -> Tensor:
        return state[edge_index[0]] - state[edge_index[1]]

    def add_communication_links(self, data: Data) -> Data:
        pos_diff = data.pos.unsqueeze(1) - data.pos.unsqueeze(0)  # [i, j]: j -> i
        dist = torch.norm(pos_diff, dim=-1)[data.agent_mask]
        dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=self.device) * (self._params['comm_radius'] + 1)

        # filter out top k neighbors
        if self._max_neighbors is not None:
            _, dist_id = torch.topk(dist, self._max_neighbors, dim=-1, largest=False)
            for i in range(dist_id.shape[0]):
                neighbor_mask = index_to_mask(dist_id[i], dist.shape[1])
                dist[i, ~neighbor_mask] += self._params['comm_radius'] + 1

        dist_mask = torch.less(dist, self._params['comm_radius'])
        edge_index = torch.nonzero(dist_mask, as_tuple=False).t()[[1, 0]]

        edge_attr = self.edge_attr(data.states, edge_index)
        data.update(Data(edge_index=edge_index, edge_attr=edge_attr))
        return data

    @property
    def state_lim(self) -> Tuple[Tensor, Tensor]:
        low_lim = torch.tensor(
            [self._xyz_min[0], self._xyz_min[1], self._xyz_min[2], -10, -10, -10], device=self.device)
        high_lim = torch.tensor(
            [self._xyz_max[0], self._xyz_max[1], self._xyz_max[2], 10, 10, 10], device=self.device)
        return low_lim, high_lim

    @property
    def action_lim(self) -> Tuple[Tensor, Tensor]:
        upper_limit = torch.ones(self.action_dim, device=self.device) * 10.0
        lower_limit = - upper_limit
        return lower_limit, upper_limit

    def u_ref(self, data: Data) -> Tensor:
        states = data.states[data.agent_mask]
        states = states.reshape(-1, self.num_agents, self.state_dim)
        diff = states - self._goal

        if self._K is None:
            # get used A, B, Q, R
            A = self._A.cpu().numpy() * self.dt + np.eye(self.state_dim)
            B = self._B.cpu().numpy() * self.dt
            Q = np.eye(self.state_dim)
            R = np.eye(self.action_dim)
            K_np = lqr(A, B, Q, R)
            self._K = torch.from_numpy(K_np).type_as(data.states)

        # feedback control
        action = - torch.einsum('us,bns->bnu', self._K, diff)
        action = action.reshape(-1, self.action_dim)

        # adapt to speed limit
        states = states.reshape(-1, self.state_dim)
        over_speed_agent = torch.where(states[:, 3:].norm(dim=1) - self._params['speed_limit'] > 0)[0]
        if over_speed_agent.shape[0] > 0:
            # add penalty to the action
            v_over_speed = states[over_speed_agent, 3:]
            v_dir = v_over_speed / v_over_speed.norm(dim=1, keepdim=True)
            action[over_speed_agent] -= \
                (v_over_speed.norm(dim=1, keepdim=True) - self._params['speed_limit']) * v_dir * 10

        return action

    def safe_mask(self, data: Union[Data, Batch], return_edge: bool = False) -> Tensor:
        if return_edge:
            pos_diff = data.edge_attr[:, :3]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=-1)
            safe = torch.greater(dist, 4 * self._params['drone_radius'])
            return safe

        if isinstance(data, Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]

        mask = []
        for graph in data_list:
            state_diff = graph.states.unsqueeze(1) - graph.states.unsqueeze(0)
            pos_diff = state_diff[graph.agent_mask, :, :3]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=2)
            dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=self.device) * (
                    4 * self._params['drone_radius'] + 1)
            safe = torch.greater(dist, 4 * self._params['drone_radius'])
            mask.append(torch.min(safe, dim=1)[0])
        mask = torch.cat(mask, dim=0).bool()

        return mask

    def unsafe_mask(self, data: Union[Data, Batch], return_edge: bool = False) -> Tensor:
        if return_edge:
            pos_diff = data.edge_attr[:, :3]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=-1)
            collision = torch.less(dist, 2 * self._params['drone_radius'])
            return collision

        warn_dist = 4 * self._params['drone_radius']
        if isinstance(data, Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]
        mask = []
        for graph in data_list:
            # collision
            state_diff = graph.states.unsqueeze(1) - graph.states.unsqueeze(0)
            pos_diff = state_diff[graph.agent_mask, :, :3]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=2)
            dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=self.device) * (
                    2 * self._params['drone_radius'] + 1)
            collision = torch.less(dist, 2 * self._params['drone_radius'])
            graph_mask = torch.max(collision, dim=1)[0]

            # unsafe direction
            warn_zone = torch.less(dist, warn_dist)
            pos_vec = -(pos_diff / (torch.norm(pos_diff, dim=2, keepdim=True) + 0.0001))  # [i, j]: i -> j
            v = graph.states[graph.agent_mask, 3:].norm(dim=1, keepdim=True) + 0.00001
            theta_vec = torch.cat(
                [graph.states[graph.agent_mask, 3].unsqueeze(1) / v, graph.states[graph.agent_mask, 4].unsqueeze(1) / v,
                 graph.states[graph.agent_mask, 5].unsqueeze(1)], dim=1
            ).repeat(pos_vec.shape[1], 1, 1).transpose(0, 1)  # [i, j]: theta[i]
            inner_prod = torch.sum(pos_vec * theta_vec, dim=2)
            unsafe_threshold = torch.cos(torch.asin(self._params['drone_radius'] * 2 / (dist + 0.0000001)))
            unsafe = torch.greater(inner_prod, unsafe_threshold)
            unsafe = torch.max(torch.logical_and(unsafe, warn_zone), dim=1)[0]
            graph_mask = torch.logical_or(graph_mask, unsafe)

            mask.append(graph_mask)
        mask = torch.cat(mask, dim=0).bool()

        return mask

    def collision_mask(self, data: Data) -> Tensor:
        if isinstance(data, Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]

        if self._mode == 'train' or self._mode == 'test' or self._mode == 'demo_2':
            mask = []
            for graph in data_list:
                state_diff = graph.states.unsqueeze(1) - graph.states.unsqueeze(0)
                pos_diff = state_diff[graph.agent_mask, :, :3]  # [i, j]: j -> i
                dist = pos_diff.norm(dim=2)
                dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=self.device) * (
                        2 * self._params['drone_radius'] + 1)
                collision = torch.less(dist, 2 * self._params['drone_radius'])
                mask.append(torch.max(collision, dim=1)[0])
        else:
            raise NotImplementedError
        mask = torch.cat(mask, dim=0).bool()
        return mask
