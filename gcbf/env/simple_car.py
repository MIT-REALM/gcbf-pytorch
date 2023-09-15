import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Tuple, Optional, Union
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_geometric.utils import mask_to_index
from cvxpy import Expression

from .utils import lqr, plot_graph
from .base import MultiAgentEnv


class SimpleCar(MultiAgentEnv):
    """
    demo 2: limit the maximum travelling distance
    """

    def __init__(
            self,
            num_agents: int,
            device: torch.device,
            dt: float = 0.03,
            params: Optional[dict] = None,
            max_neighbors: Optional[int] = None
    ):
        super(SimpleCar, self).__init__(num_agents, device, dt, params, max_neighbors)

        # builder of the graph
        self._builder = RadiusGraph(self._params['comm_radius'],
                                    max_num_neighbors=self.num_agents if max_neighbors is None else max_neighbors)

        # parameters for the reference controller
        self._K = None
        self._goal = None

        # parameters for plotting
        self._xy_min = None
        self._xy_max = None

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def node_dim(self) -> int:
        return 4

    @property
    def edge_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def max_episode_steps(self) -> int:
        if self._mode == 'train':
            return 500
        else:
            return 2500

    @property
    def default_params(self) -> dict:
        return {
            'm': 1.0,  # mass of the car
            'comm_radius': 1.0,  # communication radius
            'car_radius': 0.05,  # radius of the cars
            'dist2goal': 0.04,  # goal reaching threshold
            'speed_limit': 0.8,  # maximum speed
            'max_distance': 4.0,  # maximum moving distance to goal
            'area_size': 4.0
        }

    def dynamics(self, data: Data, u: Union[Tensor, Expression]) -> Union[Tensor, Expression]:
        x = data.states
        if isinstance(u, Expression):
            x = x.cpu().detach().numpy()
            A = np.zeros((self.state_dim, self.state_dim))
            A[0, 2] = 1.
            A[1, 3] = 1.
            B = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
            xdot = x @ A.T + u @ B.T
            return xdot
        else:
            return torch.cat([x[:, 2:], u], dim=1)

    def reset(self) -> Data:
        self._t = 0
        side_length = self._params['area_size']
        states = torch.zeros(self.num_agents, 2, device=self.device)
        goals = torch.zeros(self.num_agents, 2, device=self.device)

        if self._mode == 'train' or self._mode == 'test' or self._mode == 'demo_2':
            # randomly generate positions of agents
            i = 0
            while i < self.num_agents:
                candidate = torch.rand(2, device=self.device) * side_length
                dist_min = torch.norm(states - candidate, dim=1).min()
                if dist_min <= self._params['car_radius'] * 4:
                    continue
                states[i] = candidate
                i += 1

            # randomly generate goals of agents
            i = 0
            while i < self.num_agents:
                if self._mode == 'demo_2':
                    candidate = (torch.rand(2, device=self.device) * 2 - 1) * self._params['max_distance'] + states[i]
                    if (candidate > self._params['area_size']).any() or (candidate < 0).any():
                        continue
                else:
                    candidate = torch.rand(2, device=self.device) * side_length
                dist_min = torch.norm(goals - candidate, dim=1).min()
                if dist_min <= self._params['car_radius'] * 4:
                    continue
                goals[i] = candidate
                i += 1
        else:
            raise ValueError('Reset environment: unknown type of mode!')

        # add velocity
        states = torch.cat([states, torch.zeros(self.num_agents, 2, device=self.device)], dim=1)

        # record goals
        self._goal = goals

        # build graph
        data = Data(x=torch.zeros_like(states), pos=states[:, :2], states=states)
        data = self.add_communication_links(data)
        self._data = data

        # set parameters for plotting
        points = torch.cat([states[:, :2], goals], dim=0).cpu().detach().numpy()
        xy_min = np.min(points, axis=0) - self._params['car_radius'] * 5
        xy_max = np.max(points, axis=0) + self._params['car_radius'] * 5
        max_interval = (xy_max - xy_min).max()
        self._xy_min = xy_min - 0.5 * (max_interval - (xy_max - xy_min))
        self._xy_max = xy_max + 0.5 * (max_interval - (xy_max - xy_min))

        return data

    def step(self, action: Tensor) -> Tuple[Data, float, bool, dict]:
        self._t += 1

        # calculate next state using dynamics
        reward_action = -torch.norm(action, dim=1) * 0.0001
        action = action + self.u_ref(self._data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        prev_reach = torch.less(torch.norm(self.data.states[:, :2] - self._goal, dim=1), self._params['dist2goal'])
        with torch.no_grad():
            state = self.forward(self.data, action)

        # construct graph using the new states
        data = Data(x=torch.zeros_like(state), pos=state[:, :2], states=state)
        self._data = self.add_communication_links(data)

        # the episode ends when reaching max_episode_steps or all the agents reach the goal
        time_up = self._t >= self.max_episode_steps
        reach = torch.less(torch.norm(self.data.states[:, :2] - self._goal, dim=1), self._params['dist2goal'])
        done = time_up or reach.all()

        # reward function
        reward_step = -0.01
        reward_collision = -self.collision_mask(data).int() * 2
        reward_reach = (reach.int() - prev_reach.int()) * 4
        reward = reward_reach + reward_collision + reward_step + reward_action
        reward = reward.detach().cpu().numpy()

        safe = float(1.0 - self.collision_mask(data).sum() / self.num_agents)
        collision_agent = torch.where(self.collision_mask(data) > 0)[0]
        return self.data, reward, done, {'safe': safe, 'reach': reach, 'collision': collision_agent}

    def forward_graph(self, data: Data, action: Tensor) -> Data:
        # calculate next state using dynamics
        action = action + self.u_ref(data)
        lower_lim, upper_lim = self.action_lim
        action = torch.clamp(action, lower_lim, upper_lim)
        state = self.forward(data, action)

        # construct the graph of the next step, retaining the connection
        data_next = Data(
            x=torch.zeros_like(state),
            edge_index=data.edge_index,
            edge_attr=self.edge_attr(state, data.edge_index),
            pos=state[:, :2],
            states=state
        )

        return data_next

    def render(
            self, traj: Optional[Tuple[Data, ...]] = None, return_ax: bool = False, plot_edge: bool = True, ax=None
    ) -> Union[Tuple[np.array, ...], np.array]:
        return_tuple = True
        if traj is None:
            data = self.data
            traj = (data,)
            return_tuple = False

        r = self._params['car_radius']
        gif = []
        for data in traj:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=80)

            # plot the cars and the communication network
            plot_graph(ax, data, radius=r, color='#FF8C00', with_label=True, plot_edge=plot_edge, alpha=0.8)

            # plot the goals
            goal_data = Data(pos=self._goal[:, :2])
            plot_graph(ax, goal_data, radius=r, color='#3CB371',
                       with_label=True, plot_edge=False, alpha=0.8)

            # texts
            fontsize = 14
            collision_text = ax.text(0., 0.97, "", transform=ax.transAxes, fontsize=fontsize)
            unsafe_index = mask_to_index(self.collision_mask(data))
            collision_text.set_text(f'Collision: {unsafe_index.cpu().detach().numpy()}')

            # set axis limit
            x_interval = self._xy_max[0] - self._xy_min[0]
            y_interval = self._xy_max[1] - self._xy_min[1]
            ax.set_xlim(self._xy_min[0], self._xy_min[0] + max(x_interval, y_interval))
            ax.set_ylim(self._xy_min[1], self._xy_min[1] + max(x_interval, y_interval))
            plt.axis('off')

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
        data = self._builder(data)
        data.update(Data(edge_attr=self.edge_attr(data.states, data.edge_index)))
        return data

    @property
    def state_lim(self) -> Tuple[Tensor, Tensor]:
        low_lim = torch.tensor(
            [self._xy_min[0], self._xy_min[1], -self._params['speed_limit'], -self._params['speed_limit']],
            device=self.device)
        high_lim = torch.tensor(
            [self._xy_max[0], self._xy_max[1], self._params['speed_limit'], self._params['speed_limit']],
            device=self.device)
        return low_lim, high_lim

    @property
    def action_lim(self) -> Tuple[Tensor, Tensor]:
        upper_limit = torch.ones(2, device=self.device) * 10.
        lower_limit = - upper_limit
        return lower_limit, upper_limit

    def u_ref(self, data: Data) -> Tensor:
        goal = torch.cat([self._goal, torch.zeros_like(self._goal)], dim=1)
        states = data.states.reshape(-1, self.num_agents, self.state_dim)
        diff = (states - goal)

        if self._K is None:
            # calculate the LQR controller
            A = np.array([[0., 0., 1., 0.],
                          [0., 0., 0., 1.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.]]) * self.dt + np.eye(self.state_dim)
            B = np.array([[0., 0.],
                          [0., 0.],
                          [1., 0.],
                          [0., 1.]]) * self.dt
            Q = np.eye(self.state_dim)
            R = np.eye(self.action_dim)
            K_np = lqr(A, B, Q, R)
            self._K = torch.from_numpy(K_np).type_as(data.states)

        # feedback control
        action = - torch.einsum('us,bns->bnu', self._K, diff)
        action = action.reshape(-1, self.action_dim)

        # adapt to speed limit
        states = states.reshape(-1, self.state_dim)
        over_speed_agent = torch.where(states[:, 2:].norm(dim=1) - self._params['speed_limit'] > 0)[0]
        if over_speed_agent.shape[0] > 0:
            # add penalty to the action
            v_over_speed = states[over_speed_agent, 2:]
            v_dir = v_over_speed / v_over_speed.norm(dim=1, keepdim=True)
            action[over_speed_agent] -= \
                (v_over_speed.norm(dim=1, keepdim=True) - self._params['speed_limit']) * v_dir * 50

        return action

    def safe_mask(self, data: Data, return_edge: bool = False) -> Tensor:
        if return_edge:
            pos_diff = data.edge_attr[:, :2]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=-1)
            safe = torch.greater(dist, 4 * self._params['car_radius'])
            return safe

        if isinstance(data, Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]

        mask = []
        for graph in data_list:
            state_diff = graph.states.unsqueeze(1) - graph.states.unsqueeze(0)
            pos_diff = state_diff[:, :, :2]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=2)
            dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=self.device) * (
                    4 * self._params['car_radius'] + 1)
            safe = torch.greater(dist, 4 * self._params['car_radius'])
            mask.append(torch.min(safe, dim=1)[0])
        mask = torch.cat(mask, dim=0).bool()

        return mask

    def unsafe_mask(self, data: Data, return_edge: bool = False) -> Tensor:
        if return_edge:
            pos_diff = data.edge_attr[:, :2]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=-1)
            collision = torch.less(dist, 2 * self._params['car_radius'])
            return collision

        warn_dist = 4 * self._params['car_radius']
        if isinstance(data, Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]
        mask = []
        for graph in data_list:
            # collision
            state_diff = graph.states.unsqueeze(1) - graph.states.unsqueeze(0)
            pos_diff = state_diff[:, :, :2]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=2)
            dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=self.device) * (
                    4 * self._params['car_radius'] + 1)
            collision = torch.less(dist, 2 * self._params['car_radius'])
            graph_mask = torch.max(collision, dim=1)[0]

            # unsafe direction
            warn_zone = torch.less(dist, warn_dist)
            pos_vec = -(pos_diff / (torch.norm(pos_diff, dim=2, keepdim=True) + 0.0001))  # [i, j]: i -> j
            v = graph.states[:, 2:].norm(dim=1, keepdim=True) + 0.00001
            theta_vec = torch.cat(
                [graph.states[:, 2].unsqueeze(1) / v, graph.states[:, 3].unsqueeze(1) / v], dim=1
            ).repeat(pos_vec.shape[1], 1, 1).transpose(0, 1)  # [i, j]: theta[i]
            inner_prod = torch.sum(pos_vec * theta_vec, dim=2)
            unsafe_threshold = torch.cos(torch.asin(self._params['car_radius'] * 2 / (dist + 0.0000001)))
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
        mask = []
        for graph in data_list:
            state_diff = graph.states.unsqueeze(1) - graph.states.unsqueeze(0)
            pos_diff = state_diff[:, :, :2]  # [i, j]: j -> i
            dist = pos_diff.norm(dim=2)
            dist[:, :dist.shape[0]] += torch.eye(dist.shape[0], device=self.device) * (
                    2 * self._params['car_radius'] + 1)
            collision = torch.less(dist, 2 * self._params['car_radius'])
            mask.append(torch.max(collision, dim=1)[0])
        mask = torch.cat(mask, dim=0).bool()
        return mask
