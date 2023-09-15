import numpy as np
import torch

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, Optional, Union
from torch import Tensor
from torch_geometric.data import Data
from cvxpy import Variable, Expression


class MultiAgentEnv(ABC):

    def __init__(
            self,
            num_agents: int,
            device: torch.device,
            dt: float = 0.03,
            params: dict = None,
            max_neighbors: Optional[int] = None
    ):
        super(MultiAgentEnv, self).__init__()
        self._num_agents = num_agents
        self._device = device
        self._dt = dt
        if params is None:
            params = self.default_params
        self._params = params
        self._max_neighbors = max_neighbors
        self._data = None
        self._t = 0
        self._mode = 'train'

    def train(self):
        self._mode = 'train'

    def test(self):
        self._mode = 'test'

    def demo(self, idx: int):
        self._mode = f'demo_{idx}'

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def dt(self) -> float:
        """
        Returns
        -------
        dt: float,
            simulation time interval
        """
        return self._dt

    @property
    def device(self) -> torch.device:
        """
        Returns
        -------
        device: torch.device
            device of all the Tensors
        """
        return self._device

    @property
    def data(self) -> Data:
        """
        Returns
        -------
        data: Data,
            data of the current graph
        """
        return self._data

    @property
    def state(self) -> Tensor:
        """
        Returns
        -------
        state: Tensor (n, state_dim),
            the agents' states
        """
        return self._data.states

    @abstractproperty
    def state_dim(self) -> int:
        """
        Returns
        -------
        state_dim: int,
            dimension of the state
        """
        pass

    @abstractproperty
    def node_dim(self) -> int:
        """
        Returns
        -------
        node_dim: int,
            dimension of the node information
        """
        pass

    @abstractproperty
    def edge_dim(self) -> int:
        """
        Returns
        -------
        edge_dim: int,
            dimension of the edge information
        """
        pass

    @abstractproperty
    def action_dim(self) -> int:
        """
        Returns
        -------
        action_dim: int,
            dimension of the control action
        """
        pass

    @abstractproperty
    def max_episode_steps(self) -> int:
        """
        Get maximum simulation time steps.
        The simulation will be ended if the time step exceeds the maximum time steps.

        Returns
        -------
        max_episode_steps: int,
            maximum simulation time steps
        """
        pass

    @abstractproperty
    def default_params(self) -> dict:
        """
        Get default parameters.

        Returns
        -------
        params: dict,
            a dict of default parameters
        """
        pass

    @abstractmethod
    def dynamics(self, data: Data, u: Union[Tensor, Expression]) -> Union[Tensor, Expression]:
        """
        Dynamics of a single agent.

        Parameters
        ----------
        data: Data,
            current graph data
        u: Union[Tensor, Expression] (bs, action_dim) or (action_dim,),
            control input

        Returns
        -------
        xdot: Union[Tensor, Expression] (bs, state_dim),
            time derivative of the state
        """
        pass

    @abstractmethod
    def reset(self) -> Data:
        """
        Reset the environment and return the current graph.

        Returns
        -------
        data: Data,
            data of the current graph
        """
        pass

    @abstractmethod
    def step(self, action: Tensor) -> Tuple[Data, float, bool, dict]:
        """
        Simulation the system for one step.

        Parameters
        ----------
        action: Tensor (n, action_dim),
            action of all the agents

        Returns
        -------
        next_data: Data,
            graph data of the next time step
        reward: float,
            reward signal
        done: bool,
            if the simulation is ended or not
        info: dict,
            other useful information, including safe or unsafe
        """
        pass

    @abstractmethod
    def forward_graph(self, data: Data, action: Tensor) -> Data:
        """
        Get the graph of the next timestep after doing the action.
        The connection of the graph will be retained.

        Parameters
        ----------
        data: Data,
            batched graph data using Batch.from_datalist
        action: Tensor (bs x n, action_dim),
            action of all the agents in the batch

        Returns
        -------
        next_data: Data,
            batched graph data of the next time step
        """
        pass

    @abstractmethod
    def render(
            self, traj: Optional[Tuple[Data, ...]] = None, return_ax: bool = False, plot_edge: bool = True, ax=None
    ) -> Union[Tuple[np.array, ...], np.array]:
        """
        Plot the environment for the current time step.
        If traj is not None, plot the environment for the trajectory.

        Parameters
        ----------
        traj: Optional[Tuple[Data, ...]],
            a tuple of Tensor containing the graph of the trajectories.
        return_ax: bool,
            if return the axis of the figure
        plot_edge: bool,
            if plot the edge of the graph
        ax: matplotlib.axes.Axes,
            axis of the figure

        Returns
        -------
        fig: numpy array,
            if traj is None: an array of the figure of the current environment
            if traj is not None: an array of the figures of the trajectory
        """
        pass

    @ abstractmethod
    def edge_attr(self, state: Tensor, edge_index: Tensor) -> Tensor:
        """
        Get the edge attributes.

        Parameters
        ----------
        state: Tensor (n, state_dim),
            state of all the agents
        edge_index: Tensor (2, num_edges),
            edge index of the graph

        Returns
        -------
        edge_attr: Tensor (num_edges, edge_dim),
            edge attributes
        """
        pass

    @abstractmethod
    def add_communication_links(self, data: Data) -> Data:
        """
        Add communication links to the graph.

        Parameters
        ----------
        data: Data,
            graph data

        Returns
        -------
        data: Data,
            graph with added edge_index and edge_attrs
        """
        pass

    @abstractproperty
    def state_lim(self) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        lower limit, upper limit: Tuple[Tensor, Tensor],
            limits of the state
        """
        pass

    @abstractproperty
    def action_lim(self) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[Tensor, Tensor],
            limits of the action
        """
        pass

    @abstractmethod
    def u_ref(self, data: Data) -> Tensor:
        """
        Get reference control to finish the task without considering safety.

        Parameters
        ----------
        data: Data,
            current graph

        Returns
        -------
        u_ref: Tensor (bs x n, action_dim)
            reference control signal
        """
        pass

    @abstractmethod
    def safe_mask(self, data: Data, return_edge: bool = False) -> Tensor:
        """
        Mask out the safe agents. Masks are applied to each agent indicating safe (1) or dangerous (0)

        Parameters
        ----------
        data: Data,
            current network
        return_edge: bool,
            if return the mask on edges

        Returns
        -------
        mask: Tensor (bs x n, mask),
            the agent is safe (1) or unsafe (0)
        """
        pass

    @abstractmethod
    def unsafe_mask(self, data: Data, return_edge: bool = False) -> Tensor:
        """
        Mask out the unsafe agents. Masks are applied to each agent indicating safe (0) or dangerous (1)

        Parameters
        ----------
        data: Data,
            current network
        return_edge: bool,
            if return the mask on edges

        Returns
        -------
        mask: Tensor (bs x n, mask),
            the agent is safe (0) or unsafe (1)
        """
        pass

    @abstractmethod
    def collision_mask(self, data: Data) -> Tensor:
        """
        Mask out the agents that in collision. Masks are applied to each agent indicating safe (0) or dangerous (1).
        Used in testing

        Parameters
        ----------
        data: Data,
            current network

        Returns
        -------
        mask: Tensor (bs x n, mask),
            the agent is in collision (1) or not (0)
        """
        pass

    def forward(self, data: Data, u: Tensor) -> Tensor:
        """
        Simulate the single agent for one time step.

        Parameters
        ----------
        data: Data,
            current graph
        u: Tensor (bs x n, action_dim),
            control input

        Returns
        -------
        x_next: Tensor (bs x n, state_dim),
            next state of the agent
        """
        xdot = self.dynamics(data, u)
        return data.states + xdot * self.dt
