import copy

import numpy as np
import random
import torch

from typing import Tuple, List
from torch_geometric.data import Data


class Buffer:
    def __init__(self):
        self._data = []  # list with all graphs
        self.safe_data = []  # list of positions with safe graphs
        self.unsafe_data = []  # list of positions with unsafe graphs
        self.MAX_SIZE = 100000

    def append(self, data: Data, is_safe: bool):
        self._data.append(data)
        self.safe_data.append(self.size - 1) if is_safe else self.unsafe_data.append(self.size - 1)
        if self.size > self.MAX_SIZE:
            del self._data[0]  # remove oldest data
            try:
                self.safe_data.remove(0)
            except ValueError:
                self.unsafe_data.remove(0)
            self.safe_data = [i - 1 for i in self.safe_data]
            self.unsafe_data = [i - 1 for i in self.unsafe_data]

    @property
    def data(self) -> List[Data]:
        return self._data

    @property
    def size(self) -> int:
        return len(self._data)

    def merge(self, other):
        size_init = self.size
        self._data += other.data
        other_safe_data = [i + size_init for i in other.safe_data]
        self.safe_data.extend(other_safe_data)
        other_unsafe_data = [i + size_init for i in other.unsafe_data]
        self.unsafe_data.extend(other_unsafe_data)
        if self.size > self.MAX_SIZE:
            for i in range(self.size - self.MAX_SIZE):
                try:
                    self.safe_data.remove(i)
                except ValueError:
                    self.unsafe_data.remove(i)

            self.safe_data = [i - (self.size - self.MAX_SIZE) for i in self.safe_data]
            self.unsafe_data = [i - (self.size - self.MAX_SIZE) for i in self.unsafe_data]
            del self._data[:self.size - self.MAX_SIZE]  # remove oldest data

    def clear(self):
        self._data.clear()
        self.safe_data = []
        self.unsafe_data = []

    def sample(self, n: int, m: int = 1, balanced_sampling: bool = False) -> List[Data]:
        """
        Sample at random segments of trajectory from buffer.
        Each segment is selected as a symmetric ball w.r.t. randomly sampled data points
        (apart from data points at beginning or end)

        Parameters
        ----------
        n: int,
            number of sample segments
        m: int,
            maximal length of each sampled trajectory segment
        balanced_sampling: bool,
            balance the samples from safe states and unsafe states
        """
        assert self.size >= max(n, m)
        data_list = []
        if not balanced_sampling:
            index = np.sort(np.random.randint(0, self.size, n))

        else:
            index_unsafe, index_safe = [], []
            if len(self.unsafe_data) > 0:
                index_unsafe = random.choices(self.unsafe_data, k=n // 2)
            if len(self.safe_data) > 0:
                index_safe = random.choices(self.safe_data, k=n // 2)
            index = sorted(index_safe + index_unsafe)

        ub = 0
        for i in index:
            lb = max(i - m // 2, ub)  # max with ub avoids replicas of the same graph in data_list
            ub = min(i + m // 2 + 1, self.size)
            data_list.extend(self._data[lb:ub])

        return data_list


class RolloutBuffer:
    """
    Rollout buffer that often used in training RL agents.
    """

    def __init__(
            self,
            num_agents: int,
            buffer_size: int,
            action_dim: int,
            device: torch.device,
    ):
        self._n = 0
        self._p = 0
        self._data = []
        self.device = device
        self.buffer_size = buffer_size
        self.num_agents = num_agents

        self.data = [Data() for _ in range(self.buffer_size)]
        self.actions = torch.empty(
            (self.buffer_size, num_agents, action_dim), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.buffer_size, num_agents), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.buffer_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.buffer_size, num_agents), dtype=torch.float, device=device)
        self.next_data = [Data() for _ in range(self.buffer_size)]

    def append(
            self,
            data: Data,
            action: torch.Tensor,
            reward: float,
            done: bool,
            log_pi: float,
            next_data: Data
    ):
        """
        Save a transition in the buffer.
        """
        if action.ndim == 2:
            action = action.squeeze(0)

        self.data[self._p] = copy.deepcopy(data)
        self.actions[self._p].copy_(action)
        self.rewards[self._p].copy_(torch.from_numpy(reward))
        self.dones[self._p] = float(done)
        self.log_pis[self._p].copy_(log_pi)
        self.next_data[self._p] = copy.deepcopy(next_data)

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def get(self) -> Tuple[List[Data], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Data]]:
        """
        Get all data in the buffer.
        """
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.buffer_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.data[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_data[idxes]
        )

    def sample(
            self,
            batch_size: int
    ) -> Tuple[List[Data], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Data]]:
        """
        Sample data from the buffer.

        Parameters
        ----------
        batch_size: int
            batch size

        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        log_pis: torch.Tensor
        next_states: torch.Tensor
        """
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return_data = []
        return_next_data = []
        for i in idxes:
            return_data.append(self.data[i])
            return_next_data.append(self.next_data[i])
        return (
            return_data,
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            return_next_data
        )
