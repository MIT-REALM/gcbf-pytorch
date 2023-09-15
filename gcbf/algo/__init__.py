import torch

from typing import Optional

from .base import Algorithm
from .gcbf import GCBF
from .nominal import Nominal
from .macbf import MACBF
from ..env import MultiAgentEnv


def make_algo(
        algo: str,
        env: MultiAgentEnv,
        num_agents: int,
        node_dim: int,
        edge_dim: int,
        action_dim: int,
        device: torch.device,
        batch_size: int = 128,
        hyperparams: Optional[dict] = None
) -> Algorithm:
    if algo == 'nominal':
        return Nominal(
            env, num_agents, node_dim, edge_dim, action_dim, device
        )
    if algo == 'gcbf':
        return GCBF(
            env, num_agents, node_dim, edge_dim, action_dim, device, batch_size, hyperparams
        )
    elif algo == 'macbf':
        return MACBF(
            env, num_agents, node_dim, edge_dim, action_dim, device, batch_size, hyperparams
        )
    else:
        raise NotImplementedError('Unknown Algorithm!')
