import torch

from typing import Optional

from .base import MultiAgentEnv
from .simple_car import SimpleCar
from .simple_drone import SimpleDrone
from .dubins_car import DubinsCar


def make_env(
        env: str,
        num_agents: int,
        device: torch.device,
        dt: float = 0.03,
        params: Optional[dict] = None,
        max_neighbors: Optional[int] = None
):
    if env == 'SimpleCar':
        return SimpleCar(num_agents, device, dt, params, max_neighbors)
    elif env == 'SimpleDrone':
        return SimpleDrone(num_agents, device, dt, params, max_neighbors)
    elif env == 'DubinsCar':
        return DubinsCar(num_agents, device, dt, params, max_neighbors)
    else:
        raise NotImplementedError('Env name not supported!')
