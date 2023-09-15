import torch
import math
import torch.nn as nn

from typing import Tuple


def init_param(module: nn.Module, gain: float = 1.):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


def calculate_log_pi(log_stds: torch.Tensor, noises: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Calculate log(\pi(a|s)) given log(std) of the distribution, noises, and actions to take
    Parameters
    ----------
    log_stds: torch.Tensor
        log(std) of the distribution
    noises: torch.Tensor
        noises added to the action
    actions: torch.Tensor
        actions to take
    Returns
    -------
    log_pi: torch.Tensor
        log(\pi(a|s))
    """
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    return gaussian_log_probs - torch.log(
        1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    # gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
    #     dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.shape[-1]
    #
    # return gaussian_log_probs - torch.log(1 - torch.tanh(actions).pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(means: torch.Tensor, log_stds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get action and its log_pi according to mean and log_std
    Parameters
    ----------
    means: torch.Tensor
        mean value of the action
    log_stds: torch.Tensor
        log(std) of the action
    Returns
    -------
    actions: torch.Tensor
        actions to take
    log_pi: torch.Tensor
        log_pi of the actions
    """
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)
    # noises = torch.randn_like(means)
    # us = means + noises * log_stds.exp()
    # actions = torch.tanh(us)
    # return us, calculate_log_pi(log_stds, noises, actions)


def atanh(x: torch.Tensor) -> torch.Tensor:
    """
    Return atanh of the input. Modified torch.atanh in case the output is nan.
    Parameters
    ----------
    x: torch.Tensor
        input
    Returns
    -------
    y: torch.Tensor
        atanh(x)
    """
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_log_pi(means: torch.Tensor, log_stds: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the log(\pi(a|s)) of the given action
    Parameters
    ----------
    means: torch.Tensor
        mean value of the action distribution
    log_stds: torch.Tensor
        log(std) of the action distribution
    actions: torch.Tensor
        actions taken
    Returns
    -------
    log_pi: : torch.Tensor
        log(\pi(a|s))
    """
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)
    # noises = (actions - means) / (log_stds.exp() + 1e-8)
    # return calculate_log_pi(log_stds, noises, torch.tanh(actions))
