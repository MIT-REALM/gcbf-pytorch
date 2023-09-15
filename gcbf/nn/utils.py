import torch.nn as nn


def init_param(module: nn.Module, gain: float = 1.):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module
