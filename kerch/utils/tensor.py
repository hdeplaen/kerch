import torch
from torch import Tensor as T


def eye_like(m: T) -> T:
    return torch.eye(*m.size(), out=torch.empty_like(m))


def ones_like(m: T) -> T:
    return torch.ones(*m.size(), out=torch.empty_like(m))
