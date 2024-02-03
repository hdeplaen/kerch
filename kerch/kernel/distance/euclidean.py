from __future__ import annotations

from abc import ABCMeta
import torch
from .distance_squared import DistanceSquared
from ...utils import extend_docstring


@extend_docstring(DistanceSquared)
class Euclidean(DistanceSquared, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super(Euclidean, self).__init__(*args, **kwargs)

    def __str__(self):
        return "euclidean"

    def _square_dist(self, x, y) -> torch.Tensor:
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y
        return torch.sum(diff * diff, dim=0, keepdim=False)
