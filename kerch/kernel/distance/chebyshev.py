from __future__ import annotations

from abc import ABCMeta
import torch
from .distance import Distance
from ...utils import extend_docstring


@extend_docstring(Distance)
class Chebyshev(Distance, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super(Chebyshev, self).__init__(*args, **kwargs)

    def __str__(self):
        return "chebyshev"

    def _dist(self, x, y) -> torch.Tensor:
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y
        return torch.max(torch.abs(diff), dim=0, keepdim=False).values
