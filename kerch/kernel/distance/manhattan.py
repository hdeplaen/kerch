from __future__ import annotations

from abc import ABCMeta
import torch
from .distance import Distance
from ...utils import extend_docstring


@extend_docstring(Distance)
class Manhattan(Distance, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super(Manhattan, self).__init__(*args, **kwargs)

    def __str__(self):
        return "manhattan"

    def _dist(self, x, y) -> torch.Tensor:
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y
        return torch.sum(torch.abs(diff), dim=0, keepdim=False)
