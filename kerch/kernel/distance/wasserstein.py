from __future__ import annotations

from abc import ABCMeta
import torch
from .distance import Distance
from ...utils import extend_docstring


@extend_docstring(Distance)
class Wasserstein(Distance, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super(Wasserstein, self).__init__(*args, **kwargs)
        raise NotImplementedError

    def __str__(self):
        return "wasserstein"

    def _dist(self, x, y) -> torch.Tensor:
        pass
