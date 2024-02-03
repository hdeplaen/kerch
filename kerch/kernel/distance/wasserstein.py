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

    def _compute_cost(self, type_cost: str | None = None, size_input: tuple | None = None):
        if type_cost is None:
            raise ValueError(
                "No cost has been defined an it cannot be computed as the argument type_cost has not been provided.")
        if size_input is None:
            raise ValueError(
                "No cost has been defined an it cannot be computed as the argument size_input has not been provided.")


    def _dist(self, x, y) -> torch.Tensor:
        pass
