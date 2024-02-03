from __future__ import annotations

from abc import ABCMeta
import torch
from .distance import Distance
from ...utils import extend_docstring


@extend_docstring(Distance)
class Minkowski(Distance, metaclass=ABCMeta):
    r"""
    :param minkowski_order: the order :math:`p` of the Minkowski distance.
    :type minkowski_order: float
    """

    def __init__(self, *args, **kwargs):
        super(Minkowski, self).__init__(*args, **kwargs)
        order = kwargs.pop('minkowski_order', None)
        if order is None:
            raise ValueError('Please provide an order for the Minkowski distance through the argument minkowski_order.')
        self._minkowski_order = order
    def __str__(self):
        return "manhattan"

    @property
    def minkowski_order(self) -> float:
        r"""
        Order :math:`p` of the Minkowski distance.
        """
        return self._minkowski_order

    def _dist(self, x, y) -> torch.Tensor:
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y
        return torch.sum(torch.abs(diff).pow(self.minkowski_order), dim=0, keepdim=False).pow(1 / self.minksowski_order)
