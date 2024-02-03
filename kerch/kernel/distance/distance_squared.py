from __future__ import annotations
from abc import ABCMeta, abstractmethod
import torch

from ... import utils
from .sigma import Sigma


@utils.extend_docstring(Sigma)
class DistanceSquared(Sigma, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super(DistanceSquared, self).__init__(*args, **kwargs)

    def _sample_square_dist(self, destroy=False) -> torch.Tensor:
        return self._get(key="_kernel_square_dist_sample", default_level='total', force=True, destroy=destroy,
                         fun=lambda: self._square_dist(self.current_sample_projected, self.current_sample_projected))

    def _determine_sigma(self) -> None:
        if not self._sigma_defined:
            with torch.no_grad():
                d = self._sample_square_dist()
                sigma = .5 * torch.sqrt(torch.median(d))
                self.sigma = sigma
                self._logger.warning(f"The kernel bandwidth sigma has not been provided and is assigned by a "
                                     f"heuristic (sigma={self.sigma:.2e}).")

    def _dist_sigma(self, x, y):
        return torch.sqrt(self._square_dist_sigma(x, y))

    def _square_dist_sigma(self, x, y):
        _ = self.sigma
        if self._sigma_fact is None:
            fact = 1 / self._sigma
            return self._square_dist(fact * x, fact * y)
        else:
            if id(x) == id(y) and id(x) == id(self.current_sample_projected):
                d = self._sample_square_dist(destroy=True)
            else:
                d = self._square_dist(x, y)
            return self._sigma_fact ** 2 * d

    @abstractmethod
    def _square_dist(self, x, y) -> torch.Tensor:
        pass

    def _dist(self, x, y) -> torch.Tensor:
        return torch.sqrt(self._square_dist(x, y))

    @property
    def _sigma_fact(self) -> float | None:
        return None
