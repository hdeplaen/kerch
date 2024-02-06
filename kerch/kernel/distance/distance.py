from __future__ import annotations
from abc import ABCMeta, abstractmethod
import torch

from ... import utils
from ._distance import _Distance


@utils.extend_docstring(_Distance)
class Distance(_Distance, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super(Distance, self).__init__(*args, **kwargs)

    def _sample_dist(self, destroy=False) -> torch.Tensor:
        return self._get(key="_kernel_dist_sample", default_level='total', force=True, destroy=destroy,
                         fun=lambda: self._dist(self.current_sample_projected, self.current_sample_projected))

    def _determine_sigma(self) -> None:
        if not self._sigma_defined:
            with torch.no_grad():
                d = self._sample_dist()
                sigma = .5 * torch.median(d)
                self.sigma = sigma
                self._logger.warning(f"The kernel bandwidth sigma has not been provided and is assigned by a "
                                     f"heuristic (sigma={self.sigma:.2e}).")

    def _dist_sigma(self, x, y):
        _ = self.sigma
        if id(x) == id(y) and id(x) == id(self.current_sample_projected):
            d = self._sample_dist(destroy=True)
        else:
            d = self._dist(x, y)
        return self._sigma_fact * d

    def _square_dist_sigma(self, x, y):
        return self._dist_sigma(x, y) ** 2

    def _square_dist(self, x, y) -> torch.Tensor:
        return self._dist(x, y) ** 2

    @abstractmethod
    def _dist(self, x, y) -> torch.Tensor:
        pass
