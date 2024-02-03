from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Iterator
import torch

from ... import utils
from ..implicit import Implicit
from ...feature import Sample


@utils.extend_docstring(Implicit)
class _Distance(Implicit, metaclass=ABCMeta):
    r"""
    :param sigma: Bandwidth :math:`\sigma` of the kernel. If `None`, the value is filled by a heuristic on
        the sample data: half of the square root of the median of the pairwise distances. Computing the heuristic on
        the full sample data can be expensive and `idx_sample` or `prop_sample` could be specified to only compute
        it on a subset only., defaults to `None`.
    :param sigma_trainable: `True` if the gradient of the bandwidth is to be computed. If so, a graph is computed
        and the bandwidth can be updated. `False` just leads to a static computation., defaults to `False`
    :type sigma: float, optional
    :type sigma_trainable: bool, optional
    """

    _cache_elements = ["_kernel_dist_sample", "_kernel_square_dist_sample"]

    def __init__(self, *args, **kwargs):
        self._sigma_defined = False
        super(_Distance, self).__init__(*args, **kwargs)

        sigma = kwargs.pop('sigma', None)
        self._sigma_trainable = kwargs.pop('sigma_trainable', False)
        self._sigma_defined = sigma is not None
        self._sigma = torch.nn.Parameter(torch.ones(1, dtype=utils.FTYPE), requires_grad=self._sigma_trainable)
        if self._sigma_defined:
            self.sigma = sigma

        self._default_square = False

    def __str__(self):
        return "generic distance"

    @property
    def sigma(self):
        r"""
        Bandwidth :math:`\sigma` of the kernel.
        """
        if self._sigma_defined:
            return self._sigma.data.detach().cpu().item()
        elif not self._sigma_defined and not self.empty_sample:
            self._determine_sigma()
            return self.sigma
        raise utils.NotInitializedError(cls=self, message='The kernel bandwidth sigma value is unset. It cannot be '
                                                          'deduced based on a heuristic as the sample is also unset.')

    @sigma.setter
    def sigma(self, val):
        self._logger.debug(f"Setting sigma to {val}.")
        self._reset_cache(reset_persisting=False, avoid_classes=[Sample, _Distance])
        self._sigma_defined = True
        self._sigma.data = utils.castf(val, tensor=False, dev=self._sigma.device)

    @property
    def sigma_trainable(self) -> bool:
        r"""
        Boolean indicating of the bandwidth :math:`\sigma` is trainable.
        """
        return self._sigma_trainable

    @sigma_trainable.setter
    def sigma_trainable(self, val: bool):
        self._sigma_trainable = val
        self._sigma.requires_grad = self._sigma_trainable

    @property
    def hparams_variable(self):
        return {'Kernel parameter sigma': self.sigma,
                **super(_Distance, self).hparams_variable}

    @property
    def hparams_fixed(self):
        return {"Trainable sigma": self.sigma_trainable,
                **super(_Distance, self).hparams_fixed}

    @abstractmethod
    def _determine_sigma(self) -> None:
        pass

    @abstractmethod
    def _dist_sigma(self, x, y):
        pass

    @abstractmethod
    def _square_dist_sigma(self, x, y):
        pass

    def _slow_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        yield from super(_Distance, self)._slow_parameters(recurse)
        yield self._sigma

    @abstractmethod
    def _square_dist(self, x, y) -> torch.Tensor:
        pass

    @abstractmethod
    def _dist(self, x, y) -> torch.Tensor:
        pass

    @property
    def _sigma_fact(self) -> float | None:
        return 1 / self._sigma
