"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from typing import Iterator
import torch

from abc import ABCMeta, abstractmethod
from .. import utils
from ._Implicit import _Implicit, _Projected


@utils.extend_docstring(_Implicit)
class _Exponential(_Implicit, metaclass=ABCMeta):
    r"""
    :param sigma: Bandwidth :math:`\sigma` of the kernel. If `None`, the value is filled by a heuristic on
        the sample data: half of the square root of the median of the pairwise distances. Computing the heuristic on
        the full sample data can be expensive and `idx_sample` or `prop_sample` could be specified to only compute
        it on a subset only., defaults to `None`.
    :param sigma_trainable: `True` if the gradient of the bandwidth is to be computed. If so, a graph is computed
        and the bandwidth can be updated. `False` just leads to a static computation., defaults to `False`
    :type sigma: double, optional
    :type sigma_trainable: bool, optional
    """

    @utils.kwargs_decorator(
        {"sigma": None, "sigma_trainable": False})
    def __init__(self, **kwargs):
        self._sigma = kwargs["sigma"]
        # _Exponential kernels are always naturally normalized

        super(_Exponential, self).__init__(**kwargs)

        self._sigma_trainable = kwargs["sigma_trainable"]
        if self._sigma is not None:
            sigma = torch.tensor(self._sigma, dtype=utils.FTYPE)
            self._sigma = torch.nn.Parameter(sigma, requires_grad=self._sigma_trainable)

    def __str__(self):
        if self._sigma is None:
            return f"exponential kernel (sigma undefined)"
        return f"exponential kernel (sigma: {str(self.sigma)})"

    @property
    def _naturally_normalized(self) -> bool:
        return True

    @property
    def sigma(self):
        r"""
        Bandwidth :math:`\sigma` of the kernel.
        """
        if isinstance(self._sigma, torch.nn.Parameter):
            return self._sigma.data.cpu().numpy()
        elif self._sigma is None and not self.empty_sample:
            self.k(explicit=True, projections=[])
            return self.sigma
        return self._sigma

    @sigma.setter
    def sigma(self, val):
        self._reset_cache()
        self._sigma.data = utils.castf(val, tensor=False, dev=self._sigma.device)

    @property
    def sigma_trainable(self) -> bool:
        r"""
        Boolean indicating of the bandwidth is trainable.
        """
        return self._sigma_trainable

    @sigma_trainable.setter
    def sigma_trainable(self, val: bool):
        self._sigma_trainable = val
        self._sigma.requires_grad = self._sigma_trainable

    @property
    def params(self):
        return {'Sigma': self.sigma}

    @property
    def hparams(self):
        return {"Trainable sigma": self.sigma_trainable, **super(_Exponential, self).hparams}

    @abstractmethod
    def _dist(self, x, y):
        pass

    def _implicit(self, x, y):
        D = self._dist(x, y)

        # define sigma if not set by the user
        if self._sigma is None:
            sigma = .5 * torch.sqrt(torch.median(D))
            self._sigma = torch.nn.Parameter(sigma, requires_grad=self._sigma_trainable)
            self._log.warning(f"Bandwidth sigma not provided and assigned by a heuristic (sigma={self.sigma}).")

        fact = 1 / (2 * torch.abs(self._sigma) ** 2)
        output = torch.exp(torch.mul(D, -fact))

        return output

    def _implicit_self(self, x=None):
        if x is None:
            x = self.current_sample_projected

        return torch.ones(x.shape[0], dtype=utils.FTYPE, device=x.device)

    def _slow_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        yield from super(_Exponential, self)._slow_parameters(recurse)
        if self._sigma is not None:
            yield self._sigma
