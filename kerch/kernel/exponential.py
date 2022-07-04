"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from abc import ABCMeta, abstractmethod
from .. import utils
from .implicit import implicit, base


@utils.extend_docstring(base)
class exponential(implicit, metaclass=ABCMeta):
    r"""
    :param sigma: Bandwidth :math:`\sigma` of the kernel. If `None`, the value is filled by a heuristic on
        the sample dataset: 3/10th of the median of the pairwise distances. Computing the heuristic on the full sample
        dataset can be expensive and `idx_sample` or `prop_sample` could be specified to only compute it on a subset
        only., defaults to `None`.
    :param sigma_trainable: `True` if the gradient of the bandwidth is to be computed. If so, a graph is computed
        and the bandwidth can be updated. `False` just leads to a static computation., defaults to `False`
    :type sigma: double, optional
    :type sigma_trainable: bool, optional
    """

    @utils.kwargs_decorator(
        {"sigma": None, "sigma_trainable": False})
    def __init__(self, **kwargs):
        self._sigma = kwargs["sigma"]
        super(exponential, self).__init__(**kwargs)

        # Exponential kernels are always naturally normalized if not centered
        if not self._center:
            self._normalize = False

        self._sigma_trainable = kwargs["sigma_trainable"]
        if self._sigma is not None:
            sigma = torch.tensor(self._sigma, dtype=utils.FTYPE)
            self._sigma = torch.nn.Parameter(sigma, requires_grad=self._sigma_trainable)

    def __str__(self):
        if self._sigma is None:
            return f"exponential kernel (sigma undefined)"
        return f"exponential kernel (sigma: {str(self.sigma)})"

    @property
    def normalize(self) -> bool:
        r"""
        Indicates if the kernel is normalized. This value cannot be changed for exponential kernels.
        """
        if self._center:
            return self._normalize
        else:
            return True

    @normalize.setter
    def normalize(self, val: bool):
        self._normalize_requested = val
        if self._center:
            self._normalize = self._normalize_requested
        else:
            self._log.info('Changing the normalization has not effect on the RBF kernel as it is always normalized by '
                           'definition if non-centered.')

    @property
    def center(self) -> bool:
        r"""
        Indicates if the kernel has to be centered. Changing this value leads to a recomputation of the statistics.
        """
        return self._center

    @center.setter
    def center(self, val: bool):
        self._center = val
        if self._center:
            self._normalize = self._normalize_requested
        else:
            self._normalize = False
        self._reset()

    @property
    def sigma(self):
        r"""
        Bandwidth :math:`\sigma` of the kernel.
        """
        if isinstance(self._sigma, torch.nn.Parameter):
            return self._sigma.data.cpu().numpy()
        elif self._sigma is None and not self._empty_sample:
            self._compute_K()
            return self.sigma
        return self._sigma

    @sigma.setter
    def sigma(self, val):
        self._reset()
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
        return {"Trainable sigma": self.sigma_trainable, **super(exponential, self).hparams}

    @abstractmethod
    def _dist(self, x, y):
        pass

    def _implicit(self, x=None, y=None):
        x, y = super(exponential, self)._implicit(x, y)

        D = self._dist(x, y)

        # define sigma if not set by the user
        if self._sigma is None:
            sigma = .3 * torch.median(D)
            self._sigma = torch.nn.Parameter(sigma, requires_grad=self._sigma_trainable)
            self._log.warning(f"Bandwidth sigma not provided and assigned by a heuristic (sigma={self.sigma}).")

        fact = 1 / (2 * torch.abs(self._sigma) ** 2)
        output = torch.exp(torch.mul(D, -fact))

        return output

    def _implicit_self(self, x=None):
        if x is None:
            x = self.current_sample

        return torch.ones(x.shape[0], dtype=utils.FTYPE, device=x.device)

    def _slow_parameters(self, recurse=True):
        if self._sigma is not None:
            yield self._sigma
        yield from super(exponential, self)._slow_parameters(recurse)
