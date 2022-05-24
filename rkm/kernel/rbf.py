"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
import logging

from .. import utils
from .implicit import implicit, base


@utils.extend_docstring(base)
class rbf(implicit):
    r"""
    RBF kernel (radial basis function).

    .. math::
        k(x,y) = \exp\left( -\frac{\lVert x-y \rVert_2^2}{2\texttt{sigma}^2} \right).

    :param sigma: Bandwidth :math:`\sigma` of the RBF kernel. If `None`, the value is filled by a heuristic on
        the sample dataset: 7/10th of the median of the pairwise distances. Computing the heuristic on the full sample
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
        super(rbf, self).__init__(**kwargs)

        # RBF kernels are always naturally normalized
        self._is_normalized = True
        self._normalize = False

        self._sigma_trainable = kwargs["sigma_trainable"]
        sigma = kwargs["sigma"]
        if sigma is None:
            self._compute_K()
            self._sigma = None
        else:
            self._sigma = torch.nn.Parameter(
                torch.tensor(kwargs["sigma"], dtype=utils.FTYPE), requires_grad=self._sigma_trainable)

    def __str__(self):
        return f"RBF kernel (sigma: {str(self.sigma.data.cpu().numpy())})"

    @property
    def normalize(self) -> bool:
        r"""
        Indicates if the kernel has to be normalized. Changing this value leads to a recomputation of the statistics.
        """
        return self._is_normalized

    @normalize.setter
    def normalize(self, val: bool):
        logging.info('Changing the normalization has not effect on the RBF kernel as it is always normalized by '
                     'definition')

    @property
    def sigma(self):
        r"""
        Bandwidth of the RBF kernel.
        """
        return self._sigma.data

    @sigma.setter
    def sigma(self, val):
        self._reset()
        self._sigma = val

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
        return {"Kernel": "RBF", "Trainable sigma": self.sigma_trainable, **super(rbf, self).hparams}

    def _implicit(self, x=None, y=None):
        x, y = super(rbf, self)._implicit(x, y)

        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y
        norm2 = torch.sum(diff * diff, dim=0, keepdim=True)

        if self._sigma is None:
            sigma = .7 * torch.median(norm2)
            self._sigma = torch.nn.Parameter(
                torch.tensor(sigma, dtype=utils.FTYPE), requires_grad=self._sigma_trainable)

        fact = 1 / (2 * torch.abs(self._sigma) ** 2)
        output = torch.exp(torch.mul(norm2, -fact))

        return output.squeeze(0)
