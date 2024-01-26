# coding=utf-8
"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from .. import utils
from .exponential import Exponential


@utils.extend_docstring(Exponential)
class RBF(Exponential):
    r"""
    RBF kernel (radial basis function) of bandwidth :math:`\sigma>0`.

    .. math::
        k(x,y) = \exp\left( -\frac{\lVert x-y \rVert_2^2}{2\sigma^2} \right).


    .. note::
        If working with big datasets, one may consider an explicit approximation of the RBF kernel using
        Random Fourier Features (:class:`kerch.kernel.RFF`). This will be faster provided :math:`2\times\texttt{num_weights} < n`,
        where :math:`\texttt{num_weights}` is the number of weights used to control the RFF approximation and :math:`n` is
        the number of datapoints. The latter class however does not offer so much flexibility, as the automatic determination
        of the bandwidth :math:`\sigma` using a heuristic for example.

        Other considerations may come into play. If a centered or normalized kernel on an out-of-sample is required, this may require extra
        computations when directly using the kernel matrix as doing it on the explicit feature is more straightforward.
    """

    def __init__(self, *args, **kwargs):
        super(RBF, self).__init__(*args, **kwargs)

    def __str__(self):
        if self._sigma_defined:
            return f"RBF kernel (sigma: {str(self.sigma)})"
        return f"RBF kernel (sigma undefined)"


    @property
    def hparams_fixed(self):
        return {"Kernel": "RBF", **super(RBF, self).hparams_fixed}

    def _dist(self, x, y):
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y
        return torch.sum(diff * diff, dim=0, keepdim=False)


