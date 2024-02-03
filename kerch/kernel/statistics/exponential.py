# coding=utf-8
"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from kerch import utils
from kerch.kernel.distance.select_distance import SelectDistance
from math import sqrt


@utils.extend_docstring(SelectDistance)
class Exponential(SelectDistance):
    r"""
    Generic exponential kernels.

    .. math::
        k(x,y) = \exp\left(-\frac{d(x,y)^2}{2\sigma^2}\right),

    for the argument ``squared=True`` (default) and

    .. math::
        k(x,y) = \exp\left(-\frac{d(x,y)}{\sqrt{2}\sigma}\right),

    for the argument ``squared=False``.

    .. note::
        The ``squared``In the case of ``distance='euclidean'`` (default), these shapes correspond to the :py:class:`kerch.kernel.RBF` for
        the ``squared=True`` (default) and :py:class:`kerch.kernel.Laplacian` for ``squared=False``. In other words is
        :py:class:`kerch.kernel.Exponential` equivalent to :py:class:`kerch.kernel.RBF` with the default set of parameters.


    :param squared: Boolean indicating whether the norm in the exponential is squared. Defaults to ``True``.
    :type squared: bool

    """

    def __init__(self, *args, **kwargs):
        super(Exponential, self).__init__(*args, **kwargs)
        self._squared = kwargs.pop('squared', True)

    def __str__(self):
        return f"exponential kernel {SelectDistance.__str__(self)}"

    @property
    def hparams_fixed(self) -> dict:
        return {"Kernel": "Exponential",
                "Squared exp. distance": self._squared,
                **super(Exponential, self).hparams_fixed}

    @property
    def squared(self) -> bool:
        r"""
        Boolean indicating whether the norm in the exponential is squared.
        """
        return self._squared

    @property
    def _naturally_normalized(self) -> bool:
        # _Exponential kernels are always naturally normalized
        return True

    def _implicit(self, x, y):
        if self._squared:
            fact = -.5
            d = self._square_dist_sigma(x, y)
        else:
            fact = -sqrt(.5)
            d = self._dist_sigma(x, y)
        return torch.exp(torch.mul(d, fact))

    def _implicit_self(self, x=None):
        if x is None:
            x = self.current_sample_projected
        return torch.ones(x.shape[0], dtype=utils.FTYPE, device=x.device)
