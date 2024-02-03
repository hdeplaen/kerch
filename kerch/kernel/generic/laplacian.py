# coding=utf-8
"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from ...feature.logger import _GLOBAL_LOGGER
from kerch.kernel.statistics.exponential import Exponential
from ...utils import extend_docstring
from ..distance.euclidean import Euclidean


@extend_docstring(Euclidean)
class Laplacian(Exponential):
    r"""
    Laplacian kernel.

    .. math::
        k(x,y) = \exp\left( -\frac{\lVert x-y \rVert_2}{\sqrt{2}\sigma} \right).


    .. note::
        The norm inside the exponential is never squared. If you wish a squared norm, this corresponds to the
        :py:class:`~kerch.kernel.RBF` kernel. If another distance than the Euclidean one is required, we refer to
        the more generic :py:class:`~kerch.kernel.Exponential` kernel.

    """

    def __new__(cls, *args, **kwargs):
        distance = kwargs.pop('distance', None)
        if distance is not None and distance != 'euclidean':
            _GLOBAL_LOGGER._logger.warning('A specific distance has been requested for the Laplacian kernel. The '
                                           'Laplacian kernel'
                                           'is defined as a particular exponential kernel with euclidean distance '
                                           'only. This value will be neglected. Please use the more generic '
                                           'Exponential kernel if you wish to use another distance')
        return Exponential.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        squared = kwargs.pop('squared', None)
        if squared is not None and squared != False:
            self._logger.warning('A squared Laplacian kernel has been requested. The Laplacian kernel '
                                 'is defined as a particular exponential kernel with non-squared and '
                                 'euclidean distance only. This value will be neglected. Please use the '
                                 'RBF kernel if you wish to use an euclidean distance with a '
                                 'non-squared norm or the more generic Exponential kernel if you'
                                 'wish to use another distance')
        kwargs['squared'] = False
        super(Laplacian, self).__init__(*args, **kwargs)

    def __str__(self):
        if self._sigma_defined:
            return f"Laplacian kernel (sigma: {str(self.sigma)})"
        return f"Laplacian kernel (sigma undefined)"

    @property
    def hparams_fixed(self):
        return {"Kernel": "Laplacian", **super(Laplacian, self).hparams_fixed}
