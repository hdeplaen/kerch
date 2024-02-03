# coding=utf-8
"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from ...feature.logger import _GLOBAL_LOGGER
from .exponential import Exponential
from ...utils import extend_docstring
from ..distance.euclidean import Euclidean


@extend_docstring(Euclidean)
class Laplacian(Exponential):
    r"""
    Laplacian kernel.

    .. math::
        k(x,y) = \exp\left( -\frac{\lVert x-y \rVert_2}{2\sigma^2} \right).


    .. note::
        The difference with the RBF kernel is in the squaring or not of the euclidean norm inside the exponential.

    """
    def __new__(cls, *args, **kwargs):
        distance = kwargs.pop('distance', None)
        squared = kwargs.pop('squared', None)
        if distance is not None and distance != 'euclidean':
            _GLOBAL_LOGGER._logger.warning('A specific distance has been provided for the RBF kernel. The RBF kernel '
                                           'is defined as a particular exponential kernel with euclidean distance '
                                           'only. This value will be neglected. Please use the more generic '
                                           'Exponential kernel if you wish to use another distance')
        if squared is not None and squared != False:
            _GLOBAL_LOGGER._logger.warning('A non-squared exponential kernel has been requested. The RBF kernel '
                                           'is defined as a particular exponential kernel with squared norm and '
                                           'euclidean distance only. This value will be neglected. Please use the '
                                           'Laplacian kernel if you wish to use an euclidean distance with a '
                                           'non-squared norm or the more generic Exponential kernel if you'
                                           'wish to use another distance')
        kwargs['squared'] = False
        return Exponential.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super(Laplacian, self).__init__(*args, **kwargs)

    def __str__(self):
        if self._sigma_defined:
            return f"Laplacian kernel (sigma: {str(self.sigma)})"
        return f"Laplacian kernel (sigma undefined)"

    @property
    def hparams_fixed(self):
        return {"Kernel": "Laplacian", **super(Laplacian, self).hparams_fixed}
