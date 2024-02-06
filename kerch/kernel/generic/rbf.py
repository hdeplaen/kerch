# coding=utf-8
"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""
from ...utils import extend_docstring
from ...feature.logger import _GLOBAL_LOGGER
from kerch.kernel.statistics.exponential import Exponential
from ..distance.euclidean import Euclidean


@extend_docstring(Euclidean)
class RBF(Exponential):
    r"""
    RBF kernel (radial basis function) of bandwidth :math:`\sigma>0`.

    .. math::
        k(x,y) = \exp\left( -\frac{\lVert x-y \rVert_2^2}{2\sigma^2} \right).


    .. note::
        If working with big datasets, one may consider an explicit approximation of the RBF kernel using
        Random Fourier Features (:py:class:`~kerch.kernel.RFF`). This will be faster provided :math:`2\times\texttt{num_weights} < n`,
        where :math:`\texttt{num_weights}` is the number of weights used to control the RFF approximation and :math:`n` is
        the number of datapoints. The latter class however does not offer so much flexibility, as the automatic determination
        of the bandwidth :math:`\sigma` using a heuristic for example.

        Other considerations may come into play. If a centered or normalized kernel on an out-of-sample is required, this may require extra
        computations when directly using the kernel matrix as doing it on the explicit feature is more straightforward.


    .. note::
        The norm inside the exponential is always squared. If you wish a non-squared norm, this corresponds to the
        :py:class:`~kerch.kernel.Laplacian` kernel. If another distance than the Euclidean one is required, we refer to
        the more generic :py:class:`~kerch.kernel.Exponential` kernel.

    """

    def __new__(cls, *args, **kwargs):
        distance = kwargs.pop('distance', None)
        if distance is not None and distance != 'euclidean':
            _GLOBAL_LOGGER._logger.warning('A specific distance has been provided for the RBF kernel. The RBF kernel '
                                           'is defined as a particular exponential kernel with euclidean distance '
                                           'only. This value will be neglected. Please use the more generic '
                                           'Exponential kernel if you wish to use another distance')

        return Exponential.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        squared = kwargs.pop('squared', None)
        if squared is not None and squared != True:
            self._logger.warning('A non-squared RBF kernel has been requested. The RBF kernel '
                                 'is defined as a particular exponential kernel with squared and '
                                 'euclidean distance only. This value will be neglected. Please use the '
                                 'Laplacian kernel if you wish to use an euclidean distance with a '
                                 'non-squared norm or the more generic Exponential kernel if you'
                                 'wish to use another distance')
        super(RBF, self).__init__(*args, **kwargs)

    def __str__(self):
        if self._sigma_defined:
            return f"RBF kernel (sigma: {self.sigma:1.2e})"
        return f"RBF kernel (sigma undefined)"

    @property
    def hparams_fixed(self):
        return {"Kernel": "RBF", **super(RBF, self).hparams_fixed}
