"""
File containing the linear kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from .. import utils
from ._statistics import _Statistics
from abc import ABCMeta, abstractmethod


@utils.extend_docstring(_Statistics)
class _Explicit(_Statistics, metaclass=ABCMeta):

    @utils.kwargs_decorator({})
    def __init__(self, **kwargs):
        """
        no specific parameters to the linear kernel
        """
        super(_Explicit, self).__init__(**kwargs)
        self._dim_feature = None

    def __str__(self):
        return f"Explicit kernel."

    @property
    def explicit(self) -> bool:
        return True

    @property
    def dim_feature(self) -> int:
        if self._dim_feature is None:
            # if it has not been set before, we can compute it with a minimal example
            self._dim_feature = self._explicit(x=self.current_sample[0:1, :]).shape[1]
        return self._dim_feature

    def _implicit(self, x=None, y=None):
        phi_oos = self._explicit(x)
        phi_sample = self._explicit(y)
        return phi_oos @ phi_sample.T

    @abstractmethod
    def _explicit(self, x=None):
        phi = super(_Explicit, self)._explicit(x)
        return phi

    def explicit_preimage(self, phi=None) -> torch.Tensor:
        r"""
            Returns a pseudo-inverse of the explicit feature map if available.

            .. note::
                The normalized version is not implemented

            :param phi: Image to be pseudo-inverted. Defaults to the explicit feature map of the sample.
            :type phi: Tensor(N, dim_feature)
            :return: Pseudo-inverted values of the value of phi.
            :rtype: Tensor(N, dim_input)
        """

        if phi is None:
            phi = self.phi()
        phi = self._explicit_statistics.revert(phi)
        x_tilde = self._explicit_preimage(phi)
        return self.sample_transforms.revert(x_tilde)

    @abstractmethod
    def _explicit_preimage(self, phi):
        pass

    ## DIRECT ATTRIBUTES
    @property
    def C(self) -> torch.Tensor:
        r"""
        Returns the explicit matrix on the sample datapoints.

        .. math::
            C = \frac1N\sum_i^N \phi(x_i)\phi(x_i)^\top.
        """
        return self._C()

    @property
    def Phi(self) -> torch.Tensor:
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the sample datapoints. Same as calling
        :py:func:`phi()`, but faster.
        It is loaded from memory if already computed and unchanged since then, to avoid re-computation when reccurently
        called.
        """
        return self._phi()

    @property
    def Cov(self) -> torch.Tensor:
        r"""
        Returns the covariance matrix of the sample. Same as calling self.cov().
        """
        return self.cov()

    @property
    def Corr(self) -> torch.Tensor:
        r"""
        Returns the correlation matrix of the sample. Same as calling self.corr().
        """
        return self.corr()
