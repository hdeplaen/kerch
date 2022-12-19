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

    @abstractmethod
    def phi_pinv(self, phi=None, centered=None, normalized=None) -> torch.Tensor:
        r"""
            Returns a pseudo-inverse of the explicit feature map if available.

            .. note::
                The normalized version is not implemented

            :param phi: Image to be pseudo-inverted. Defaults to the explicit feature map of the sample.
            :name phi: Tensor(N, dim_feature)
            :param centered: Indicates whether the explicit feature map is centered and has to be "de-centered"
                before being inverted. Defaults to the default value used to compute the explicit feature map phi.
            :name centered: bool
            :param normalized: Indicated whether the explicit feature map is normalized and has to be be scaled before
                being pseudo-inverted. Defaults to the default value used to compute the explicit feature map phi.
            :name normalized: bool
            :return: Pseudo-inverted values of the value of phi.
            :rtype: Tensor(N, dim_input)
        """

        raise NotImplementedError

        # if phi is None:
        #     phi = self.phi()
        # if centered is None:
        #     centered = self._center
        # if normalized is None:
        #     normalized = self._normalize
        # if normalized:
        #     self._log.error("Pseudo-inversion of normalized explicit feature maps is not implemented.")
        #     raise NotImplementedError
        # if centered:
        #     if self._explicit_statistics() is None:
        #         self._log.error('Impossible to compute statistics on the sample (probably due to an undefined sample.')
        #         raise Exception
        #     phi = phi + self._cache["phi_mean"]
