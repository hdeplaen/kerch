# coding=utf-8
"""
File containing the linear kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from .. import utils
from .Kernel import Kernel
from abc import ABCMeta, abstractmethod


@utils.extend_docstring(Kernel)
class Explicit(Kernel, metaclass=ABCMeta):

    @utils.kwargs_decorator({})
    def __init__(self, *args, **kwargs):
        super(Explicit, self).__init__(*args, **kwargs)
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
            self._dim_feature = self._explicit_with_none(x=self.current_sample_projected[0:1, :]).shape[1]
        return self._dim_feature

    def _implicit(self, x, y):
        phi_oos = self._explicit(x)
        phi_sample = self._explicit(y)
        return phi_oos @ phi_sample.T

    @abstractmethod
    def _explicit(self, x):
        return x

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
        phi = self._explicit_Projected.revert(phi)
        x_tilde = self._explicit_preimage(phi)
        return self.sample_transform.revert(x_tilde)

    @abstractmethod
    def _explicit_preimage(self, phi):
        pass
