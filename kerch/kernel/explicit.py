"""
File containing the linear kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from .. import utils
from .base import base
from abc import ABCMeta, abstractmethod

@utils.extend_docstring(base)
class explicit(base, metaclass=ABCMeta):

    @utils.kwargs_decorator({})
    def __init__(self, **kwargs):
        """
        no specific parameters to the linear kernel
        """
        super(explicit, self).__init__(**kwargs)
        self._dim_feature = None

    def __str__(self):
        return f"Explicit kernel."

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
        phi = super(explicit, self)._explicit(x)
        return phi

    def _compute_K(self, implicit=True):
        return super(explicit, self)._compute_K(implicit)

    def k(self, x=None, y=None, implicit=True):
        return super(explicit, self).k(x, y, implicit)
