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

    def __str__(self):
        return f"Explicit kernel."

    def _implicit(self, x_oos=None, x_sample=None):
        phi_oos = self._explicit(x_oos)
        phi_sample = self._explicit(x_sample)
        return phi_oos @ phi_sample.T

    @abstractmethod
    def _explicit(self, x=None):
        return super(explicit, self)._explicit(x)
    
    def _dmatrix(self, implicit=True):
        return super(explicit, self)._dmatrix(implicit)

    def k(self, x_oos=None, x_sample=None, implicit=True):
        return super(explicit, self).k(x_oos, x_sample, implicit)