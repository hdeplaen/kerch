"""
File containing the linear kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm.src
import rkm.src.model.kernel.Kernel as Kernel
from abc import ABCMeta, abstractmethod


class ExplicitKernel(Kernel.Kernel, metaclass=ABCMeta):
    """
    Linear kernel class
    k(x,y) = < x,y >.
    """

    @rkm.src.kwargs_decorator({})
    def __init__(self, **kwargs):
        """
        no specific parameters to the linear kernel
        """
        super(ExplicitKernel, self).__init__(**kwargs)

    def __str__(self):
        return f"Explicit kernel."

    def _implicit(self, x_oos=None, x_sample=None):
        phi_oos = self._explicit(x_oos)
        phi_sample = self._explicit(x_sample)
        return phi_oos @ phi_sample.T

    @abstractmethod
    def _explicit(self, x=None):
        return super(ExplicitKernel, self)._explicit(x)
    
    def _dmatrix(self, implicit=True):
        return super(ExplicitKernel, self)._dmatrix(implicit)

    def k(self, x_oos=None, x_sample=None, implicit=True):
        return super(ExplicitKernel, self).k(x_oos, x_sample, implicit)