"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm.src
import rkm.src.model.kernel.Kernel as Kernel

import torch
from abc import ABCMeta, abstractmethod

class ImplicitKernel(Kernel.Kernel, metaclass=ABCMeta):
    """
    RBF kernel class
    k(x,y) = exp( -||x-y||^2 / 2 * sigma^2 ).
    """

    @rkm.src.kwargs_decorator(
        {"sigma": 1., "sigma_trainable": False})
    def __init__(self, **kwargs):
        """
        :param sigma: bandwidth of the kernel (default 1.)
        :param sigma_trainable: True if sigma can be trained (default False)
        """
        super(ImplicitKernel, self).__init__(**kwargs)

    def __str__(self):
        return f"Implicit kernel."

    @abstractmethod
    def _implicit(self, x_oos=None, x_sample=None):
        return super(ImplicitKernel, self)._implicit(x_oos, x_sample)

    def _explicit(self, x=None):
        raise rkm.src.model.PrimalError
