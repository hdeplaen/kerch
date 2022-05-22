"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from .. import utils
from .base import base

from abc import ABCMeta, abstractmethod


class implicit(base, metaclass=ABCMeta):
    @utils.kwargs_decorator(
        {"sigma": 1., "sigma_trainable": False})
    def __init__(self, **kwargs):
        """
        :param sigma: bandwidth of the kernel (default 1.)
        :param sigma_trainable: True if sigma can be trained (default False)
        """
        super(implicit, self).__init__(**kwargs)

    def __str__(self):
        return f"Implicit kernel."

    @abstractmethod
    def _implicit(self, x_oos=None, x_sample=None):
        return super(implicit, self)._implicit(x_oos, x_sample)

    def _explicit(self, x=None):
        raise utils.model.PrimalError