"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from .. import utils
from ._statistics import _Statistics
from torch import Tensor as T

from abc import ABCMeta, abstractmethod
from ..utils import ExplicitError


class _Implicit(_Statistics, metaclass=ABCMeta):
    @utils.kwargs_decorator(
        {"sigma": 1., "sigma_trainable": False})
    def __init__(self, **kwargs):
        """
        :param sigma: bandwidth of the kernel (default 1.)
        :param sigma_trainable: True if sigma can be trained (default False)
        """
        super(_Implicit, self).__init__(**kwargs)

    def __str__(self):
        return f"Implicit kernel."

    @property
    def explicit(self) -> bool:
        return False

    @property
    def dim_feature(self) -> int:
        raise utils.ExplicitError

    @abstractmethod
    def _implicit(self, x=None, y=None):
        return super(_Implicit, self)._implicit(x, y)

    def _explicit(self, x=None):
        raise utils.ExplicitError(self)

    def explicit_preimage(self, phi: T):
        raise utils.ExplicitError(self)
