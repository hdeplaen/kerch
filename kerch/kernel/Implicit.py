# coding=utf-8
"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from .. import utils
from .Kernel import Kernel
from torch import Tensor as T

from abc import ABCMeta
from ..utils import ExplicitError, extend_docstring


@extend_docstring(Kernel)
class Implicit(Kernel, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super(Implicit, self).__init__(*args, **kwargs)

    def __str__(self):
        return f"Implicit kernel."

    @property
    def explicit(self) -> bool:
        return False

    @property
    def dim_feature(self) -> int:
        raise utils.ExplicitError

    def _explicit(self, x):
        raise ExplicitError(self)

    def explicit_preimage(self, phi: T):
        raise ExplicitError(self)
