# coding=utf-8
"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from .. import utils
from .kernel import Kernel
from torch import Tensor as T
from math import inf

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
    def dim_feature(self) -> int | inf:
        r"""
        For implicit kernels, the feature dimension is infinite.
        """
        return inf

    def _explicit(self, x):
        raise ExplicitError(cls=self)

    def explicit_preimage(self, phi_image: T | None = None, method: str = 'explicit', **kwargs):
        r"""

        .. note::
            Not available for kernels that have no explicit feature map representation.


        """
        raise ExplicitError(cms=self)

    @abstractmethod
    def _implicit(self, x, y):
        pass

