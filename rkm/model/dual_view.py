"""
Abstract RKM view class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from .. import utils
from .view import view


@utils.extend_docstring(view)
class dual_view(view):
    r"""
    :param hidden: Hidden variables
    :type hidden: torch.nn.Parameter
    """

    def __init__(self, hidden: torch.nn.Parameter, **kwargs):
        """
        A view is made of a kernel and primal or dual variables. This second part is handled by the daughter classes.
        """
        super(dual_view, self).__init__(**kwargs)
        self._hidden = hidden

        assert len(self._hidden.shape) == 2, "The hidden variables must be a matrix."
        self._num_h, self._dim_output = self._hidden.shape
        assert self._num_h == self._kernel.num_sample, "The input dimension is not consistent with the sample size " \
                                                       "of the kernel."

    ## MATHS

    def h(self, x=None):
        if x is None:
            return self._hidden[self._idx_sample, :]
        raise NotImplementedError

    def w(self, x=None):
        return utils.PrimalError

    def wh(self, x=None):
        return utils.PrimalError

    def phiw(self, x=None):
        return self._kernel.k(x) @ self.H

    ## MISC

    @staticmethod
    def from_primal(cls, primal_view: view):
        r"""
        Returns a dual view based on a primal one. This is always possible.
        """
        raise NotImplementedError
