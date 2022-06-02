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

    @property
    def hidden(self):
        return self._hidden.data[self._idx_sample, :]

    @hidden.setter
    def hidden(self, val):
        val = utils.castf(val, tensor=False, dev=self._hidden.device)
        if val is not None:
            if self._hidden.nelement() == 0:
                self._hidden = torch.nn.Parameter(val, requires_grad=self._hidden_trainable)
            else:
                self._hidden.data = val
                # zeroing the gradients if relevant
                if self._hidden_trainable:
                    self._hidden.grad.data[self._idx_sample, :].zero_()

    @property
    def hidden_trainable(self) -> bool:
        return self._hidden_trainable

    @hidden_trainable.setter
    def hidden_trainable(self, val: bool):
        self._hidden_trainable = val
        self._hidden.requires_grad = self._hidden_trainable

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
