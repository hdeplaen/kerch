"""
File containing the polynomial kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from .. import utils
from ._projected import _Projected

import torch


@utils.extend_docstring(_Projected)
class Polynomial(_Projected):
    r"""
    Polynomial kernel. Projection onto a hypershpere.

    .. math::
        k(x,y) = \left(x^\top y + 1\right)^\texttt{degree}.

    .. note ::
        An explicit feature map also corresponds to this kernel, but is not implemented.

    :param degree: Degree of the polynomial kernel., defaults to 1
    :param degree_trainable: `True` if the gradient of the degree is to be computed. If so, a graph is computed
        and the degree can be updated. `False` just leads to a static computation., defaults to `False`
    :type degree: double, optional
    :type degree_trainable: bool, optional
    """

    @utils.kwargs_decorator(
        {"degree": 2., "degree_trainable": False})
    def __init__(self, **kwargs):
        self._degree = kwargs["degree"]
        super(Polynomial, self).__init__(**kwargs)

        self._degree_trainable = kwargs["degree_trainable"]
        self._degree = torch.nn.Parameter(torch.tensor(self._degree),
                                          requires_grad=self.degree_trainable)

    def __str__(self):
        return f"polynomial kernel of order {self.degree}."

    @property
    def explicit(self) -> bool:
        self._log.debug("The explicit formulation of the polynomial kernel is not implemented.")
        return False

    @property
    def dim_feature(self) -> int:
        raise NotImplementedError

    @property
    def degree(self):
        r"""
        Degree of the polynomial.
        """
        if isinstance(self._degree, torch.nn.Parameter):
            return self._degree.cpu().numpy()
        return self._degree

    @degree.setter
    def degree(self, val):
        self._reset_cache()
        self._degree.data = val

    @property
    def degree_trainable(self) -> bool:
        r"""
        Boolean indicating if the degree is trainable.
        """
        return self._degree_trainable

    @degree_trainable.setter
    def degree_trainable(self, val: bool):
        self._degree_trainable = val
        self._degree.requires_grad = self._degree_trainable

    @property
    def params(self):
        return {'Degree': self.degree}

    @property
    def hparams(self):
        return {"Kernel": "Polynomial"}

    def _implicit(self, x, y):
        return (x @ y.T + 1) ** self._degree

    def _explicit(self, x):
        assert (self.degree % 1) == 0, '_Explicit formulation is only possible for degrees that are natural numbers.'
        raise NotImplementedError

    def explicit_preimage(self, phi):
        raise NotImplementedError

    def _slow_parameters(self, recurse=True):
        yield self._degree
        yield from super(Polynomial, self)._slow_parameters(recurse)
