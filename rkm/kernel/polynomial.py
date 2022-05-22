"""
File containing the polynomial kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from .. import utils
from .base import base

import torch

@utils.extend_docstring(base)
class polynomial(base):
    r"""
    Polynomial kernel.

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
        super(polynomial, self).__init__(**kwargs)

        self.degree_trainable = kwargs["degree_trainable"]
        self.degree = torch.nn.Parameter(torch.tensor(kwargs["degree"]), requires_grad=self.degree_trainable)

    def __str__(self):
        return f"polynomial kernel of order {int(self.degree.data)}"

    @property
    def params(self):
        return {'Degree': self.degree}

    @property
    def hparams(self):
        return {"Kernel": "Polynomial"}

    def _implicit(self, x_oos=None, x_sample=None):
        x_oos, x_sample = super(polynomial, self)._implicit(x_oos, x_sample)
        return (x_oos.T @ x_sample + 1) ** self.degree

    def _explicit(self, x=None):
        # A primal representation is technically possible here (we will only consider the dual representation now).
        # raise rkm.PrimalError

        # if self.degree == 1.:
        #     return super(PolynomialKernel, self)._explicit(x)
        # elif self.degree == 2.:
        #     return 0
        # else:
        #     pass

        raise NotImplementedError
