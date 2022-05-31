"""
File containing the sigmoid kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from .. import utils
from .implicit import implicit, base

import torch


@utils.extend_docstring(base)
class sigmoid(implicit):
    r"""
    Sigmoid kernel.

    .. math::
        k(x,y) = \sigma\left( a (x^\top y) + b \right),

    where :math:`\sigma(\cdot)` is the sigmoid function.

    .. warning::
        This kernel is not positive semi-definite. Normalization after centering is not possible.

    :param a: Value for :math:`a`., defaults to 1
    :param b: Value for :math:`b`., defaults to 0
    :param params: `True` if the gradient of :math:`a` and :math:`b` are to be computed. If so, a graph is computed
        and the parameters can be updated. `False` just leads to a static computation., defaults to `False`
    :type a: double, optional
    :type b: double, optional
    :type params_trainable: bool, optional
    """

    @utils.kwargs_decorator(
        {"a": 1., "b": 0., "params_trainable": False})
    def __init__(self, **kwargs):
        super(sigmoid, self).__init__(**kwargs)

        self._params_trainable = kwargs["params_trainable"]
        self._a = torch.nn.Parameter(
            torch.tensor(kwargs["a"]).unsqueeze(dim=0), requires_grad=self.params_trainable)
        self._b = torch.nn.Parameter(
            torch.tensor(kwargs["b"]).unsqueeze(dim=0), requires_grad=self.params_trainable)
        self._linear = lambda x: self._a * x + self._b

    def __str__(self):
        return "Sigmoid kernel."

    @property
    def params(self):
        return {'a': self._a,
                'b': self._b}

    @property
    def hparams(self):
        return {"Kernel": "Sigmoid", **super(sigmoid, self).hparams}

    @property
    def params_trainable(self):
        r"""
        Boolean returning if the parameters a trainable or not.
        """
        return self._params_trainable

    def _implicit(self, x=None, y=None):
        x, y = super(sigmoid, self)._implicit(x, y)
        K = x @ y.T
        K = self._linear(K)
        return torch.sigmoid(K)