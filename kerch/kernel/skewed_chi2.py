"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

import torch

from .. import utils
from .implicit import implicit, base



@utils.extend_docstring(base)
class skewed_chi2(implicit):
    r"""
    Skewed Chi Squared kernel. Often used in computer vision.

    .. math::
        k(x,y) = \prod_i \frac{2\sqrt(x_i+c) \sqrt(y_i+c)}{x_i + y_i + 2c}.


    :param c: Free parameter :math:`c`., defaults to 0.
    :param c_trainable: `True` if the gradient of :math:`c` is to be computed. If so, a graph is computed
        and :math:`c` can be updated. `False` just leads to a static computation., defaults to `False`

    """

    @utils.kwargs_decorator(
        {"c": 0., "c_trainable": False})
    def __init__(self, **kwargs):
        self._c = kwargs["c"]
        super(skewed_chi2, self).__init__(**kwargs)

        self._c_trainable = kwargs["c_trainable"]
        self._c = torch.nn.Parameter(
            torch.tensor(self._c, dtype=utils.FTYPE),
            requires_grad=self._c_trainable)

    def __str__(self):
        return f"RBF kernel (c: {self.c})."

    @property
    def c(self) -> float:
        r"""
        Parameter :math:``c` of the kernel.
        """
        if isinstance(self._c, torch.nn.Parameter):
            return self._c.data.cpu().numpy()
        return self._c

    @c.setter
    def c(self, val):
        self._reset()
        self._c.data = utils.castf(val, tensor=False, dev=self._c.device)

    @property
    def params(self):
        return {'c': self.c}

    @property
    def hparams(self):
        return {"Kernel": "Skewed Chi Squred", "Trainable c": self._c_trainable, **super(skewed_chi2, self).hparams}

    def _implicit(self, x=None, y=None):
        x, y = super(skewed_chi2, self)._implicit(x, y)

        x = x.T[:, :, None]
        y = y.T[:, None, :]

        prod = torch.sqrt(x + self.c) * torch.sqrt(y + self.c)
        sum = torch.clamp(x + y + 2 * self.c, min=self._eps)
        output = torch.prod(2 * prod / sum, dim=0, keepdim=True)

        return output.squeeze(0)

    def _slow_parameters(self, recurse=True):
        yield self._c
        yield from super(skewed_chi2, self)._slow_parameters(recurse)
