"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

import torch

from .. import utils
from ._implicit import _Implicit, _Statistics



@utils.extend_docstring(_Statistics)
class SkewedChi2(_Implicit):
    r"""
    Skewed Chi Squared kernel. Often used in computer vision.

    .. math::
        k1(x,y) = \prod_i \frac{2\sqrt(x_i+p) \sqrt(y_i+p)}{x_i + y_i + 2}.


    :param p: Free parameter :math:`p`., defaults to 0.
    :param p_trainable: `True` if the gradient of :math:`p` is to be computed. If so, a graph is computed
        and :math:`p` can be updated. `False` just leads to a static computation., defaults to `False`

    """

    @utils.kwargs_decorator(
        {"p": 0., "p_trainable": False})
    def __init__(self, **kwargs):
        self._p = kwargs["p"]
        super(SkewedChi2, self).__init__(**kwargs)

        self._p_trainable = kwargs["p_trainable"]
        self._p = torch.nn.Parameter(
            torch.tensor(self._p, dtype=utils.FTYPE),
            requires_grad=self._p_trainable)

    @property
    def p(self) -> float:
        r"""
        Parameter :math:``p` of the kernel.
        """
        if isinstance(self._p, torch.nn.Parameter):
            return self._p.data.cpu().numpy().astype(float)
        return float(self._p)

    @p.setter
    def p(self, val):
        self._reset_cache()
        self._p.data = utils.castf(val, tensor=False, dev=self._p.device)

    def __str__(self):
        return f"Skewed Chi Squared kernel (p: {self.p})."

    @property
    def params(self):
        return {'p': self.p}

    @property
    def hparams(self):
        return {"Kernel": "Skewed Chi Squred", "Trainable p": self._p_trainable, **super(SkewedChi2, self).hparams}

    def _implicit(self, x=None, y=None):
        x, y = super(SkewedChi2, self)._implicit(x, y)

        x = x.T[:, :, None]
        y = y.T[:, None, :]

        prod = torch.sqrt(x + self._p) * torch.sqrt(y + self._p)
        sum = torch.clamp(x + y + 2 * self._p, min=utils.EPS)
        output = torch.prod(2 * prod / sum, dim=0, keepdim=True)

        return output.squeeze(0)

    def _slow_parameters(self, recurse=True):
        yield self._p
        yield from super(SkewedChi2, self)._slow_parameters(recurse)
