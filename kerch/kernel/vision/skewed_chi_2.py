# coding=utf-8
"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""
from typing import Iterator
import torch

from ... import utils
from ..implicit import Implicit, Kernel



@utils.extend_docstring(Kernel)
class SkewedChi2(Implicit):
    r"""
    Skewed Chi Squared kernel. Often used in computer vision.

    .. math::
        k(x,y) = \prod_i \frac{2\sqrt{x_i+p} \sqrt{y_i+p}}{x_i + y_i + 2}.


    :param p: Free parameter :math:`p`., defaults to 0.
    :param p_trainable: `True` if the gradient of :math:`p` is to be computed. If so, a graph is computed
        and :math:`p` can be updated. `False` just leads to a static computation., defaults to `False`
    :type p: float, optional
    :type p_trainable: bool, optional
    """
    def __init__(self, *args, **kwargs):
        self._p = kwargs.pop('p', 0.)
        super(SkewedChi2, self).__init__(*args, **kwargs)

        self._p_trainable = kwargs.pop('p_trainable', False)
        self._p = torch.nn.Parameter(
            torch.tensor(self._p, dtype=utils.FTYPE),
            requires_grad=self._p_trainable)

    @property
    def p(self) -> float:
        r"""
        Parameter :math:`p` of the kernel.
        """
        if isinstance(self._p, torch.nn.Parameter):
            return self._p.data.cpu().numpy().astype(float)
        return float(self._p)

    @p.setter
    def p(self, val):
        self._reset_cache(reset_persisting=False)
        self._p.data = utils.castf(val, tensor=False, dev=self._p.device)

    def __str__(self):
        return f"Skewed Chi Squared kernel (p: {self.p})."

    @property
    def hparams_variable(self):
        return {'Kernel parameter p': self.p}

    @property
    def hparams_fixed(self):
        return {"Kernel": "Skewed Chi Squred", "Trainable p": self._p_trainable, **super(SkewedChi2, self).hparams_fixed}

    def _implicit(self, x, y):
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        prod = torch.sqrt(x + self._p) * torch.sqrt(y + self._p)
        sum = torch.clamp(x + y + 2 * self._p, min=utils.EPS)
        output = torch.prod(2 * prod / sum, dim=0, keepdim=True)

        return output.squeeze(0)

    def _slow_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        yield self._p
        yield from super(SkewedChi2, self)._slow_parameters(recurse)
