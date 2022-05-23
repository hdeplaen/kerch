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
        super(skewed_chi2, self).__init__(**kwargs)

        self._c_trainable = kwargs["c_trainable"]
        self._c = torch.nn.Parameter(
            torch.tensor([kwargs["c"]], dtype=utils.FTYPE), requires_grad=self._c_trainable)

    def __str__(self):
        return f"RBF kernel (c: {str(self.c.cpu().numpy()[0])})"

    @property
    def c(self):
        return self._c.data

    @property
    def params(self):
        return {'Sigma': self.sigma}

    @property
    def hparams(self):
        return {"Kernel": "Skewed Chi Squred", "Trainable c": self._c_trainable, **super(skewed_chi2, self).hparams}

    def _implicit(self, x_oos=None, x_sample=None):
        x_oos, x_sample = super(skewed_chi2, self)._implicit(x_oos, x_sample)

        x_oos = x_oos.T[:, :, None]
        x_sample = x_sample.T[:, None, :]

        prod = torch.sqrt(x_oos + self.c) * torch.sqrt(x_sample + self.c)
        sum = x_oos + x_sample + 2 * self.c
        output = torch.prod(2 * prod / sum, dim=0, keepdim=True)

        return output.squeeze(0)
