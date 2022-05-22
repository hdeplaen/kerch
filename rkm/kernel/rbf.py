"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from .. import utils
from .implicit import implicit, base

@utils.extend_docstring(base)
class rbf(implicit):
    r"""
    RBF kernel (radial basis function).

    .. math::
        k(x,y) = \exp\left( -\frac{\lVert x-y \rVert_2^2}{2\texttt{sigma}^2} \right).

    :param sigma: Bandwidth of the polynomial kernel., defaults to 1.
    :param sigma_trainable: `True` if the gradient of the bandwidth is to be computed. If so, a graph is computed
        and the bandwidth can be updated. `False` just leads to a static computation., defaults to `False`
    :type sigma: double, optional
    :type sigma_trainable: bool, optional
    """

    @utils.kwargs_decorator(
        {"sigma": 1., "sigma_trainable": False})
    def __init__(self, **kwargs):
        super(rbf, self).__init__(**kwargs)

        self.sigma_trainable = kwargs["sigma_trainable"]
        self.sigma = torch.nn.Parameter(
            torch.tensor([kwargs["sigma"]], dtype=utils.FTYPE), requires_grad=self.sigma_trainable)

    def __str__(self):
        return f"RBF kernel (sigma: {str(self.sigma.data.cpu().numpy()[0])})"

    @property
    def params(self):
        return {'Sigma': self.sigma}

    @property
    def hparams(self):
        return {"Kernel": "RBF", "Trainable sigma": self.sigma_trainable, **super(rbf, self).hparams}

    def _implicit(self, x_oos=None, x_sample=None):
        x_oos, x_sample = super(rbf, self)._implicit(x_oos, x_sample)

        x_oos = x_oos.T[:, :, None]
        x_sample = x_sample.T[:, None, :]

        diff = x_oos - x_sample
        norm2 = torch.sum(diff * diff, dim=0, keepdim=True)
        fact = 1 / (2 * torch.abs(self.sigma) ** 2)
        output = torch.exp(torch.mul(norm2, -fact))

        return output.squeeze(0)
