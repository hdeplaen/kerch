"""
File containing the indicator kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

from .. import utils
from .implicit import implicit, base

import torch


@torch.jit.script
@utils.extend_docstring(base)
class hat(implicit):
    r"""
    Hat kernel.

    .. math::
        k(x,y) = \left\{
        \begin{array}[lll]
        \texttt{lag} + 1 - |x-y| & \text{ if } & |x-y|\leq \texttt{lag}, \\
        0 & \text{ otherwise.} &
        \end{array}
        \right.

    :param lag: Lag parameter., defaults to 1
    :param lag_trainable: `True` if the gradient of the lag is to be computed. If so, a graph is computed
        and the lag can be updated. `False` just leads to a static computation., defaults to `False`
    :type lag: double, optional
    :type lag_trainable: bool, optional
    """

    @utils.kwargs_decorator(
        {"lag": 1,
         "lag_trainable": False})
    def __init__(self, **kwargs):
        super(hat, self).__init__(**kwargs)
        assert self._dim_sample == 1, "The hat kernel is only defined for 1-dimensional entries."

        self.lag_trainable = kwargs["lag_trainable"]
        self.lag = torch.nn.Parameter(
            torch.tensor([kwargs["lag"]], dtype=utils.FTYPE), requires_grad=self.lag_trainable)

    def __str__(self):
        return f"Hat kernel (lag: {str(self.lag.data.cpu().numpy())})"

    @property
    def params(self):
        return {'Lag': self.lag}

    @property
    def hparams(self):
        return {"Kernel": "Hat", **super(hat, self).hparams}

    def _implicit(self, x_oos=None, x_sample=None):
        x_oos, x_sample = super(hat, self)._implicit(x_oos, x_sample)

        x_oos = x_oos[:, :, None]
        x_sample = x_sample[:, None, :]

        diff = (x_oos-x_sample).squeeze()
        assert len(diff.shape) == 2, 'Hat kernel is only defined for 1-dimensional entries.'

        output = self.lag + 1 - torch.abs(diff)
        output = torch.nn.ReLU(output)

        return output
