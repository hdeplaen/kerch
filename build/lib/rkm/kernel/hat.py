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



@utils.extend_docstring(base)
class hat(implicit):
    r"""
    Hat kernel.

    .. math::
        k(x,y) = \left\{
        \begin{array}
        lp + 1 - |x-y| & \text{ if } |x-y|\leq p, \\
        0 & \text{ otherwise.}
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

        self._lag_trainable = kwargs["lag_trainable"]
        self._lag = torch.nn.Parameter(
            torch.tensor(kwargs["lag"], dtype=utils.FTYPE), requires_grad=self._lag_trainable)

        self._relu = torch.nn.ReLU(inplace=False)

    def __str__(self):
        return f"Hat kernel (lag: {str(self._lag.data.cpu().numpy())})"

    @property
    def lag(self):
        r"""
        Lah :math:`p` of the kernel.
        """
        return self._lag.data.cpu().numpy()

    @lag.setter
    def lag(self, val):
        self._reset()
        self._lag.data = utils.castf(val, tensor=False, dev=self._lag.device)

    @property
    def lag_trainable(self) -> bool:
        r"""
        Boolean indicating if the lag :math:`p` is trainable.
        """
        return self._sigma_trainable

    @lag_trainable.setter
    def lag_trainable(self, val: bool):
        self._lag_trainable = val
        self._lag.requires_grad = self._lag_trainable

    @property
    def params(self):
        return {'Lag': self.lag}

    @property
    def hparams(self):
        return {"Kernel": "Hat", **super(hat, self).hparams}

    def _implicit(self, x=None, y=None):
        x, y = super(hat, self)._implicit(x, y)

        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = (x-y).squeeze()
        assert len(diff.shape) == 2, 'Hat kernel is only defined for 1-dimensional entries.'

        output = self._lag + 1 - torch.abs(diff)
        output = self._relu(output)

        return output
