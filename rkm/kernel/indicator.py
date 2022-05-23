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
class indicator(implicit):
    r"""
    Indicator kernel.

    .. math::
        k(x,y) = \left\{
        \begin{array}[lll]
        \texttt{gamma} & \text{ if } & |x-y|=0, \\
        1 & \text{ if } & 0 < |x-y| \leq \texttt{lag}, \\
        0 & \text{ otherwise.} &
        \end{array}
        \right.

    .. note ::
        If the default value for `gamma` is used and the `lag` is to be trained, their two values will be linked.

    .. warning::
        Depending on the choice of `gamma`, the kernel may not be positive semi-definite. The default value however
        ensures it.

    .. warning::
        For this type of kernel, the input dimension of the datapoints `dim_sample` must be 1.

    :param lag: Lag parameter., defaults to 1.
    :param gamma: Identity value of the kernel. If `None`, the value will be `gamma`:math:` = 2*``lag`:math:`+1` to
        ensure positive semi-definiteness., defaults to `None`
    :param lag_trainable: `True` if the gradient of the lag is to be computed. If so, a graph is computed
        and the lag can be updated. `False` just leads to a static computation., defaults to `False`
    :param gamma_trainable: `True` if the gradient of the gamma is to be computed. If so, a graph is computed
        and the gamma can be updated. `False` just leads to a static computation., this value will be tied to the
        evolution of `lag`., defaults to `False`
    :type lag: double, optional
    :type gamma: double, optional
    :type lag_trainable: bool, optional
    :type gamma_trainable: bool, optional
    """

    @utils.kwargs_decorator(
        {"lag": 1,
         "gamma": None,
         "lag_trainable": False,
         "gamma_trainable": False})
    def __init__(self, **kwargs):
        """
        :param lag: bandwidth of the kernel (default 1)
        :param gamma: value on the diagonal (default 2 * lag + 1, which ensures PSD)
        """
        super(indicator, self).__init__(**kwargs)
        assert self._dim_sample == 1, "The indicator kernel is only defined for 1-dimensional entries."

        self.lag_trainable = kwargs["lag_trainable"]
        self.lag = torch.nn.Parameter(
            torch.tensor(kwargs["lag"], dtype=utils.FTYPE), requires_grad=self.lag_trainable)

        self.gamma_trainable = kwargs["gamma_trainable"]
        if kwargs["gamma"] is None:
            self._link_training = True
            self._gamma = torch.nn.Parameter(
                torch.tensor(2 * self.lag + 1, dtype=utils.FTYPE), requires_grad=False)
        else:
            self._link_training = False
            self._gamma = torch.nn.Parameter(
                torch.tensor(kwargs["gamma"], dtype=utils.FTYPE), requires_grad=self.gamma_trainable)

    def __str__(self):
        return f"Indicator kernel (lag: {str(self.lag.data.cpu().numpy())}, gamma: {str(self.gamma.data.cpu().numpy())})"

    @property
    def params(self):
        return {'Lag': self.lag,
                'Gamma': self.gamma}

    @property
    def hparams(self):
        return {"Kernel": "Indicator", **super(indicator, self).hparams}

    @property
    def gamma(self):
        if self._link_training and self.lag_trainable:
            self._gamma.data = 2 * self.lag + 1
        return self._gamma

    def _implicit(self, x_oos=None, x_sample=None):
        x_oos, x_sample = super(indicator, self)._implicit(x_oos, x_sample)

        x_oos = x_oos.T[:, :, None]
        x_sample = x_sample.T[:, None, :]

        diff = (x_oos - x_sample).squeeze()
        assert len(diff.shape) == 2, 'Indicator kernel is only defined for 1-dimensional entries.'

        output = (torch.abs(diff) <= self.lag).type(dtype=utils.FTYPE)
        output[diff == 0] = self.gamma

        return output
