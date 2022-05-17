"""
File containing the indicator kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

import rkm
import rkm.src
import rkm.src.model.kernel.ImplicitKernel as ImplicitKernel

import torch

class IndicatorKernel(ImplicitKernel.ImplicitKernel):
    """
    RBF kernel class
    k(x,y) = exp( -||x-y||^2 / 2 * sigma^2 ).
    """

    @rkm.src.kwargs_decorator(
        {"lag": 1,
         "gamma": None})
    def __init__(self, **kwargs):
        """
        :param lag: bandwidth of the kernel (default 1)
        :param gamma: value on the diagonal (default 2 * lag + 1, which ensures PSD)
        """
        super(IndicatorKernel, self).__init__(**kwargs)
        assert self.size_in == 1, "The indicator kernel is only defined for 1-dimensional entries."

        self.lag = kwargs["lag"]
        if kwargs["gamma"] is None:
            self.gamma = 2 * self.lag + 1
        else:
            self.gamma = kwargs["gamma"]

    def __str__(self):
        return f"Indicator kernel (lag: {str(self.lag.data.cpu().numpy())}, gamma: {str(self.gamma.data.cpu().numpy())})"

    @property
    def params(self):
        return {'Lag': self.lag,
                'Gamma': self.gamma}

    @property
    def hparams(self):
        return {"Kernel": "Indicator", **super(IndicatorKernel, self).hparams}

    def _implicit(self, x_oos=None, x_sample=None):
        x_oos, x_sample = super(IndicatorKernel, self)._implicit(x_oos, x_sample)

        x_oos = x_oos.T[:, :, None]
        x_sample = x_sample.T[:, None, :]

        diff = (x_oos-x_sample).squeeze()
        assert len(diff.shape) == 2, 'Indicator kernel is only defined for 1-dimensional entries.'

        output = (torch.abs(diff) <= self.lag).type(dtype=rkm.ftype)
        output.fill_diagonal_(self.gamma)

        return output
