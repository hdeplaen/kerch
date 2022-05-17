"""
File containing the indicator kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

import rkm.src
import rkm.src.model.kernel.ImplicitKernel as ImplicitKernel

import torch

class HatKernel(ImplicitKernel.ImplicitKernel):
    """
    RBF kernel class
    k(x,y) = exp( -||x-y||^2 / 2 * sigma^2 ).
    """

    @rkm.src.kwargs_decorator(
        {"lag": 1})
    def __init__(self, **kwargs):
        """
        :param lag: bandwidth of the kernel (default 1)
        :param gamma: value on the diagonal (default 2 * lag + 1, which ensures PSD)
        """
        super(HatKernel, self).__init__(**kwargs)
        assert self.size_in == 1, "The hat kernel is only defined for 1-dimensional entries."

        self.lag = kwargs["lag"]

    def __str__(self):
        return f"Hat kernel (lag: {str(self.lag.data.cpu().numpy())})"

    @property
    def params(self):
        return {'Lag': self.lag}

    @property
    def hparams(self):
        return {"Kernel": "Hat", **super(HatKernel, self).hparams}

    def _implicit(self, x_oos=None, x_sample=None):
        x_oos, x_sample = super(HatKernel, self)._implicit(x_oos, x_sample)

        x_oos = x_oos[:, :, None]
        x_sample = x_sample[:, None, :]

        diff = (x_oos-x_sample).squeeze()
        assert len(diff.shape) == 2, 'Hat kernel is only defined for 1-dimensional entries.'

        output = self.lag + 1 - torch.abs(diff)
        output = torch.nn.ReLU(output)

        return output
