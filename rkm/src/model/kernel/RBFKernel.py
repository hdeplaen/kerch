"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

import rkm.src
import rkm.src.model.kernel.ImplicitKernel as ImplicitKernel


class RBFKernel(ImplicitKernel.ImplicitKernel):
    """
    RBF kernel class
    k(x,y) = exp( -||x-y||^2 / 2 * sigma^2 ).
    """

    @rkm.src.kwargs_decorator(
        {"sigma": 1., "sigma_trainable": False})
    def __init__(self, **kwargs):
        """
        :param sigma: bandwidth of the kernel (default 1.)
        :param sigma_trainable: True if sigma can be trained (default False)
        """
        super(RBFKernel, self).__init__(**kwargs)

        self.sigma_trainable = kwargs["sigma_trainable"]
        self.sigma = torch.nn.Parameter(
            torch.tensor([kwargs["sigma"]], dtype=rkm.ftype), requires_grad=self.sigma_trainable)

    def __str__(self):
        return f"RBF kernel (sigma: {str(self.sigma.data.cpu().numpy()[0])})"

    @property
    def params(self):
        return {'Sigma': self.sigma}

    @property
    def hparams(self):
        return {"Kernel": "RBF", "Trainable sigma": self.sigma_trainable, **super(RBFKernel, self).hparams}

    def _implicit(self, x_oos=None, x_sample=None):
        x_oos, x_sample = super(RBFKernel, self)._implicit(x_oos, x_sample)

        x_oos = x_oos.T[:, :, None]
        x_sample = x_sample.T[:, None, :]

        diff = x_oos - x_sample
        norm2 = torch.sum(diff * diff, dim=0, keepdim=True)
        fact = 1 / (2 * torch.abs(self.sigma) ** 2)
        output = torch.exp(torch.mul(norm2, -fact))

        return output.squeeze(0)
