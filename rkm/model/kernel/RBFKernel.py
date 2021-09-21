"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm
import rkm.model as mdl

import torch

class RBFKernel(mdl.kernel.Kernel):
    """
    RBF kernel class
    k(x,y) = exp( -||x-y||^2 / 2 * sigma^2 ).
    """

    @rkm.kwargs_decorator(
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

    def reset(self, idx_kernels=None):
        super(RBFKernel, self).reset(idx_kernels=idx_kernels)
        # try:
        #     self.sigma.data = self.sigma.data + torch.normal(0, .001, size=(1,1))[0,0]
        # except AttributeError:
        #     pass

    def _implicit(self, x):
        xs = x[:, None, :].expand(-1, self.num_idx(), -1)
        params = self.kernels[self._idx_kernels,:].expand(x.size(0), -1, -1)

        diff = xs - params
        norm2 = torch.sum(diff * diff, dim=2)
        fact = 1 / (2 * torch.abs(self.sigma) ** 2)
        output = torch.exp(torch.mul(norm2, -fact))

        return output

    def _explicit(self, x):
        raise mdl.PrimalError
