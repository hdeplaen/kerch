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
        self.sigma = torch.nn.Parameter(torch.tensor(kwargs["sigma"]), requires_grad=self.sigma_trainable)

    def __str__(self):
        return "RBF kernel"

    def implicit(self, x, idx_kernels=None):
        xs = x[:, None, :].expand(-1, len(idx_kernels), -1)
        params = self.kernels(idx_kernels).expand(x.size(0), -1, -1)

        diff = xs - params
        norm2 = torch.sum(diff * diff, dim=2)
        fact = 1 / (2 * self.sigma ** 2)
        output = torch.exp(-fact * norm2)

        return output

    def explicit(self, x, idx_kernels=None):
        raise mdl.PrimalError
