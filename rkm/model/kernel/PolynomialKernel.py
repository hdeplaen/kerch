"""
File containing the polynomial kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm
import rkm.model as mdl

import torch

class PolynomialKernel(mdl.kernel.LinearKernel.LinearKernel):
    """
    Polynomial kernel class
    k(x,y) = ( < x,y > + 1 ) ** d.
    """

    @rkm.kwargs_decorator(
        {"deg": 1., "deg_trainable": False})
    def __init__(self, **kwargs):
        """
        :param deg: degree of the kernel (default 1)
        :param deg_trainable: True if the degree can be trained (default False)
        """
        super(PolynomialKernel, self).__init__(**kwargs)

        self.deg_trainable = kwargs["deg_trainable"]
        self.deg = torch.nn.Parameter(torch.tensor(kwargs["deg"].unsqueeze(0)), requires_grad=self.deg_trainable)

    def __str__(self):
        return f"polynomial kernel of order {int(self.deg.data)}"

    @property
    def params(self):
        return {'Degree': self.deg}

    @property
    def hparams(self):
        return {"Kernel": "Polynomial"}

    def implicit(self, x):
        return (super()._implicit(x, self._idx_kernels) + 1) ** self.deg

    def explicit(self, x):
        # A primal representation is technically possible here (we will only consider the dual representation now).
        # raise rkm.PrimalError
        raise NotImplementedError
