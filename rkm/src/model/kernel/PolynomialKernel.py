"""
File containing the polynomial kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm.src
import rkm.src.model.kernel.Kernel as Kernel

import torch

class PolynomialKernel(Kernel.Kernel):
    """
    Polynomial kernel class
    k(x,y) = ( < x,y > + 1 ) ** d.
    """

    @rkm.src.kwargs_decorator(
        {"deg": 2., "deg_trainable": False})
    def __init__(self, **kwargs):
        """
        :param deg: degree of the kernel (default 1)
        :param deg_trainable: True if the degree can be trained (default False)
        """
        super(PolynomialKernel, self).__init__(**kwargs)

        self.deg_trainable = kwargs["deg_trainable"]
        self.deg = torch.nn.Parameter(torch.tensor(kwargs["deg"]), requires_grad=self.deg_trainable)

    def __str__(self):
        return f"polynomial kernel of order {int(self.deg.data)}"

    @property
    def params(self):
        return {'Degree': self.deg}

    @property
    def hparams(self):
        return {"Kernel": "Polynomial"}

    def _implicit(self, x_oos=None, x_sample=None):
        return (super(PolynomialKernel, self)._implicit(x_oos, x_sample) + 1) ** self.deg

    def _explicit(self, x=None):
        # A primal representation is technically possible here (we will only consider the dual representation now).
        # raise rkm.PrimalError

        # if self.deg == 1.:
        #     return super(PolynomialKernel, self)._explicit(x)
        # elif self.deg == 2.:
        #     return 0
        # else:
        #     pass

        raise NotImplementedError
