"""
File containing the sigmoid kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm
import rkm.model as mdl

import torch

class SigmoidKernel(mdl.kernel.LinearKernel.LinearKernel):
    """
    Sigmoid kernel class
    k(x,y) = tanh( a * < x,y > + b ).
    """

    @rkm.kwargs_decorator(
        {"a": 1., "b": 0., "params_trainable": False})
    def __init__(self, **kwargs):
        """
        :param a: value for a (initial if trainable)
        :param b: value for b (initial if trainable)
        :param params_trainable: trainable parameters a and b in sigmoid kernel
        """
        super(SigmoidKernel, self).__init__(**kwargs)
        self.__a = torch.nn.Parameter(
            torch.tensor(kwargs["a"]).unsqueeze(dim=0), requires_grad=kwargs["params_trainable"])
        self.__b = torch.nn.Parameter(
            torch.tensor(kwargs["b"]).unsqueeze(dim=0), requires_grad=kwargs["params_trainable"])
        self.__linear = lambda x: self.__a * x + self.__b

    def __str__(self):
        return "sigmoid kernel"

    def implicit(self, x, idx_kernels=None):
        x = super().implicit(x, idx_kernels)
        x = self.__linear(x)
        return torch.sigmoid(x)

    def explicit(self, x, idx_kernels=None):
        raise mdl.PrimalError
