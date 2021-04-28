"""
Custom optimizer for RKMs

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from .cayley import cayley_stiefel_optimizer

import rkm

class Optimizer():
    @rkm.kwargs_decorator({"lr": 1e-3})
    def __init__(self, euclidean_params, stiefel_params, type="sgd", **kwargs):
        self.__kwargs = kwargs
        self.__type = type

        euclidean_switcher = {"sgd": lambda: torch.optim.SGD(euclidean_params, **self.__kwargs),
                              "adam": lambda: torch.optim.Adam(stiefel_params, **self.__kwargs)}

        stiefel_switcher = {"sgd": lambda: cayley_stiefel_optimizer.SGDG(stiefel_params, **self.__kwargs),
                              "adam": lambda: cayley_stiefel_optimizer.AdamG(stiefel_params, **self.__kwargs)}

        self.__euclidean = euclidean_switcher.get(type, "Incorrect optimizer type.")()
        self.__stiefel = stiefel_switcher.get(type, "Incorrect optimizer type.")()

    @property
    def type(self):
        return self.__type

    @property
    def params(self):
        return self.__kwargs

    def step(self, closure=None):
        self.__euclidean.step(closure)
        self.__stiefel.step(closure)

    def zero_grad(self):
        self.__euclidean.zero_grad()
        self.__stiefel.zero_grad()
