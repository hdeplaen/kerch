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
        self._kwargs = kwargs
        self._type = type

        self._euclidean = None
        self._stiefel = None

        if len(euclidean_params) > 0:
            euclidean_switcher = {"sgd": lambda: torch.optim.SGD(euclidean_params, **self._kwargs),
                                "adam": lambda: torch.optim.Adam(stiefel_params, **self._kwargs)}
            self._euclidean = euclidean_switcher.get(type, "Incorrect optimizer type.")()

        if len(stiefel_params) > 0:
            stiefel_switcher = {"sgd": lambda: cayley_stiefel_optimizer.SGDG(stiefel_params, **self._kwargs),
                                "adam": lambda: cayley_stiefel_optimizer.AdamG(stiefel_params, **self._kwargs)}
            self._stiefel = stiefel_switcher.get(type, "Incorrect optimizer type.")()

    @property
    def type(self):
        return self._type

    @property
    def params(self):
        return self._kwargs

    def step(self, closure=None):
        if self._euclidean is not None: self._euclidean.step(closure)
        if self._stiefel is not None: self._stiefel.step(closure)

    def zero_grad(self):
        if self._euclidean is not None: self._euclidean.zero_grad()
        if self._stiefel is not None: self._stiefel.zero_grad()
