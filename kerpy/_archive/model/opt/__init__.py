"""
Custom optimizer for RKMs

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from .cayley import cayley_stiefel_optimizer

import kerpy

class Optimizer():
    @staticmethod
    def param_state(params):
        out = torch.nn.ParameterList([])
        for p in params:
            if p.requires_grad is True:
                out.append(p)
        return out

    @kerpy.kwargs_decorator({"lr": 5e-3, "kernel_rate": 1.})
    def __init__(self, euclidean_params, slow_params, stiefel_params, type="sgd", **kwargs):
        self._kwargs = kwargs
        self._type = type

        euclidean_params = Optimizer.param_state(euclidean_params)
        slow_params = Optimizer.param_state(slow_params)
        stiefel_params = Optimizer.param_state(stiefel_params)

        self._dict = []
        self._opt = None
        self._requires_grad = False

        if len(euclidean_params) > 0:
            dict_euclidean = {'params': euclidean_params, 'stiefel': False}
            self._dict.append({**dict_euclidean, **kwargs})
            self._requires_grad = True

        if len(slow_params) > 0:
            dict_slow = {'params': slow_params, 'stiefel': False, 'lr': kwargs['lr'] / 1}
            dict_slow = {**dict_slow, **kwargs}
            dict_slow['lr'] = dict_slow['lr'] * self._kwargs["kernel_rate"]
            self._dict.append(dict_slow)
            self._requires_grad = True

        if len(stiefel_params) > 0:
            dict_stiefel = {'params': stiefel_params, 'stiefel': True}
            self._dict.append({**dict_stiefel, **kwargs})
            self._requires_grad = True

        if self._dict:
            opt_switcher = {"sgd": cayley_stiefel_optimizer.SGDG,
                            "adam": cayley_stiefel_optimizer.AdamG}
            self._opt = opt_switcher.get(type, "Incorrect optimizer type.")(self._dict)

    @property
    def requires_grad(self):
        return self._requires_grad


    @property
    def type(self):
        return self._type

    @property
    def hparams(self):
        return {'type': self._type, **self._kwargs}

    @property
    def params(self):
        return self._kwargs

    def reduce(self, rate):
        if self._opt is not None:
            for params in self._opt.param_groups:
                params['lr'] /= rate

    def step(self, closure=None):
        if self._opt is not None: self._opt.step(closure)
        else: closure()

    def zero_grad(self):
        if self._opt is not None: self._opt.zero_grad()
