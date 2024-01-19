"""
Custom optimizer for RKMs

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""
from __future__ import annotations

import torch
from .stiefel import stiefel_optimizer

from .. import utils
from kerch.module.Module import Module


class Optimizer():
    @staticmethod
    def param_state(params):
        out = torch.nn.ParameterList([])
        for p in params:
            if p.requires_grad is True:
                out.append(p)
        return out

    @utils.kwargs_decorator({"stiefel_lr": 1.e-3,
                             "euclidean_lr": 1.e-3,
                             "slow_lr": 1.e-4
                             })
    def __init__(self, module: Module, type="sgd", **kwargs):
        self._kwargs = kwargs
        self._type = type
        self._module = module

        euclidean_params = Optimizer.param_state(module.manifold_parameters(type='euclidean'))
        stiefel_params = Optimizer.param_state(module.manifold_parameters(type='stiefel'))
        slow_params = Optimizer.param_state(module.manifold_parameters(type='slow'))

        self._dict = []
        self._opt = None
        self._requires_grad = False

        if len(euclidean_params) > 0:
            dict_euclidean = {'params': euclidean_params, 'stiefel': False, 'lr': kwargs['euclidean_lr']}
            self._dict.append({**kwargs, **dict_euclidean})
            self._requires_grad = True

        if len(slow_params) > 0:
            dict_slow = {'params': slow_params, 'stiefel': False, 'lr': kwargs['slow_lr']}
            dict_slow = {**kwargs, **dict_slow}
            self._dict.append(dict_slow)
            self._requires_grad = True

        if len(stiefel_params) > 0:
            dict_stiefel = {'params': stiefel_params, 'stiefel': True, 'lr': kwargs['stiefel_lr']}
            self._dict.append({**kwargs, **dict_stiefel})
            self._requires_grad = True

        if self._dict:
            opt_switcher = {"sgd": stiefel_optimizer.SGDG,
                            "adam": stiefel_optimizer.AdamG}
            self._opt = opt_switcher.get(type, "Incorrect optimizer name.")(self._dict)

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def type(self):
        return self._type

    @property
    def hparams(self):
        return {'[Optimizer] ' + key: val for key, val in self._kwargs.items()}

    @property
    def module(self) -> Module:
        return self._module

    def reduce(self, rate):
        if self._opt is not None:
            for params in self._opt.param_groups:
                params['lr'] *= rate

    def step(self, closure=None) -> float | None:
        if self._opt is not None: loss = self._opt.step(closure)
        else: loss = closure()
        return loss

    def zero_grad(self):
        if self._opt is not None: self._opt.zero_grad()
