"""
Custom optimizer for RKMs

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from .stiefel import stiefel_optimizer

from .. import utils
from .._module import _Module


class Optimizer():
    @staticmethod
    def param_state(params):
        out = torch.nn.ParameterList([])
        for p in params:
            if p.requires_grad is True:
                out.append(p)
        return out

    @utils.kwargs_decorator({"stiefel_lr": 1.e-4,
                             "euclidean_lr": 1.e-4,
                             "slow_lr": 1.e-4
                             })
    def __init__(self, mdl: _Module, type="sgd", **kwargs):
        self._kwargs = kwargs
        self._type = type

        euclidean_params = Optimizer.param_state(mdl.manifold_parameters(type='euclidean'))
        slow_params = Optimizer.param_state(mdl.manifold_parameters(type='slow'))
        stiefel_params = Optimizer.param_state(mdl.manifold_parameters(type='stiefel'))

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
