"""
Primal linear class for a RKM Level.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import kerch._archive as mdl
import kerch._archive.model.level.Linear as Linear

import torch


class PrimalLinear(Linear.Linear):
    def __init__(self, **kwargs):
        super(PrimalLinear, self).__init__(**kwargs)
        self._weight = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.Tensor(kwargs["size_out"], kwargs["size_in"])),
            requires_grad=self._soft)
        self._bias = torch.nn.Parameter(torch.tensor([0.]), requires_grad=self._soft and self._requires_bias)

    @property
    def weight(self):
        return self._weight.t()

    @weight.setter
    def weight(self, val):
        self._weight.data[:,:] = val.data.t()

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, val):
        self._bias.data[:] = val.data

    def set(self, a, b=None):
        self.weight = a
        if b is not None: self.bias = b

    def forward(self, x, idx_sv):
        return x @ self.weight + self.bias.expand((x.shape[0], 1))

    def project(self):
        pass

    def merge(self, idxs):
        raise mdl.DualError

    def reduce(self, idxs):
        raise mdl.DualError

    def reduce_idxs(self, **kwargs):
        raise mdl.DualError
