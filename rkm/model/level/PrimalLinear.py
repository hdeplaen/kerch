"""
Primal linear class for a RKM level.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm.model as mdl
import rkm.model.level.Linear as Linear

import torch


class PrimalLinear(Linear.Linear):
    def __init__(self, **kwargs):
        super(PrimalLinear, self).__init__(**kwargs)
        self.__weight = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.Tensor(kwargs["size_in"], kwargs["size_out"])),
            requires_grad=self._soft)
        self.__bias = torch.nn.Parameter(torch.tensor(0.), requires_grad=self._soft and self._requires_bias)

    @property
    def weight(self):
        return self.__weight

    @property
    def bias(self):
        return self.__bias

    def set(self, a, b=None):
        self.__weight.data = a.data
        if b is not None: self.__bias.data = b.data

    def forward(self, x, idx_sv):
        return x @ self.__weight + self.__bias.expand(x.shape[0])

    def project(self):
        pass

    def merge(self, idxs):
        raise mdl.DualError

    def reduce(self, idxs):
        raise mdl.DualError

    def reduce_idxs(self, **kwargs):
        raise mdl.DualError
