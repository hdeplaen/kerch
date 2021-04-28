"""
Dual linear class for a RKM level.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

import rkm.model.level.Linear as Linear

class DualLinear(Linear.Linear):
    def __init__(self, **kwargs):
        super(DualLinear, self).__init__(**kwargs)
        self.__alpha = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.Tensor(kwargs["num_kernels"], kwargs["size_out"])),
            requires_grad=self.__soft)
        self.__bias = torch.nn.Parameter(torch.tensor(0.), requires_grad=self.__soft and self.__requires_bias)

    @property
    def alpha(self):
        return self.__alpha

    @property
    def bias(self):
        return self.__bias

    def set(self, a, b=None):
        self.__alpha.data = a.data
        if b is not None: self.__bias.data = b.data

    def forward(self, x, idx_sv):
        return x @ self.__alpha[idx_sv] + self.__bias.expand(x.shape[0])

    def project(self):
        self.__alpha.data -= torch.mean(self.__alpha.data)

    def merge(self, idxs):
        self.__alpha[idxs[:, 0]] += self.__alpha[idxs[:, 1]]
        self.__alpha.gather(dim=0, index=idxs[:, 1], out=self.__alpha)

    def reduce(self, idxs):
        self.__alpha.gather(dim=0, index=idxs, out=self.__alpha)

    def reduce_idxs(self, **kwargs):
        return torch.nonzero(torch.abs(self.__alpha) < kwargs["rtol"], as_tuple=True)
