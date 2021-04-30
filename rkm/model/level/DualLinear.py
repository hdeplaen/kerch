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
        self._alpha = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.Tensor(kwargs["init_kernels"], kwargs["size_out"])),
            requires_grad=self._soft)
        self._bias = torch.nn.Parameter(torch.tensor(0.), requires_grad=self._soft and self._requires_bias)

    @property
    def alpha(self):
        return self._alpha

    @property
    def bias(self):
        return self._bias

    def set(self, a, b=None):
        self._alpha.data = a.data
        if b is not None: self._bias.data = b.data

    def forward(self, x, idx_sv):
        return x @ self._alpha[idx_sv] + self._bias.expand(x.shape[0])

    def project(self):
        self._alpha.data -= torch.mean(self._alpha.data)

    def merge(self, idxs):
        self._alpha[idxs[:, 0]] += self._alpha[idxs[:, 1]]
        self._alpha.gather(dim=0, index=idxs[:, 1], out=self._alpha)

    def reduce(self, idxs):
        self._alpha.gather(dim=0, index=idxs, out=self._alpha)

    def reduce_idxs(self, **kwargs):
        return torch.nonzero(torch.abs(self._alpha) < kwargs["rtol"], as_tuple=True)
