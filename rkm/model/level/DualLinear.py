"""
Dual linear class for a RKM level.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

import rkm
import rkm.model.level.Linear as Linear


class DualLinear(Linear.Linear):
    def __init__(self, **kwargs):
        super(DualLinear, self).__init__(**kwargs)
        self._alpha = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty((kwargs["size_out"], kwargs["init_kernels"]),
                                                  dtype=rkm.ftype)),
            requires_grad=self._soft)
        self._bias = torch.nn.Parameter(torch.zeros([kwargs["size_out"]], dtype=rkm.ftype),
                                        requires_grad=self._soft and self._requires_bias)

    @property
    def alpha(self):
        return self._alpha.t()

    @property
    def bias(self):
        return self._bias

    def set(self, a, b=None):
        self._alpha.data = a.data.t() #not doing anything
        if b is not None: self.bias.data = b.data

    def forward(self, x, idx_sv):
        x_tilde = x @ self.alpha[idx_sv] + self.bias.expand([x.shape[0], 1])
        return x_tilde.squeeze()

    def project(self):
        self.alpha.data -= torch.mean(self.alpha.data)

    def merge(self, idxs):
        self.alpha[idxs[:, 0]] += self.alpha[idxs[:, 1]]
        self.alpha.gather(dim=0, index=idxs[:, 1], out=self._alpha)

    def reduce(self, idxs):
        self.alpha.gather(dim=0, index=idxs, out=self._alpha)

    def reduce_idxs(self, **kwargs):
        return torch.nonzero(torch.abs(self.alpha) < kwargs["rtol"], as_tuple=True)
