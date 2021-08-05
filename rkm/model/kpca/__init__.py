"""
KPCA level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm
from rkm.model.level import Level
from rkm.model import RepresentationError
import torch
from abc import ABCMeta


class KPCA(Level, metaclass=ABCMeta):
    @rkm.kwargs_decorator(
        {"centering": True,
         "requires_bias": False})
    def __init__(self, **kwargs):
        """

        :param centering: True if input and kernel are centered (False by default).
        """
        super(KPCA, self).__init__(**kwargs)
        self._generate_representation(**kwargs)

        self._last_var = torch.tensor(0.)

    @property
    def hparams(self):
        return {"Type": "KPCA",
                "Centering": self._centering,
                **super(KPCA, self).hparams}

    @property
    def last_loss(self):
        return self._last_var.data

    def _generate_representation(self, **kwargs):
        # REGULARIZATION
        def primal_var(idx_kernels):
            C, _ = self.kernel.pmatrix(None, idx_kernels)
            if self._centering: C = self._center(C, idx_kernels)
            V = self.linear.weight
            return torch.trace(C) - torch.trace(V.t() @ C @ V)

        def dual_var(idx_kernels):
            K = self.kernel.dmatrix(idx_kernels)
            if self._centering: K = self._center(K, idx_kernels)
            H = self.linear.alpha[idx_kernels, :]
            # return torch.trace(K) - torch.trace(H @ H.t() @ K)
            return (torch.trace(K) - torch.trace(H @ H.t() @ K)) / torch.sum(K, (0, 1))

        switcher_var = {"primal": lambda idx_kernels: primal_var(idx_kernels),
                        "dual": lambda idx_kernels: dual_var(idx_kernels)}
        self._var = switcher_var.get(kwargs["representation"], RepresentationError)

    def loss(self, x=None, y=None):
        idx_kernels = self._idxk.idx_kernels
        var = self._var(idx_kernels)
        self._last_var = var.data
        return var, None

    def solve(self, x, y=None):
        switcher = {'primal': lambda: self.primal(x),
                    'dual': lambda: self.dual(x)}

        return switcher.get(self.representation, RepresentationError)()

    def primal(self, x, y=None):
        C = self.kernel.pmatrix()
        s, v = torch.lobpcg(C, k=self._size_out)
        w = v @ torch.diag(s)

        return w.data, None

    def dual(self, x, y=None):
        K = self.kernel.dmatrix()
        _, v = torch.lobpcg(K, k=self._size_out)
        h = v
        # U, s, _ = torch.svd(K)
        # h = U[:, self._size_out]

        return h.data, None

    def get_params(self, slow_names=None):
        euclidean = torch.nn.ParameterList(
            [p for n, p in self._model['kernel'].named_parameters() if p.requires_grad and n not in slow_names])
        slow = torch.nn.ParameterList(
            [p for n, p in self._model['kernel'].named_parameters() if p.requires_grad and n in slow_names])
        stiefel = self._model['linear'].parameters()
        return euclidean, slow, stiefel
