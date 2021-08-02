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
        {"centering": False})
    def __init__(self, **kwargs):
        """

        :param centering: True if input and kernel are centered (False by default).
        """
        add_kwargs = {"requires_bias": False}
        new_kwargs = {**kwargs, **add_kwargs}
        super(KPCA, self).__init__(**new_kwargs)
        self._centering = kwargs["centering"]
        self._generate_representation(**kwargs)

        self._last_var = torch.tensor(0.)

        assert not self._centering, NotImplementedError  # True is not implemented.

    @property
    def hparams(self):
        return {"Type": "KPCA",
                "Centering": self._centering,
                **super(KPCA, self).hparams}

    @property
    def last_var(self):
        return self._last_var.data

    def _generate_representation(self, **kwargs):
        # REGULARIZATION
        def primal_var(idx_kernels):
            C, _ = self._model["kernel"].pmatrix(None, idx_kernels)
            V = self._model["linear"].weight
            return torch.trace(C) - torch.trace(V.t() @ C @ V)

        def dual_var(idx_kernels):
            K = self._model["kernel"].dmatrix(idx_kernels)
            H = self._model["linear"].alpha[idx_kernels,:]
            # return torch.trace(K) - torch.trace(H @ H.t() @ K)
            return (torch.trace(K) - torch.trace(H @ H.t() @ K)) / torch.sum(K, (0,1))

        switcher_var = {"primal": lambda idx_kernels: primal_var(idx_kernels),
                        "dual": lambda idx_kernels: dual_var(idx_kernels)}
        self._var = switcher_var.get(kwargs["representation"], RepresentationError)

    def loss(self, x=None, y=None, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self._all_kernels
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
            [p for n, p in  self._model['kernel'].named_parameters() if p.requires_grad and n not in slow_names])
        slow = torch.nn.ParameterList(
            [p for n, p in  self._model['kernel'].named_parameters() if p.requires_grad and n in slow_names])
        stiefel = self._model['linear'].parameters()
        return euclidean, slow, stiefel

    # @staticmethod
    # def create(**kwargs):
    #     switcher = {"hard": lambda: HardKPCA.HardKPCA(**kwargs),
    #                 "soft": lambda: SoftKPCA.SoftKPCA(**kwargs)}
    #     func = switcher.get(kwargs["type"], "Invalid KPCA type (must be hard or soft).")
    #     return func()


