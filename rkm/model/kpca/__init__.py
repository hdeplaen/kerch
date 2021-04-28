"""
KPCA level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm
import rkm.model as mdl
import rkm.model.kpca.HardKPCA as HardKPCA
import rkm.model.kpca.SoftKPCA as SoftKPCA

import torch
from abc import ABCMeta


class KPCA(rkm.model.level.Level, metaclass=ABCMeta):
    @rkm.kwargs_decorator(
        {"centering": False})
    def __init__(self, **kwargs):
        """

        :param centering: True if input and kernel are centered (False by default).
        """
        new_kwargs = {"requires_bias": False}
        super(KPCA, self).__init__({**kwargs, **new_kwargs})
        self.__centering = kwargs["centering"]
        self.__generate_representation(**kwargs)

        assert not self.__centering, NotImplementedError  # True is not implemented.

    def __generate_representation(self, **kwargs):
        super().__generate_representation(**kwargs)

        # REGULARIZATION
        def primal_var(idx_kernels):
            C = self.__model["kernel"].cov()
            V = self.__model["linear"].weight
            return torch.trace(C) - torch.trace(V.t() @ C @ V)

        def dual_var(idx_kernels):
            K = self.__model["kernel"].corr(idx_kernels)
            H = self.__model["linear"].alpha[idx_kernels]
            return torch.trace(K) - torch.trace(H.t() @ K @ H)

        switcher_var = {"primal": lambda idx_kernels: primal_var(idx_kernels),
                        "dual": lambda idx_kernels: dual_var(idx_kernels)}
        self.__var = switcher_var.get(kwargs["representation"], mdl.RepresentationError)

    def loss(self, x=None, y=None, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self.all_kernels
        self.__var(idx_kernels)

    def solve(self, x, y=None):
        switcher = {'primal': lambda: self.primal(x),
                    'dual': lambda: self.dual(x)}

        return switcher.get(self.representation, mdl.RepresentationError)()

    def primal(self, x, y=None):
        C = self.__model["kernel"].cov()
        s, v = torch.lobpcg(C, k=self.__size_out)
        w = v @ torch.diag(s)

        return w.data, None

    def dual(self, x, y=None):
        K = self.__model["kernel"].corr()
        s, v = torch.lobpcg(K, k=self.__size_out)
        h = v @ torch.diag(s)

        return h.data, None

    def get_params(self):
        euclidean = self.__model['kernel'].parameters()
        stiefel = self.__model['linear'].parameters()
        return euclidean, stiefel

    @staticmethod
    def create(**kwargs):
        switcher = {"hard": lambda: HardKPCA.HardKPCA(**kwargs),
                    "soft": lambda: SoftKPCA.SoftKPCA(**kwargs)}
        func = switcher.get(kwargs["type"], "Invalid KPCA type (must be hard or soft).")
        return func()


