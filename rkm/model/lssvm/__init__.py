"""
LS-SVM abstract level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
import random
import numpy as np
from abc import ABCMeta, abstractmethod

import rkm
from rkm.model.level import Level
from rkm.model import RepresentationError

class LSSVM(Level, metaclass=ABCMeta):
    """
    Abstract LSSVM class.
    """

    @rkm.kwargs_decorator(
        {"gamma": 1})
    def __init__(self, **kwargs):
        """

        :param gamma: reconstruction / regularization trade-off.
        """
        new_kwargs = {"requires_bias": True}  # LS-SVM level becomes ridge regression if this is False.
        super(LSSVM, self).__init__({**kwargs, **new_kwargs})

        self.__gamma = kwargs["gamma"]
        self.__criterion = torch.nn.MSELoss(reduction="mean")
        self.__generate_representation(**kwargs)

    @property
    def gamma(self):
        return self.__gamma

    def __generate_representation(self, **kwargs):
        super().__generate_representation(**kwargs)

        # REGULARIZATION
        def primal_reg(idx_kernels):
            weight = self.__model['linear'].weight
            return (1 / len(idx_kernels)) * weight.t() @ weight

        def dual_reg(idx_kernels):
            alpha = self.__model["linear"].alpha[idx_kernels]
            K = self.__model["kernel"].matrix(idx_kernels)
            return (1 / len(idx_kernels)) * alpha.t() @ K @ alpha

        switcher_reg = {"primal": lambda idx_kernels: primal_reg(idx_kernels),
                        "dual": lambda idx_kernels: dual_reg(idx_kernels)}
        self.__reg = switcher_reg.get(kwargs["representation"], RepresentationError)

    def recon(self, x, y, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self.all_kernels
        x_tilde = self.forward(x, idx_kernels)
        return self.__criterion(x_tilde, y)

    def reg(self, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self.all_kernels
        return torch.trace(self.__reg(idx_kernels))

    def loss(self, x=None, y=None, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self.all_kernels
        recon = self.recon(idx_kernels, x, y)
        reg = self.reg(idx_kernels)
        return .5 * reg + .5 * self.__gamma * recon

    def solve(self, x, y=None):
        assert y is not None, "Tensor y is unspecified. This is not allowed for a LSSVM level."
        switcher = {'primal': lambda: self.primal(x, y),
                    'dual': lambda: self.dual(x, y)}

        return switcher.get(self.representation, RepresentationError)()

    def primal(self, x, y):
        assert y.size(1) == 1, "Not implemented for multi-dimensional output (as for now)."

        C, phi = self.__model['kernel'].cov(x)
        n = phi.size(1)
        I = torch.eye(n)
        P = torch.sum(phi, dim=0)
        S = torch.sum(y, dim=0)
        Y = phi.t() @ y

        A = torch.cat((torch.cat((C + (1 / self.__gamma) * I, P.t()), dim=1),
                       torch.cat((P, n), dim=1)), dim=0)
        B = torch.cat((Y, S), dim=0)

        sol = torch.solve(A, B)
        weight = sol[0:-1].data
        bias = sol[-1].data

        return weight, bias

    def dual(self, x, y):
        assert y.size(1) == 1, "Not implemented for multi-dimensional output (as for now)."
        n = x.size(0)

        K = self.__model["kernel"].matrix()
        I = torch.eye(n)
        N = torch.ones((n, 1))
        A = torch.cat((torch.cat((K + (1 / self.__gamma) * I, N), dim=1),
                       torch.cat((N.t(), torch.tensor(0.)), dim=1)), dim=0)
        B = torch.cat((y, torch.tensor(0.)), dim=0)

        sol = torch.solve(A, B)
        alpha = sol[0:-1].data
        beta = sol[-1].data

        return alpha, beta

    def get_params(self):
        euclidean = self.__model.parameters()
        stiefel = torch.nn.ParameterList()
        return euclidean, stiefel

    # @staticmethod
    # def create(**kwargs):
    #     switcher = {"hard": lambda: HardLSSVM.HardLSSVM(**kwargs),
    #                 "soft": lambda: SoftLSSVM.SoftLSSVM(**kwargs)}
    #     func = switcher.get(kwargs["type"], "Invalid LSSVM type (must be hard or soft).")
    #     return func()
