"""
File containing the abstract kernel classes.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

import rkm
import rkm.model as mdl


class Kernel(nn.Module, metaclass=ABCMeta):
    """
    Abstract kernel mother class
    k(x,y) = f(x,y)
    """

    # @property
    # def kernels(self, idx_kernels=None):
    #     if idx_kernels is None: idx_kernels = self.all_kernels
    #     return self.kernels.gather(dim=1, index=idx_kernels)

    @abstractmethod
    @rkm.kwargs_decorator({"size_in": 1,
                           "init_kernels": 1,
                           "kernels_trainable": False,
                           "centering": False})
    def __init__(self, **kwargs):
        """
        Creates a mother class for a kernel. This class is useless as such and must be inherited.

        :param size_in: dimension of the input (default 1)
        :param init_kernels: number of kernel
        :param kernels_trainable: True if support vectors / kernel are trainable (default False)
        """
        super(Kernel, self).__init__()
        self.size_in = kwargs["size_in"]
        self.init_kernels = kwargs["init_kernels"]
        self.kernels_trainable = kwargs["kernels_trainable"]
        self._centering = kwargs["centering"]

        self.kernels = nn.Parameter(nn.init.orthogonal_(torch.empty((self.init_kernels, self.size_in))),
                                    requires_grad=self.kernels_trainable)

        self._K = None
        self._K_mean = None
        self._K_mean_tot = None
        self._phi = None
        self._C = None
        self._C_mean = None
        self._idx_kernels = None
        self.reset()

    @abstractmethod
    def __str__(self):
        pass

    @property
    def params(self):
        return {}

    #@property
    def num_kernels(self):
        return self.kernels.shape[0]

    @property
    def hparams(self):
        return {"Trainable Kernels": self.kernels_trainable,
                "Centering": self._centering}

    def all_kernels(self):
        return range(self.num_kernels())

    def forward(self, x, representation, idx_kernels=None):
        if idx_kernels is not None: self.reset(idx_kernels)

        def primal(x):
            self.pmatrix()

            phi = self._explicit(x)
            if self._centering:
                phi = phi - self._C_mean

            return phi

        def dual(x):
            self.dmatrix()

            Ky = self._implicit(x)
            if self._centering:
                Ky = Ky - self._K_mean \
                        - torch.mean(Ky, dim=1) \
                        + self._K_mean_tot

            return Ky

        switcher = {"primal": primal,
                    "dual": dual}

        fun = switcher.get(representation, mdl.RepresentationError)
        return fun(x)

    @abstractmethod
    def _implicit(self, x):
        # implicit without centering
        pass

    @abstractmethod
    def _explicit(self, x):
        # explicit without centering
        pass

    def reset(self, idx_kernels=None):
        if idx_kernels is None:
            self._idx_kernels = self.all_kernels()
        else:
            self._idx_kernels = idx_kernels

        self._K = None
        self._K_mean = None
        self._K_mean_tot = None
        self._phi = None
        self._C = None
        self._C_mean = None

    def update_kernels(self, x):
        assert x is not None, "Kernels updated with None values."
        if not self.kernels_trainable:
            self.kernels.data = x.data

    def merge_idxs(self, **kwargs):
        raise NotImplementedError
        self.dmatrix()
        return torch.nonzero(torch.triu(self.dmatrix()) > (1 - kwargs["mtol"]), as_tuple=False)

    def merge(self, idxs):
        raise NotImplementedError
        # suppress added up kernel
        self.kernels = (self.kernels.gather(dim=0, index=idxs[:, 1]) +
                        self.kernels.gather(dim=0, index=idxs[:, 0])) / 2

        self.dmatrix()
        # suppress added up kernel entries in the kernel matrix
        self._K.gather(dim=0, index=idxs[:, 1], out=self._K)
        self._K.gather(dim=1, index=idxs[:, 1], out=self._K)

    def reduce(self, idxs):
        raise NotImplementedError
        self.kernels.gather(dim=0, index=idxs, out=self.kernels)

    def dmatrix(self):
        """
        Computes the dual matrix, also known as the kernel matrix.
        Its size is len(idx_kernels) * len(idx_kernels).

        :param idx_kernels: Index of the support vectors used to compute the kernel matrix. If nothing is provided, the kernel uses all of them.
        :return: Kernel matrix.
        """
        if self._K is None:
            if self._idx_kernels is None:
                self.reset()

            self._K = self._implicit(self.kernels[self._idx_kernels, :])

            if self._centering:
                n = self.num_kernels()
                self._K_mean = torch.mean(self._K, dim=0)
                self._K_mean_tot = torch.mean(self._K, dim=(0, 1))
                self._K = self._K - self._K_mean.expand(n, n) \
                          - self._K_mean.t().expand(n, n) \
                          - self._K_mean_tot

        return self._K

    def pmatrix(self):
        """
        Computes the primal matrix, i.e. correlation between the different outputs.
        Its size is output * output.
        """
        if self._C is None:
            if self._idx_kernels is None:
                self.reset()

            k = self.kernels[self._idx_kernels, :]
            self._phi = self._explicit(k)

            if self._centering:
                self._C_mean = torch.mean(self._phi, dim=0)
                self._phi = self._phi - self._C_mean
            self._C = self._phi.t() @ self._phi
        return self._C, self._phi
