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

    @property
    def kernels(self, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self.all_kernels
        return self.__kernels.gather(dim=1, index=idx_kernels)

    @kernels.setter
    def kernels(self, val):
        self.__kernels = val

    @abstractmethod
    @rkm.kwargs_decorator({"size_in": 1, "init_kernels": 1, "kernels_trainable": False})
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
        self.K = None
        self.C = None

        self.kernels = nn.Parameter(nn.init.orthogonal_(torch.empty((self.init_kernels, self.size_in))),
                                    requires_grad=self.kernels_trainable)

    @abstractmethod
    def __str__(self):
        pass

    def forward(self, x, representation, idx_kernels=None):
        switcher = {"primal": lambda: self.explicit(x, idx_kernels),
                    "dual": lambda: self.implicit(x, idx_kernels)}

        return switcher.get(representation, mdl.RepresentationError)()

    @abstractmethod
    def implicit(self, x, idx_kernels=None):
        pass

    @abstractmethod
    def explicit(self, x, idx_kernels=None):
        pass

    def update(self, x):
        if not self.kernels_trainable:
            self.kernels.data = x.data

    def merge_idxs(self, **kwargs):
        assert self.K is not None, "Kernel matrix must be computed first to perform aggregation."
        return torch.nonzero(torch.triu(self.K) > (1 - kwargs["mtol"]), as_tuple=False)

    def merge(self, idxs):
        # suppress added up kernel
        self.kernels = (self.kernels.gather(dim=0, index=idxs[:, 1]) +
                        self.kernels.gather(dim=0, index=idxs[:, 0])) / 2

        # suppress added up kernel entries in the kernel matrix
        if self.K is not None:
            self.K.gather(dim=0, index=idxs[:, 1], out=self.K)
            self.K.gather(dim=1, index=idxs[:, 1], out=self.K)

    def reduce(self, idxs):
        self.kernels.gather(dim=0, index=idxs, out=self.kernels)

    def corr(self, idx_kernels=None):
        """
        Computes the correlation matrix, also known as the kernel matrix.

        :param idx_kernels: Index of the support vectors used to compute the kernel matrix. If nothing is provided, the kernel uses all of them.
        :return: Kernel matrix.
        """
        self.K = self.implicit(self.kernels(idx_kernels), idx_kernels)
        return self.K

    def cov(self, x=None, idx_kernels=None):
        k = self.kernels(idx_kernels)
        k = self.explicit(k)

        if x is None: phi = k
        else: phi = self.explicit(x)

        self.C = phi.t() @ k
        return self.C, phi

    @property
    def all_kernels(self):
        return range(self.kernels.size(0))
