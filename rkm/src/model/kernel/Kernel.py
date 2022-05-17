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

import rkm.src
import rkm.src.utils.type as type_utils


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
    @rkm.src.kwargs_decorator({"size_in": 1,
                                "kernels_trainable": False,
                                "centering": False,
                                "sample": None,
                                "init_kernels": None})
    def __init__(self, **kwargs):
        """
        Creates a mother class for a kernel. This class is useless as such and must be inherited.

        :param size_in: dimension of the input (default 1)
        :param init_kernels: number of kernel
        :param kernels_trainable: True if support vectors / kernel are trainable (default False)
        """
        super(Kernel, self).__init__()

        self.kernels_trainable = kwargs["kernels_trainable"]
        self._centering = kwargs["centering"]

        input_sample = kwargs["sample"]
        input_sample = type_utils.castf(input_sample)
        if input_sample is not None:
            self.init_kernels, self.size_in = input_sample.shape
            self.kernels = nn.Parameter(input_sample.data,
                requires_grad=self.kernels_trainable)
        elif kwargs["init_kernels"] is not None:
            self.size_in = kwargs["size_in"]
            self.init_kernels = kwargs["init_kernels"]
            self.kernels = nn.Parameter(
                nn.init.orthogonal_(torch.empty((self.init_kernels, self.size_in), dtype=rkm.ftype)),
                requires_grad=self.kernels_trainable)
        else:
            raise NameError("Nor the dimensions, nor sample data has been provided.")


        self._K = None
        self._K_mean = None
        self._K_mean_tot = None
        self._phi = None
        self._C = None
        self._phi_mean = None
        self._idx_kernels = None
        self.reset()

    @abstractmethod
    def __str__(self):
        pass

    @property
    def params(self):
        return {}

    def _empty_cache(self):
        self._K = None
        self._K_mean = None
        self._K_mean_tot = None
        self._phi = None
        self._C = None
        self._phi_mean = None

    # @property
    def num_kernels(self):
        return self.kernels.shape[0]

    # @property
    def num_idx(self):
        return len(self._idx_kernels)

    @property
    def hparams(self):
        return {"Trainable Kernels": self.kernels_trainable,
                "Centering": self._centering}

    def all_kernels(self):
        return range(self.num_kernels())

    def reset(self, idx_kernels=None):
        self._empty_cache()
        if idx_kernels is None:
            self._idx_kernels = self.all_kernels()
        else:
            self._idx_kernels = idx_kernels

    def kernels_init(self, x):
        x = type_utils.castf(x)
        self._empty_cache()
        assert x is not None, "Kernels updated with None values."
        self.kernels.data = x.data

    def update_kernels(self, x, idx_kernels):
        x = type_utils.castf(x)
        self._empty_cache()
        assert x is not None, "Kernels updated with None values."
        if not self.kernels_trainable:
            self.kernels.data[idx_kernels, :] = x.data

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

###################################################################################################
################################### MATHS ARE HERE ################################################
###################################################################################################

    @abstractmethod
    def _implicit(self, x_oos=None, x_sample=None):
        # implicit without centering
        # explicit without centering
        if x_oos is None:
            x_oos = self.kernels[self._idx_kernels, :]
        if x_sample is None:
            x_sample = self.kernels[self._idx_kernels, :]
        return x_oos, x_sample

    @abstractmethod
    def _explicit(self, x=None):
        # explicit without centering
        if x is None:
            x =  self.kernels[self._idx_kernels, :]
        return x

    def _dmatrix(self, implicit=False):
        """
        Computes the dual matrix, also known as the kernel matrix.
        Its size is len(idx_kernels) * len(idx_kernels).

        :param idx_kernels: Index of the support vectors used to compute the kernel matrix. If nothing is provided, the kernel uses all_kernels of them.
        :return: Kernel matrix.
        """
        if self._K is None and not implicit:
            if self._idx_kernels is None:
                self.reset()

            # self._k = self._implicit(self.kernels.gather(0, self._idx_kernels))
            self._K = self._implicit()

            if self._centering:
                n = self.num_kernels()
                self._K_mean = torch.mean(self._K, dim=0)
                self._K_mean_tot = torch.mean(self._K, dim=(0, 1))
                self._K = self._K - self._K_mean.expand(n, n) \
                          - self._K_mean.expand(n, n).t() \
                          + self._K_mean_tot

        elif self._K is None and implicit:
            phi = self.phi()
            self._K = phi @ phi.T

        return self._K

    def _pmatrix(self):
        """
        Computes the primal matrix, i.e. correlation between the different outputs.
        Its size is output * output.
        """
        if self._C is None:
            if self._idx_kernels is None:
                self.reset()

            self._phi = self._explicit()

            if self._centering:
                self._phi_mean = torch.mean(self._phi, dim=0)
                self._phi = self._phi - self._phi_mean
            self._C = self._phi.T @ self._phi
        return self._C

    def phi(self, x=None):
        # if x is None, phi(x) for x in the sample is returned.
        x = type_utils.castf(x)

        self._pmatrix()

        phi = self._explicit(x)
        if self._centering:
            phi = phi - self._phi_mean

        return phi

    def k(self, x_oos=None, x_sample=None, implicit=False):
        x_oos = type_utils.castf(x_oos)
        x_sample = type_utils.castf(x_sample)

        if x_oos is None and x_sample is None:
            return self._dmatrix(implicit=implicit)

        if x_sample is not None and not implicit and self._centering:
            raise NameError(
                "Impossible to compute centered out-of-sample to out-of-sample kernels for implicit-defined kernels as the centering statistic is only defined on the sample.")

        if implicit:
            self._pmatrix()
            phi_sample = self.phi(x_sample)
            phi_oos = self.phi(x_oos)
            Ky = phi_oos @ phi_sample.T
        else:
            self._dmatrix()
            Ky = self._implicit(x_oos, x_sample)
            if self._centering:
                Ky = Ky - torch.mean(Ky, dim=1, keepdim=True).expand(-1, Ky.shape[1])
                Ky = Ky - self._K_mean
                Ky = Ky + self._K_mean_tot

        return Ky

    def forward(self, x, representation="dual", idx_kernels=None):
        if idx_kernels is not None: self.reset(idx_kernels)

        def primal(x):
            return self.phi(x)

        def dual(x):
            return self.k(x)

        switcher = {"primal": primal,
                    "dual": dual}

        fun = switcher.get(representation, rkm.src.model.RepresentationError)
        return fun(x)

    @property
    def K(self):
        return self._dmatrix()

    @property
    def C(self):
        return self._pmatrix()[0]

    @property
    def phi_sample(self):
        return self._pmatrix()[1]