"""
Abstract RKM level class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm
import rkm.model as mdl
import rkm.model.level.PrimalLinear as PrimalLinear
import rkm.model.level.DualLinear as DualLinear

import torch
from abc import ABCMeta, abstractmethod
import random
import numpy as np


class Level(torch.nn.Module, metaclass=ABCMeta):
    @rkm.kwargs_decorator(
        {"size_in": 1, "size_out": 1, "eta": 1, "representation": "dual", "init_kernels": 1,
         "type": "hard", "live_update": True})
    def __init__(self, **kwargs):
        """

        :param representation: "primal" or "dual" representation (default "dual").
        :param init_kernels: number of suppor vectors / kernel to be instantiated.
        :param type: Type of level ("hard" or "soft").
        :param live_update: Live update of the value of the kernel (default True).
        """
        super(Level, self).__init__()

        self.__size_in = kwargs["size_in"]
        self.__size_out = kwargs["size_out"]
        self.__eta = kwargs["eta"]

        self.__init_kernels = kwargs["init_kernels"]
        self.__num_kernels = kwargs["init_kernels"]
        self.__live_update = kwargs["live_update"]
        self.__representation = kwargs["representation"]
        self.__model = torch.nn.ModuleDict({})

        self.__generate_representation(**kwargs)

    @abstractmethod
    def __str__(self):
        pass

    def __generate_representation(self, **kwargs):
        # MODEL
        switcher = {"primal": lambda: PrimalLinear.PrimalLinear(**kwargs),
                    "dual": lambda: DualLinear.DualLinear(**kwargs)}

        self.__model = torch.nn.ModuleDict({
            "kernel": mdl.kernel.Kernel.create(**kwargs),
            "linear": switcher.get(kwargs["representation"], mdl.RepresentationError)()
        })

    def forward(self, x, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self.__all_kernels
        if self.__live_update: self.__model["kernel"].update(x)
        x = self.__model["kernel"](x, self.__representation, idx_kernels)
        x = self.__model["linear"](x, idx_kernels)
        return x

    @abstractmethod
    def loss(self, x, y=None):
        pass

    @property
    def eta(self):
        return self.__eta

    @property
    def num_kernels(self):
        return self.__num_kernels

    @property
    def representation(self) -> str:
        return self.__representation

    @property
    def linear(self):
        return self.__model["linear"]

    @property
    def kernel(self):
        return self.__model["kernel"]

    @abstractmethod
    def before_step(self, x=None, y=None):
        pass

    @abstractmethod
    def after_step(self, x=None, y=None):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def solve(self, x, y):
        pass

    @abstractmethod
    def primal(self, x, y):
        pass

    @abstractmethod
    def dual(self, x, y):
        pass

    def kernels_init(self, x=None):
        self.__model["kernel"].kernels_init(x)
        self.kernels_initialized = True

    @property
    def __all_kernels(self):
        return range(self.__num_kernels)

    @property
    def __stoch_kernels(self):
        if self.__stochastic < 1.:
            idx = random.choices(self.all_kernels, k=np.maximum(
                torch.round(self.__stochastic * self.init_kernels), self.num_kernels))
        else:
            idx = self.all_kernels
        return idx

    @rkm.kwargs_decorator({"mtol": 1.0e-2, "rtol": 1.0e-4})
    def aggregate(self, **kwargs):
        """
        Aggregates the kernel.

        :param mtol: Merges the kernel if the value is not 0.
        :param rtol: Reduces the kernel if the value is not 0.
        """
        assert self.__size_out == 1, NotImplementedError

        def __merge(idxs):
            self.__model["kernel"].merge(idxs)
            self.__model["linear"].merge(idxs)
            self.__num_kernels -= idxs.size(0)

        def __reduce(idxs):
            self.__model["kernel"].reduce(idxs)
            self.__model["linear"].reduce(idxs)
            self.__num_kernels -= idxs.size(0)

        if kwargs["mtol"] is not None:
            idxs_merge = self.__model["kernel"].merge_idxs(**kwargs)
            __merge(idxs_merge)
        if kwargs["rtol"] is not None:
            idxs_reduce = self.__model["linear"].reduce_idxs(**kwargs)
            __reduce(idxs_reduce)
