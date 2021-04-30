"""
Abstract RKM level class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm
import torch
from abc import ABCMeta, abstractmethod
import random
import numpy as np

from .DualLinear import DualLinear
from .PrimalLinear import PrimalLinear

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

        self._size_in = kwargs["size_in"]
        self._size_out = kwargs["size_out"]
        self._eta = kwargs["eta"]

        self._init_kernels = kwargs["init_kernels"]
        self._num_kernels = kwargs["init_kernels"]
        self._live_update = kwargs["live_update"]
        self._representation = kwargs["representation"]
        self._model = torch.nn.ModuleDict({})

        self._generate_model(**kwargs)

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self._str__()

    def _generate_model(self, **kwargs):
        # MODEL
        switcher = {"primal": PrimalLinear,
                    "dual": DualLinear}

        if kwargs["representation"] not in switcher:
            raise rkm.model.RepresentationError
        linear = switcher[kwargs["representation"]](**kwargs)

        self._model = torch.nn.ModuleDict({
            "kernel": rkm.model.kernel.KernelFactory.KernelFactory.create(**kwargs["kernel"]),
            "linear": linear})

    def forward(self, x, idx_kernels=None):
        if idx_kernels is None: idx_kernels = self._all_kernels
        if self._live_update: self._model["kernel"].update(x)
        x = self._model["kernel"](x, self._representation, idx_kernels)
        x = self._model["linear"](x, idx_kernels)
        return x

    @abstractmethod
    def loss(self, x, y=None):
        pass

    @property
    def eta(self):
        return self._eta

    @property
    def num_kernels(self):
        return self._num_kernels

    @property
    def representation(self) -> str:
        return self._representation

    @property
    def linear(self):
        return self._model["linear"]

    @property
    def kernel(self):
        return self._model["kernel"]

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
        self._model["kernel"].kernels_init(x)
        self.kernels_initialized = True

    @property
    def __all_kernels(self):
        return range(self._num_kernels)

    @property
    def __stoch_kernels(self):
        if self._stochastic < 1.:
            idx = random.choices(self.all_kernels, k=np.maximum(
                torch.round(self._stochastic * self.init_kernels), self.num_kernels))
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
        assert self._size_out == 1, NotImplementedError

        def __merge(idxs):
            self._model["kernel"].merge(idxs)
            self._model["linear"].merge(idxs)
            self._num_kernels -= idxs.size(0)

        def __reduce(idxs):
            self._model["kernel"].reduce(idxs)
            self._model["linear"].reduce(idxs)
            self._num_kernels -= idxs.size(0)

        if kwargs["mtol"] is not None:
            idxs_merge = self._model["kernel"].merge_idxs(**kwargs)
            __merge(idxs_merge)
        if kwargs["rtol"] is not None:
            idxs_reduce = self._model["linear"].reduce_idxs(**kwargs)
            __reduce(idxs_reduce)
