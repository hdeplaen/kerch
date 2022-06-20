"""
Abstract RKM Level class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import kerch
import torch
from abc import ABCMeta, abstractmethod

import kerch.kernel.factory as KernelFactory
import kerch._archive.model.level.DualLinear as DualLinear
import kerch._archive.model.level.PrimalLinear as PrimalLinear
import kerch._archive.model.level.IDXK as IDXK

class Level(torch.nn.Module, metaclass=ABCMeta):
    @kerch.kwargs_decorator(
        {"size_in": 1,
         "size_out": 1,
         "eta": 1,
         "representation": "dual",
         "constraint": "hard",
         "live_update": False,
         "classifier": False})
    def __init__(self, device='cpu', **kwargs):
        """

        :param representation: "primal" or "dual" representation (default "dual").
        :param init_kernels: number of support vectors / kernel to be instantiated.
        :param type: Type of Level ("hard" or "soft").
        :param live_update: Live update of the value of the kernel (default True).
        """
        super(Level, self).__init__()
        self.device = device

        self._size_in = kwargs["size_in"]
        self._size_out = kwargs["size_out"]
        self._eta = kwargs["eta"]
        self._classifier = kwargs["classifier"]

        self._init_kernels = kwargs["init_kernels"]
        self._idxk = IDXK.IDXK(**kwargs)
        self._live_update = kwargs["live_update"]
        self._representation = kwargs["representation"]
        self._model = torch.nn.ModuleDict({})

        self._y = torch.nn.Parameter(torch.ones((kwargs["init_kernels"], kwargs["size_out"]), dtype=kerch.ftype),
                                     requires_grad=False)

        self._input = None
        self._output = None

        self._generate_model(**kwargs)

    @property
    def hparams(self):
        return {"Representation": self._representation,
                "Weight": self._eta,
                "Size in": self._size_in,
                "Size out": self._size_out}

    @property
    def size_out(self):
        return self._size_out

    @property
    def size_in(self):
        return self._size_in

    @abstractmethod
    def __str__(self):
        return f"[{str(self.size_in)}, {str(self.size_out)}]"

    def __repr__(self):
        return self._str__()

    def _generate_model(self, **kwargs):
        # MODEL
        switcher = {"primal": PrimalLinear.PrimalLinear,
                    "dual": DualLinear.DualLinear}

        if kwargs["representation"] not in switcher:
            raise kerch.model.RepresentationError
        linear = switcher[kwargs["representation"]](**kwargs)

        kernel_kwargs = {**kwargs["kernel"], **{"init_kernels": self._init_kernels,
                                                "size_in": kwargs["size_in"]}}

        self._model = torch.nn.ModuleDict({
            "kernel": KernelFactory.KernelFactory.create(**kernel_kwargs),
            "linear": linear})

    def init(self, x, y):
        self._y[self._idxk.all_kernels] = y.data
        self.linear.init_y(y, self._idxk.all_kernels)

    def forward(self, x, y, init=False):
        idx_kernels = self._idxk.idx_kernels
        if self._live_update and not init:
            self.kernel.update_kernels(x, self._idxk.idx_update)
        self.hard(x, y)
        x = self.kernel(x, self._representation)
        x = self.linear(x, idx_kernels)
        return x

    def evaluate(self, x, all_kernels=False):
        # Out-of-sample
        if all_kernels:
            idx_kernels = self._idxk.all_kernels
        else:
            idx_kernels = self._idxk.idx_kernels


        x = self.kernel(x, self._representation, idx_kernels)
        x = self.linear(x, idx_kernels)
        return x

    @abstractmethod
    def loss(self, x=None, y=None):
        pass

    @property
    def eta(self):
        return self._eta

    @property
    def representation(self) -> str:
        return self._representation

    @property
    def linear(self):
        return self._model["linear"]

    @property
    def kernel(self):
        return self._model["kernel"]

    @property
    def w(self):
        if self.representation is "primal":
            return self.linear.weight
        else:
            return NotImplemented

    @property
    def h(self):
        if self.representation is "dual":
            return self.linear.alpha
        else:
            return NotImplemented


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

    # @abstractmethod
    def projection(self):
        pass

    def hard(self, x, y):
        pass

    def kernels_init(self, x=None):
        self.kernel.kernels_init(x)
        self.kernels_initialized = True

    def stoch_update(self):
        self._idxk.new_level()
        return self._idxk.idx_kernels

    def init_idxk(self, idxk: IDXK):
        self._idxk = idxk

    def reset(self):
        idx_kernels = self.stoch_update()
        self.kernel.reset(idx_kernels)

    @kerch.kwargs_decorator({"mtol": 1.0e-2, "rtol": 1.0e-4})
    def aggregate(self, **kwargs):
        """
        Aggregates the kernel.

        :param mtol: Merges the kernel if the value is not 0.
        :param rtol: Reduces the kernel if the value is not 0.
        """
        assert self._size_out == 1, NotImplemented

        def merge(idxs):
            self.kernel.merge(idxs)
            self.linear.merge(idxs)
            self._idxk.merge(idxs)

        def reduce(idxs):
            self.kernel.reduce(idxs)
            self.linear.reduce(idxs)
            self._idxk.reduce(idxs)

        if kwargs["mtol"] is not None:
            idxs_merge = self.kernel.merge_idxs(**kwargs)
            merge(idxs_merge)
        if kwargs["rtol"] is not None:
            idxs_reduce = self.linear.reduce_idxs(**kwargs)
            reduce(idxs_reduce)
