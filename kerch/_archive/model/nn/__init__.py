"""
NN Level

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: August 2021
"""

import torch
from abc import ABCMeta

import kerch
from kerch._archive.model.level import Level

class NN(Level, metaclass=ABCMeta):
    @kerch.kwargs_decorator(
        {"size_in": 1, "size_out": 1, "eta": 1, "representation": "dual", "init_kernels": 1,
         "constraint": "soft", "live_update": True})
    def __init__(self, device='cpu', **kwargs):
        """

        :param representation: "primal" or "dual" representation (default "dual").
        :param init_kernels: number of suppor vectors / kernel to be instantiated.
        :param type: Type of Level ("hard" or "soft").
        :param live_update: Live update of the value of the kernel (default True).
        """
        super(Level, self).__init__()
        self.device = device

        self._size_in = kwargs["size_in"]
        self._size_out = kwargs["size_out"]
        self._eta = kwargs["eta"]
        self._centering = kwargs["centering"]

        self._init_kernels = kwargs["init_kernels"]
        self._idxk = IDXK(**kwargs)
        self._live_update = kwargs["live_update"]
        self._representation = kwargs["representation"]
        self._model = torch.nn.ModuleDict({})

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
        switcher = {"primal": PrimalLinear,
                    "dual": DualLinear}

        if kwargs["representation"] not in switcher:
            raise kerch.model.RepresentationError
        linear = switcher[kwargs["representation"]](**kwargs)

        kernel_kwargs = {**kwargs["kernel"], **{"init_kernels": self._init_kernels,
                                                "size_in": kwargs["size_in"]}}

        self._model = torch.nn.ModuleDict({
            "kernel": kerch.model.kernel.KernelFactory.KernelFactory.create(**kernel_kwargs),
            "linear": linear})

    def forward(self, x, y):
        idx_kernels = self._idxk.idx_kernels
        if self._live_update: self.kernel.update_kernels(x)
        self.hard(x, y)
        x = self.kernel(x, self._representation)
        x = self.linear(x, idx_kernels)
        return x

    def evaluate(self, x):
        # Out-of-sample
        idx_kernels = self._idxk._all_sample
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

    def get_params(self):
        pass

    def solve(self, x, y):
        pass

    def primal(self, x, y):
        pass

    def dual(self, x, y):
        pass

    def hard(self, x, y):
        pass

    def projection(self):
        pass

    def kernels_init(self, x=None):
        pass

    def stoch_update(self):
        pass

    def reset(self):
        pass

    @kerch.kwargs_decorator({"mtol": 1.0e-2, "rtol": 1.0e-4})
    def aggregate(self, **kwargs):
        """
        Aggregates the kernel.

        :param mtol: Merges the kernel if the value is not 0.
        :param rtol: Reduces the kernel if the value is not 0.
        """
        raise NotImplemented