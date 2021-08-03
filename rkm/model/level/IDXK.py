"""
Management of the stochastic training of the kernels.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: August 2021
"""
import rkm

import random
import numpy as np

class IDXK():
    @rkm.kwargs_decorator({
        "stochastic": 1.
    })
    def __init__(self, **kwargs):
        self._stochastic = kwargs["stochastic"]
        self._idx_kernels = None
        self._init_kernels = kwargs["init_kernels"]
        self._num_kernels = kwargs["init_kernels"]
        self._stoch_kernels = None
        self.new()

    @property
    def stoch_kernels(self):
        return self._stoch_kernels

    @property
    def all_kernels(self):
        return range(self._num_kernels)

    @property
    def num_kernels(self):
        return self._num_kernels

    @property
    def idx_kernels(self):
        return self._idx_kernels

    def _restoch(self):
        self._stoch_kernels = random.sample(self.all_kernels, k=np.maximum(
            int(self._stochastic * self._init_kernels), self.num_kernels))

    def new(self):
        if self._stochastic < 1:
            self._restoch()
            self._idx_kernels = self.stoch_kernels
        else:
            self._idx_kernels = self.all_kernels

    def reduce(self, idxs):
        num = idxs.len(0)
        self._num_kernels -= num

    def merge(self, idxs):
        self.reduce(idxs)
