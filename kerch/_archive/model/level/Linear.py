"""
Abstract linear class for a RKM Level.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from abc import ABCMeta, abstractmethod

class Linear(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(Linear, self).__init__()

        switcher = {"soft": True, "hard": False}
        type = kwargs["constraint"]
        if type not in switcher:
            raise NameError("Invalid constraint (must be hard or soft).")

        self._soft = switcher[type]
        self._requires_bias = kwargs["requires_bias"]

    @abstractmethod
    def set(self, a, b):
        pass

    @abstractmethod
    def forward(self, x, idx_sv):
        pass

    @abstractmethod
    def project(self):
        pass

    def init_y(self, y, idxs):
        pass

    @abstractmethod
    def merge(self, idxs):
        pass

    @abstractmethod
    def reduce(self, idxs):
        pass

    @abstractmethod
    def reduce_idxs(self, **kwargs):
        pass