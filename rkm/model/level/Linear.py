"""
Abstract linear class for a RKM level.

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
        self.__soft = {"soft": True, "hard": False}.get(kwargs["type"], "Invalid LSSVM type (must be hard or soft).")
        self.__requires_bias = kwargs["requires_bias"]

    @abstractmethod
    def set(self, a, b):
        pass

    @abstractmethod
    def forward(self, x, idx_sv):
        pass

    @abstractmethod
    def project(self):
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