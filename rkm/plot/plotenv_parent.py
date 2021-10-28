"""
Plotting solutions for a deep RKM model.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: July 2021
"""

from abc import abstractmethod, ABCMeta

import rkm.model.rkm as RKM
import rkm.model.opt as OPT

class plotenv_parent(metaclass=ABCMeta):
    def __init__(self, model: RKM, opt: OPT.Optimizer):
        pass

    @abstractmethod
    def _hyperparameters(self, best):
        pass

    @abstractmethod
    def update(self, iter, tr_mse=None, val_mse=None, test_mse=None, es=0) -> None:
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def finish(self, best_tr, best_val, best_test):
        pass