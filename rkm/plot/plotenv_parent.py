"""
Plotting solutions for a deep RKM model.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: July 2021
"""

import torch

from abc import abstractmethod, ABCMeta
import socket
from datetime import datetime
import os

import rkm.model.rkm as RKM
import rkm.model.opt as OPT

class plotenv_parent(metaclass=ABCMeta):
    def __init__(self, model: RKM, opt: OPT.Optimizer):
        self.model = model
        self.opt = opt

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.id = current_time + '_' + socket.gethostname()
        self.expe_name = self.model.name
        self.proj_dir = os.path.join("runs", self.expe_name)

        self.save_dir = os.path.join(self.proj_dir, "torch")
        self.save_path = os.path.join(self.save_dir, self.id + '.pt')

    @property
    def names(self):
        return self.expe_name, self.proj_dir, self.id

    @abstractmethod
    def _hyperparameters(self):
        pass

    @abstractmethod
    def update(self, iter, tr_mse=None, val_mse=None, test_mse=None, es=0) -> None:
        pass

    @abstractmethod
    def save_model(self, best_tr, best_val, best_test):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.model.state_dict(), self.save_path)
        return self.save_path

    @abstractmethod
    def finish(self, best_tr, best_val, best_test):
        pass