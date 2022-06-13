"""
Plotting solutions for a deep RKM _src.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: July 2021
"""

import torch

from abc import ABCMeta
import socket
from datetime import datetime
import os


class plotenv_parent(metaclass=ABCMeta):
    def __init__(self, model: rkm, opt: opt.Optimizer):
        self.model = model
        self.opt = opt

        current_time = datetime.now().strftime('%b%d_%_H-%M-%S')
        self.id = current_time + '_' + socket.gethostname()
        self.expe_name = self.model.name
        self.proj_dir = os.path.join("runs", self.expe_name)

        self.save_dir = os.path.join(self.proj_dir, "torch", self.id)

    @property
    def names(self):
        return self.expe_name, self.proj_dir, self.id

    def _hyperparameters(self):
        pass

    def update(self, iter, tr_mse=None, val_mse=None, test_mse=None, es=0) -> None:
        pass

    def save_model(self, iter, best_tr, best_val, best_test):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_name = 'iter-' + str(iter) + '.pt'
        self.save_path = os.path.join(self.save_dir, self.save_name)
        torch.save(self.model.state_dict(), self.save_path)
        return self.save_path

    def finish(self, best_tr, best_val, best_test):
        # clean all checkpoints but last
        filelist = [f for f in os.listdir(self.save_dir) if not self.save_name]
        for f in filelist:
            os.remove(os.path.join(self.save_dir, f))
        return self.save_path