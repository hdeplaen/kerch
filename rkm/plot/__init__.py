"""
Plotting solutions for a deep RKM model.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: October 2021
"""

from abc import abstractmethod

import rkm as rkm
import rkm.plot.plotenv_parent as plotenv_parent
import rkm.plot.plotenv_wandb as plotenv_wandb
import rkm.plot.plotenv_tensorboard as plotenv_tensorboard

import rkm.model.rkm as RKM
import rkm.model.opt as OPT

class plotenv(plotenv_parent.plotenv_parent):
    def __init__(self, model: RKM, opt: OPT.Optimizer):
        super(plotenv_parent.plotenv_parent, self).__init__()
        if rkm.PLOT_ENV=='wandb':
            self = plotenv_wandb.plotenv_wandb(model, opt)
        elif rkm.PLOT_ENV=='tensorboard':
            self = plotenv_tensorboard.plotenv_tensorboard(model, opt)
        else:
            Warning('Plot environment not recognized. No plotting will occur.')

    def _hyperparameters(self, best):
        pass

    def update(self, iter, tr_mse=None, val_mse=None, test_mse=None, es=0) -> None:
        pass

    def save_model(self):
        pass

    def finish(self, best_tr, best_val, best_test):
        pass